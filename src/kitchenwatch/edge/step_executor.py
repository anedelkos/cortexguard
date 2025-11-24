import asyncio
import logging

import py_trees

from kitchenwatch.core.action_registry import ActionRegistry
from kitchenwatch.core.interfaces.base_executor import BaseExecutor
from kitchenwatch.core.interfaces.base_step_classifier import BaseStepClassifier
from kitchenwatch.edge.models.anomaly_severity import AnomalySeverity
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.plan import PlanStep, StepStatus

logger = logging.getLogger(__name__)

# Configuration constants
EXECUTOR_POLL_INTERVAL = 0.05
EXECUTOR_IDLE_INTERVAL = 0.1
STEP_RUNNING_POLL_INTERVAL = 0.05
DEFAULT_MAX_PLAN_FAILURES = 3
DEFAULT_RETRY_DELAY = 0.5


class StepExecutor(BaseExecutor):
    """
    Executor that runs PlanSteps using PyTrees behaviors.
    Watches the blackboard and executes steps asynchronously.
    Supports pause/resume, retry logic, and consulting a StepClassifier.
    """

    def __init__(
        self,
        blackboard: Blackboard,
        step_classifier: BaseStepClassifier,
        action_registry: ActionRegistry,
        default_max_retries: int = DEFAULT_MAX_PLAN_FAILURES,
        default_retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        self._blackboard = blackboard
        self._step_classifier = step_classifier
        self._action_registry = action_registry
        self._default_max_retries = default_max_retries
        self._default_retry_delay = default_retry_delay
        self._loop_running = False
        self._paused = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._loop_running:
            logger.warning("StepExecutor already running")
            return
        if self._task and not self._task.done():
            logger.warning("Previous task still active")
            return

        self._loop_running = True
        self._task = asyncio.create_task(self._executor_loop())
        logger.info("StepExecutor started.")

    async def stop(self) -> None:
        self._loop_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("Executor task cancelled")
        logger.info("StepExecutor stopped.")

    async def pause(self) -> None:
        self._paused = True
        logger.info("StepExecutor paused.")

    async def resume(self) -> None:
        self._paused = False
        logger.info("StepExecutor resumed.")

    async def _run_tree_to_completion(
        self, root: py_trees.behaviour.Behaviour
    ) -> py_trees.common.Status:
        """
        Ticks a behavior tree continuously until it returns SUCCESS, FAILURE,
        or is externally terminated. This is the core logic that allows async
        actions (like the controller call) to complete.
        """

        tree = py_trees.trees.BehaviourTree(root)

        # 1. Setup the Tree
        tree.setup(timeout=15.0)

        try:
            # Tick once to initiate the sequence and move status to RUNNING (or instantly SUCCESS/FAILURE)
            tree.tick()

            # CRITICAL YIELD: Ensure the task created in PrimitiveLeaf gets scheduled and runs
            # immediately, which is vital for instantly-resolving AsyncMocks in tests.
            await asyncio.sleep(0)

            # 2. Continuous Tick Loop
            # The loop is necessary to process the RUNNING -> SUCCESS transition
            while tree.root.status == py_trees.common.Status.RUNNING:
                # Check for critical anomaly on every tick (prevents the loop from running indefinitely)
                if await self._blackboard.is_anomaly_present(severity_min=AnomalySeverity.MEDIUM):
                    logger.warning("Step execution interrupted by critical anomaly.")
                    # Return INVALID status to signify external termination/pause
                    return py_trees.common.Status.INVALID

                tree.tick()
                # Yield control to allow async action (controller.execute) to complete
                await asyncio.sleep(STEP_RUNNING_POLL_INTERVAL)

            return tree.root.status

        except Exception as e:
            logger.exception(f"Internal PyTree execution failure: {e}")
            return py_trees.common.Status.FAILURE

        finally:
            # 3. Always terminate the tree to clean up resources
            tree.root.terminate(tree.root.status)

    async def execute_step(self, step: PlanStep) -> None:
        """
        Execute a single PlanStep using:
        - PyTrees behavior leaf (run to completion)
        - StepClassifier for success/failure
        - retry logic with delay
        """

        if step.status != StepStatus.PENDING:
            logger.debug(f"Skipping step {step.id} with status {step.status}")
            return

        # Before starting execution, check for critical anomalies
        if await self._blackboard.is_anomaly_present(
            key_prefix=None, severity_min=AnomalySeverity.MEDIUM
        ):
            logger.warning(
                f"🛑 Cannot start step {step.id}. Execution is blocked by a MEDIUM+ anomaly flag."
            )
            return

        max_attempts = getattr(step, "max_retries", self._default_max_retries)
        retry_delay = getattr(step, "retry_delay_s", self._default_retry_delay)

        for attempt in range(1, max_attempts + 1):
            step.status = StepStatus.RUNNING
            try:
                # 1. Build the behaviour (The behaviour is a py_trees.Sequence)
                behaviour = self._action_registry.build(step.name)
            except KeyError:
                logger.error(f"Step '{step.name}' not registered in ActionRegistry")
                step.status = StepStatus.FAILED
                return

            # 2. Run the tree until the PrimitiveLeaf returns SUCCESS or FAILURE
            final_tree_status = await self._run_tree_to_completion(behaviour)

            if final_tree_status == py_trees.common.Status.INVALID:
                # Execution was interrupted by an anomaly during the tick loop
                logger.info(f"Step {step.id} interrupted and returned to PENDING status.")
                step.status = StepStatus.PENDING
                return

            if final_tree_status == py_trees.common.Status.FAILURE:
                logger.warning(f"PyTree failed internally for step {step.id}.{step.name}")
                # Fall through to classifier check

            # 3. Classify the final observed state
            try:
                completion_status = self._step_classifier.classify_completion_status(step)
            except Exception as e:
                logger.exception(f"⚠️ Classifier error for step {step.id}: {e}")
                completion_status = StepStatus.FAILED  # Default to failed if classifier crashes

            if completion_status == StepStatus.COMPLETED:
                # SUCCESS: Final state matches desired post-condition
                step.status = StepStatus.COMPLETED
                logger.info(f"✅ Step {step.id}.{step.name} completed on attempt {attempt}")
                return

            # If classifier returns RUNNING or FAILED:
            if attempt < max_attempts:
                logger.warning(
                    f"⚠️ Step {step.id}.{step.name} failed/stalled on attempt {attempt}, retrying..."
                )
                step.status = StepStatus.PENDING  # Ready for next attempt
                await asyncio.sleep(retry_delay)
                continue

            # Out of retries → permanent failure
            step.status = StepStatus.FAILED
            logger.error(
                f"❌ Step {step.id}.{step.name} permanently failed after {attempt} attempts"
            )
            return

    async def _executor_loop(self) -> None:
        """
        Watches blackboard for new steps and executes them.
        Includes safety check for critical anomalies.
        """
        while self._loop_running:
            # 1. Check for manual pause
            if self._paused:
                logger.debug("Paused by user/external command.")
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
                continue

            # 2. Check for safety pause (Critical Anomaly)
            if await self._blackboard.is_anomaly_present(
                key_prefix=None, severity_min=AnomalySeverity.MEDIUM
            ):
                logger.warning("🛑 Execution paused by MEDIUM+ anomaly flag.")
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
                continue

            # 3. Normal execution path
            step = await self._blackboard.get_current_step()
            if step and step.status == StepStatus.PENDING:
                await self.execute_step(step)
                await asyncio.sleep(EXECUTOR_POLL_INTERVAL)
            else:
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
