import asyncio
import logging
from typing import Any

from kitchenwatch.core.action_registry import ActionRegistry
from kitchenwatch.core.interfaces.base_executor import BaseExecutor
from kitchenwatch.core.interfaces.base_step_classifier import BaseStepClassifier
from kitchenwatch.edge.models.action import ActionStatus
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
    Executes PlanSteps by iterating through their primitive actions directly.
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

    async def _execute_primitive_sequence(self, capability_name: str) -> bool:
        """
        Retrieves and executes the sequence of primitives for a given capability.
        Returns True if all primitives succeed, False otherwise.
        """
        try:
            # 1. Retrieve the sequence of primitives (dictionaries)
            primitives = self._action_registry.get_primitives_for_capability(capability_name)
        except KeyError:
            logger.error(f"Capability '{capability_name}' not registered.")
            return False

        # 2. Iterate and execute
        for i, primitive_data in enumerate(primitives):
            # Safety Check between primitives
            if await self._blackboard.is_anomaly_present(severity_min=AnomalySeverity.MEDIUM):
                logger.warning(f"Sequence interrupted by anomaly before primitive {i}.")
                return False

            primitive_name = primitive_data.get("primitive")
            parameters: dict[str, Any] = primitive_data.get("parameters", {})

            if not primitive_name:
                logger.error(f"Invalid primitive definition at index {i}")
                return False

            try:
                logger.debug(f"Executing primitive: {primitive_name} with {parameters}")
                # Accessing the controller directly from registry (assuming it exposes it or we use private)
                # Better pattern: ActionRegistry could expose an 'execute' method, but this works for now.
                await self._action_registry._controller.execute(primitive_name, parameters)
            except Exception as e:
                logger.error(f"Primitive '{primitive_name}' failed: {e}")
                return False

        return True

    async def execute_step(self, step: PlanStep) -> None:
        """
        Execute a single PlanStep.
        """
        if step.status != StepStatus.PENDING:
            logger.debug(f"Skipping step {step.id} with status {step.status}")
            return

        # Initial Safety Check
        if await self._blackboard.is_anomaly_present(severity_min=AnomalySeverity.MEDIUM):
            logger.warning(f"🛑 Cannot start step {step.id}. Execution blocked by anomaly.")
            return

        max_attempts = (
            getattr(step, "max_retries", self._default_max_retries) or self._default_max_retries
        )
        retry_delay = (
            getattr(step, "retry_delay_s", self._default_retry_delay) or self._default_retry_delay
        )

        action_model = step.action
        capability_name = action_model.capability

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Executing step '{step.id}' (Attempt {attempt}/{max_attempts})")

            # Update statuses to Running
            step.status = StepStatus.RUNNING
            action_model.status = ActionStatus.EXECUTING

            # 1. Run the sequence of primitives directly
            sequence_success = await self._execute_primitive_sequence(capability_name)

            if not sequence_success:
                action_model.status = ActionStatus.FAILED
                # Don't return yet, check if we can retry
            else:
                action_model.status = ActionStatus.COMPLETED

            # 2. Classify completion (Did the action actually achieve the goal?)
            # Even if primitives ran fine, the sensor check might fail.
            if sequence_success:
                try:
                    completion_status = self._step_classifier.classify_completion_status(step)
                except Exception as e:
                    logger.exception(f"Classifier error for step {step.id}: {e}")
                    completion_status = StepStatus.FAILED
            else:
                completion_status = StepStatus.FAILED

            # 3. Handle Outcome
            if completion_status == StepStatus.COMPLETED:
                step.status = StepStatus.COMPLETED
                logger.info(f"✅ Step {step.id} completed on attempt {attempt}")
                return

            # Handle Failure / Retry
            logger.warning(f"⚠️ Step {step.id} failed on attempt {attempt}.")

            if attempt < max_attempts:
                step.status = StepStatus.PENDING  # Reset for next loop
                action_model.status = ActionStatus.PENDING
                await asyncio.sleep(retry_delay)
                continue
            else:
                step.status = StepStatus.FAILED
                logger.error(f"❌ Step {step.id} permanently failed after {max_attempts} attempts")
                return

    async def _executor_loop(self) -> None:
        """
        Watches blackboard for new steps and executes them.
        """
        while self._loop_running:
            if self._paused:
                logger.debug("Paused by user/external command.")
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
                continue

            if await self._blackboard.is_anomaly_present(severity_min=AnomalySeverity.MEDIUM):
                logger.warning("🛑 Execution paused by MEDIUM+ anomaly flag.")
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
                continue

            step = await self._blackboard.get_current_step()
            if step and step.status == StepStatus.PENDING:
                await self.execute_step(step)
                await asyncio.sleep(EXECUTOR_POLL_INTERVAL)
            else:
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
