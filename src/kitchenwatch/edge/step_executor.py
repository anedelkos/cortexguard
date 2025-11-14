import asyncio
import logging

from kitchenwatch.core.action_registry import ActionRegistry
from kitchenwatch.core.interfaces.base_executor import BaseExecutor
from kitchenwatch.core.interfaces.base_step_classifier import BaseStepClassifier
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
        custom_logger: logging.Logger = logger,
    ) -> None:
        self._blackboard = blackboard
        self._step_classifier = step_classifier
        self._action_registry = action_registry
        self._default_max_retries = default_max_retries
        self._default_retry_delay = default_retry_delay
        self._loop_running = False
        self._paused = False
        self._task: asyncio.Task[None] | None = None
        self._logger = custom_logger

    async def start(self) -> None:
        if self._loop_running:
            self._logger.warning("StepExecutor already running")
            return
        if self._task and not self._task.done():
            self._logger.warning("Previous task still active")
            return

        self._loop_running = True
        self._task = asyncio.create_task(self._executor_loop())
        self._logger.info("StepExecutor started.")

    async def stop(self) -> None:
        self._loop_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                self._logger.debug("Executor task cancelled")
        self._logger.info("StepExecutor stopped.")

    async def pause(self) -> None:
        self._paused = True
        self._logger.info("StepExecutor paused.")

    async def resume(self) -> None:
        self._paused = False
        self._logger.info("StepExecutor resumed.")

    async def execute_step(self, step: PlanStep) -> None:
        """
        Execute a single PlanStep using:
        - PyTrees behavior leaf
        - StepClassifier for success/failure
        - retry logic with delay
        - running polling loop
        """

        if step.status != StepStatus.PENDING:
            self._logger.debug(f"Skipping step {step.id} with status {step.status}")
            return

        max_attempts = getattr(step, "max_retries", self._default_max_retries)
        retry_delay = getattr(step, "retry_delay_s", self._default_retry_delay)

        for attempt in range(1, max_attempts + 1):
            step.status = StepStatus.RUNNING
            try:
                sequence = self._action_registry.build(step.name)
            except KeyError:
                self._logger.error(f"Step '{step.name}' not registered in ActionRegistry")
                step.status = StepStatus.FAILED
                return

            sequence.setup(timeout=0)

            try:
                sequence.tick_once()
                completion_status = self._step_classifier.classify_completion_status(step)
            except Exception as e:
                self._logger.exception(f"⚠️ Unexpected execution error in step {step.id}: {e}")
                completion_status = StepStatus.FAILED
            finally:
                sequence.shutdown()

            if completion_status == StepStatus.COMPLETED:
                step.status = StepStatus.COMPLETED
                self._logger.info(f"✅ Step {step.id}.{step.name} completed on attempt {attempt}")
                return

            elif completion_status == StepStatus.RUNNING:
                self._logger.debug(
                    f"⏳ Step {step.id}.{step.name} still running (attempt {attempt})"
                )
                await asyncio.sleep(STEP_RUNNING_POLL_INTERVAL)
                continue

            else:  # FAILED
                if attempt < max_attempts:
                    self._logger.warning(
                        f"⚠️ Step {step.id}.{step.name} failed on attempt {attempt}, retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # Out of retries → permanent failure
                step.status = StepStatus.FAILED
                self._logger.error(
                    f"❌ Step {step.id}.{step.name} permanently failed after {attempt} attempts"
                )
                return

    async def _executor_loop(self) -> None:
        """
        Watches blackboard for new steps and executes them.
        """
        while self._loop_running:
            if self._paused:
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
                continue

            step = await self._blackboard.get_current_step()
            if step and step.status == StepStatus.PENDING:
                await self.execute_step(step)
                await asyncio.sleep(EXECUTOR_POLL_INTERVAL)
            else:
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
