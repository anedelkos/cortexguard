import asyncio
import logging
from datetime import datetime
from typing import Any

from kitchenwatch.core.interfaces.base_controller import BaseController
from kitchenwatch.core.interfaces.base_executor import BaseExecutor
from kitchenwatch.core.interfaces.base_step_classifier import BaseStepClassifier
from kitchenwatch.edge.models.anomaly_event import AnomalySeverity
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.capability_registry import CapabilityRegistry
from kitchenwatch.edge.models.plan import PlanStep, StepStatus
from kitchenwatch.edge.models.reasoning_trace_entry import ReasoningTraceEntry

logger = logging.getLogger(__name__)

EXECUTOR_POLL_INTERVAL = 0.05
EXECUTOR_IDLE_INTERVAL = 0.1
DEFAULT_MAX_PLAN_FAILURES = 3
DEFAULT_RETRY_DELAY = 0.5


class StepExecutor(BaseExecutor):
    """
    Executes PlanSteps by passing the AgentToolCall directly to the controller.
    It manages execution state, retry logic, and anomaly checks.
    """

    def __init__(
        self,
        blackboard: Blackboard,
        step_classifier: BaseStepClassifier,
        capability_registry: CapabilityRegistry,
        controller: BaseController,
        default_max_retries: int = DEFAULT_MAX_PLAN_FAILURES,
        default_retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        self._blackboard = blackboard
        self._step_classifier = step_classifier
        self._capability_registry = capability_registry
        self._controller = controller
        self._default_max_retries = default_max_retries
        self._default_retry_delay = default_retry_delay
        self._loop_running = False
        self._paused = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._loop_running:
            return
        if self._task and not self._task.done():
            return

        self._loop_running = True
        self._task = asyncio.create_task(self._executor_loop())

    async def stop(self) -> None:
        self._loop_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def pause(self) -> None:
        self._paused = True

    async def resume(self) -> None:
        self._paused = False

    async def _execute_direct_call(self, function_name: str, arguments: dict[str, Any]) -> bool:
        # 1. Validate the command against the registry schema (Enforces enum/required fields)
        try:
            self._capability_registry.validate_call(function_name, arguments)
        except Exception as e:
            await self._post_trace_entry(
                source="StepExecutor",
                event_type="VALIDATION_FAILED",
                reasoning_text=f"Command for {function_name} invalid: {e}",
                metadata={"function": function_name},
            )
            return False

        # 2. Execute the function
        try:
            await self._controller.execute(function_name, arguments)
            return True
        except Exception as e:
            await self._post_trace_entry(
                source="StepExecutor",
                event_type="EXECUTION_FAILED",
                reasoning_text=f"Controller failed on {function_name}: {e}",
                metadata={"function": function_name},
            )
            return False

    async def _post_trace_entry(
        self, source: str, event_type: str, reasoning_text: str, metadata: dict[str, Any]
    ) -> None:
        entry = ReasoningTraceEntry(
            timestamp=datetime.now(),
            source=source,
            event_type=event_type,
            reasoning_text=reasoning_text,
            metadata=metadata,
        )
        await self._blackboard.add_trace_entry(entry)

    async def execute_step(self, step: PlanStep) -> None:
        if step.status != StepStatus.PENDING:
            return

        if await self._blackboard.is_anomaly_present(severity_min=AnomalySeverity.MEDIUM):
            return

        max_attempts = (
            step.max_retries if hasattr(step, "max_retries") else self._default_max_retries
        )
        retry_delay = (
            step.retry_delay_s if hasattr(step, "retry_delay_s") else self._default_retry_delay
        )

        function_name = step.action.action_name
        arguments = step.action.arguments

        for attempt in range(1, max_attempts + 1):
            step.status = StepStatus.RUNNING
            step.attempts = attempt

            execution_success = await self._execute_direct_call(function_name, arguments)

            if execution_success:
                try:
                    completion_status = self._step_classifier.classify_completion_status(step)
                except Exception:
                    completion_status = StepStatus.FAILED
            else:
                completion_status = StepStatus.FAILED

            if completion_status == StepStatus.COMPLETED:
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now()
                await self._post_trace_entry(
                    source="StepExecutor",
                    event_type="STEP_COMPLETED",
                    reasoning_text=f"Step {step.id} successfully completed.",
                    metadata={"step_id": step.id},
                )
                return

            if attempt < max_attempts:
                step.status = StepStatus.PENDING
                await asyncio.sleep(retry_delay)
                continue
            else:
                step.status = StepStatus.FAILED
                step.completed_at = datetime.now()
                await self._post_trace_entry(
                    source="StepExecutor",
                    event_type="STEP_FAILED",
                    reasoning_text=f"Step {step.id} permanently failed.",
                    metadata={"step_id": step.id},
                )
                return

    async def _executor_loop(self) -> None:
        while self._loop_running:
            if self._paused:
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
                continue

            if await self._blackboard.is_anomaly_present(severity_min=AnomalySeverity.MEDIUM):
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
                continue

            step = await self._blackboard.get_current_step()
            if step and step.status == StepStatus.PENDING:
                await self.execute_step(step)
                await asyncio.sleep(EXECUTOR_POLL_INTERVAL)
            else:
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
