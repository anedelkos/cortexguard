from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from cortexguard.core.interfaces.base_controller import BaseController
from cortexguard.core.interfaces.base_executor import BaseExecutor
from cortexguard.core.interfaces.base_step_classifier import BaseStepClassifier
from cortexguard.edge.models.anomaly_event import AnomalySeverity
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.capability_registry import CapabilityRegistry
from cortexguard.edge.models.plan import PlanStep, StepStatus
from cortexguard.edge.models.reasoning_trace_entry import TraceSeverity
from cortexguard.edge.utils.tracing import BaseTraceSink, TraceSink

logger = logging.getLogger(__name__)

EXECUTOR_POLL_INTERVAL = 0.05
EXECUTOR_IDLE_INTERVAL = 0.1


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
        default_max_retries: int,
        default_retry_delay: float,
        default_poll_interval: float,
        default_idle_interval: float,
        trace_sink: BaseTraceSink | None = None,
    ) -> None:
        self._blackboard = blackboard
        self._trace_sink: BaseTraceSink = (
            trace_sink if trace_sink is not None else TraceSink(blackboard=self._blackboard)
        )
        self._step_classifier = step_classifier
        self._capability_registry = capability_registry
        self._controller = controller
        self._max_retries = default_max_retries
        self._retry_delay = default_retry_delay
        self._poll_interval = default_poll_interval
        self._idle_interval = default_idle_interval
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
        # 1. Validate the command against the registry schema (offload if validator is blocking)
        try:
            # validate_call returns (bool, RiskLevel) in our registry
            validate_result = await asyncio.to_thread(
                self._capability_registry.validate_call, function_name, arguments
            )
        except Exception as e:
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="VALIDATION_FAILED",
                reasoning_text=f"Validation raised exception for {function_name}: {e}",
                metadata={"function": function_name, "error": type(e).__name__},
                severity=TraceSeverity.HIGH,
            )
            return False

        # Defensive handling for older implementations that might return None
        if validate_result is None:
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="VALIDATION_FAILED",
                reasoning_text=f"Validation API returned no result for {function_name}; denying execution.",
                metadata={"function": function_name},
                severity=TraceSeverity.HIGH,
            )
            return False

        # Expect (valid: bool, risk: RiskLevel)
        try:
            valid, risk = validate_result
        except Exception:
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="VALIDATION_FAILED",
                reasoning_text=f"Validation returned unexpected shape for {function_name}; denying execution.",
                metadata={"function": function_name, "raw_result": str(validate_result)},
                severity=TraceSeverity.HIGH,
            )
            return False

        # Deny execution for invalid or high-risk calls
        if not valid or getattr(risk, "name", str(risk)).upper() == "HIGH":
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="VALIDATION_FAILED",
                reasoning_text=(
                    f"Command for {function_name} invalid: valid={valid}, risk={getattr(risk, 'name', str(risk))}"
                ),
                metadata={"function": function_name, "arguments": arguments, "risk": str(risk)},
                severity=TraceSeverity.HIGH,
            )
            return False

        # 2. Prepare attempt correlation and post attempt trace
        from uuid import uuid4

        attempt_id = f"attempt-{uuid4().hex[:8]}"
        await self._trace_sink.post_trace_entry(
            source=self,
            event_type="ACTION_ATTEMPT",
            reasoning_text=f"Executing action '{function_name}'",
            metadata={"function": function_name, "arguments": arguments},
            refs={"attempt_id": attempt_id},
        )

        # 3. Execute and handle success, cancellation, and failures with traces
        start = datetime.now(UTC)
        try:
            await self._controller.execute(function_name, arguments)

            duration_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="ACTION_RESULT",
                reasoning_text=f"Action '{function_name}' succeeded",
                metadata={"function": function_name},
                refs={"attempt_id": attempt_id},
                duration_ms=duration_ms,
            )
            return True

        except asyncio.CancelledError:
            # Schedule a best-effort cancellation trace and re-raise to preserve cancellation semantics
            duration_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
            try:
                asyncio.create_task(
                    self._trace_sink.post_trace_entry(
                        source=self,
                        event_type="ACTION_CANCELLED",
                        reasoning_text=f"Action '{function_name}' cancelled",
                        metadata={"function": function_name},
                        refs={"attempt_id": attempt_id},
                        duration_ms=duration_ms,
                    )
                )
            except Exception:
                logger.debug("Failed to schedule cancellation trace", exc_info=True)
            raise

        except Exception as e:
            duration_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
            try:
                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="EXECUTION_FAILED",
                    reasoning_text=f"Controller failed on {function_name}: {e}",
                    metadata={"function": function_name, "error": type(e).__name__},
                    refs={"attempt_id": attempt_id},
                    duration_ms=duration_ms,
                    severity=TraceSeverity.HIGH,
                )
            except Exception:
                logger.debug("Failed to post execution failure trace", exc_info=True)
            return False

    async def execute_step(self, step: PlanStep) -> None:
        # 0. Safety check: respect Arbiter flags
        if await self._blackboard.get_safety_flag("emergency_stop"):
            logger.warning("Execution aborted: emergency_stop flag set")
            step.status = StepStatus.FAILED
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="EXECUTION_ABORTED",
                reasoning_text="Step aborted due to emergency_stop flag",
                metadata={"step_id": step.id},
                severity=TraceSeverity.CRITICAL,
            )
            return

        if step.status != StepStatus.PENDING:
            return

        if await self._blackboard.is_anomaly_present(severity_min=AnomalySeverity.MEDIUM):
            return

        max_attempts = step.max_retries if hasattr(step, "max_retries") else self._max_retries
        retry_delay = step.retry_delay_s if hasattr(step, "retry_delay_s") else self._retry_delay

        function_name = step.action.action_name
        arguments = step.action.arguments

        for attempt in range(1, max_attempts + 1):
            # Safety check before each attempt
            if await self._blackboard.get_safety_flag("emergency_stop"):
                logger.warning("Execution aborted mid-step: emergency_stop flag set")
                step.status = StepStatus.FAILED
                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="EXECUTION_ABORTED",
                    reasoning_text="Step aborted mid-execution due to emergency_stop flag",
                    metadata={"step_id": step.id, "attempt": attempt},
                    severity=TraceSeverity.CRITICAL,
                )
                return

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
                step.completed_at = datetime.now(UTC)
                await self._trace_sink.post_trace_entry(
                    source=self,
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
                step.completed_at = datetime.now(UTC)
                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="STEP_FAILED",
                    reasoning_text=f"Step {step.id} permanently failed.",
                    metadata={"step_id": step.id},
                )
                return

    async def _executor_loop(self) -> None:
        while self._loop_running:
            if await self._blackboard.get_safety_flag("emergency_stop"):
                logger.warning("Executor loop halted: emergency_stop flag set")
                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="EXECUTOR_HALTED",
                    reasoning_text="Executor loop halted due to emergency_stop flag",
                    metadata={},
                    severity=TraceSeverity.CRITICAL,
                )
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
                continue

            if self._paused:
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
                continue

            if await self._blackboard.is_anomaly_present(severity_min=AnomalySeverity.MEDIUM):
                await asyncio.sleep(EXECUTOR_IDLE_INTERVAL)
                continue

            step = await self._blackboard.get_current_step()
            if step and step.status == StepStatus.PENDING:
                await self.execute_step(step)
                await asyncio.sleep(self._poll_interval)
            else:
                await asyncio.sleep(self._idle_interval)
