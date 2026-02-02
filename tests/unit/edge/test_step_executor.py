import asyncio
from typing import Any, Protocol
from unittest.mock import AsyncMock, MagicMock

import pytest

from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.capability_registry import RiskLevel
from cortexguard.edge.models.plan import PlanStep, StepStatus
from cortexguard.edge.step_executor import StepExecutor


class ControllerProtocol(Protocol):
    async def execute(self, primitive_name: str, parameters: dict[str, Any]) -> None: ...


class CapabilityRegistryProtocol(Protocol):
    def validate_call(self, function_name: str, arguments: dict[str, Any]) -> None: ...


@pytest.fixture
def mock_controller() -> AsyncMock:
    ctrl = AsyncMock(spec=ControllerProtocol)
    ctrl.execute.return_value = None
    return ctrl


@pytest.fixture
def mock_capability_registry() -> MagicMock:
    registry = MagicMock(spec=CapabilityRegistryProtocol)
    registry.validate_call.return_value = (True, RiskLevel.LOW)
    return registry


class MockStepClassifier:
    def __init__(self, outcome: StepStatus = StepStatus.COMPLETED):
        self.outcome: StepStatus = outcome
        self.call_count: int = 0

    def classify_completion_status(self, step: PlanStep) -> StepStatus:
        self.call_count += 1
        return self.outcome


@pytest.fixture
def mock_blackboard() -> AsyncMock:
    bb = AsyncMock(spec=Blackboard)

    # Coroutine methods must be AsyncMock with boolean return values
    bb.is_anomaly_present = AsyncMock(return_value=False)
    bb.get_safety_flag = AsyncMock(return_value=False)

    # Other methods that are awaited but don't return meaningful values can be plain AsyncMock
    bb.get_current_step = AsyncMock(return_value=None)
    bb.add_trace_entry = AsyncMock(return_value=None)

    return bb


def make_plan_step(
    id: str = "s1",
    action_name: str = "move_arm",
    arguments: dict[str, Any] | None = None,
    status: StepStatus = StepStatus.PENDING,
) -> PlanStep:
    return PlanStep(
        id=id,
        description="Test step description",
        action=AgentToolCall(
            action_name=action_name,
            arguments=arguments or {"x": 10, "y": 20},
        ),
        status=status,
    )


@pytest.mark.asyncio
async def test_execute_step_success(
    mock_blackboard: AsyncMock,
    mock_capability_registry: MagicMock,
    mock_controller: AsyncMock,
) -> None:
    step = make_plan_step(action_name="bake_bread", arguments={"temp": 400})
    classifier = MockStepClassifier(StepStatus.COMPLETED)

    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_capability_registry,
        mock_controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )

    await executor.execute_step(step)

    mock_capability_registry.validate_call.assert_called_once_with("bake_bread", {"temp": 400})

    mock_controller.execute.assert_called_once_with("bake_bread", {"temp": 400})

    assert step.status == StepStatus.COMPLETED
    mock_blackboard.add_trace_entry.assert_called()


@pytest.mark.asyncio
async def test_execute_step_retry_logic(
    mock_blackboard: AsyncMock,
    mock_capability_registry: MagicMock,
    mock_controller: AsyncMock,
) -> None:
    step = make_plan_step()

    class FlakyClassifier:
        def __init__(self) -> None:
            self.attempts: int = 0

        def classify_completion_status(self, s: PlanStep) -> StepStatus:
            self.attempts += 1
            if self.attempts < 2:
                return StepStatus.FAILED
            return StepStatus.COMPLETED

    classifier = FlakyClassifier()

    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_capability_registry,
        mock_controller,
        default_max_retries=3,
        default_retry_delay=0.001,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
    )

    await executor.execute_step(step)

    assert step.status == StepStatus.COMPLETED

    assert mock_controller.execute.call_count == 2
    assert mock_capability_registry.validate_call.call_count == 2
    assert step.attempts == 2


@pytest.mark.asyncio
async def test_execute_step_permanent_failure(
    mock_blackboard: AsyncMock,
    mock_capability_registry: MagicMock,
    mock_controller: AsyncMock,
) -> None:
    step = make_plan_step()

    classifier = MockStepClassifier(StepStatus.FAILED)

    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_capability_registry,
        mock_controller,
        default_max_retries=2,
        default_retry_delay=0,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
    )

    await executor.execute_step(step)

    assert step.status == StepStatus.FAILED
    assert mock_controller.execute.call_count == 2
    assert classifier.call_count == 2

    trace_calls = mock_blackboard.add_trace_entry.call_args_list
    assert trace_calls[-1][0][0].event_type == "STEP_FAILED"


@pytest.mark.asyncio
async def test_execute_step_blocked_by_anomaly(
    mock_blackboard: AsyncMock,
    mock_capability_registry: MagicMock,
    mock_controller: AsyncMock,
) -> None:
    step = make_plan_step()
    classifier = MockStepClassifier(StepStatus.COMPLETED)

    mock_blackboard.is_anomaly_present.return_value = True

    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_capability_registry,
        mock_controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )

    await executor.execute_step(step)

    mock_capability_registry.validate_call.assert_not_called()
    mock_controller.execute.assert_not_called()

    assert step.status == StepStatus.PENDING


@pytest.mark.asyncio
async def test_execute_step_validation_failure(
    mock_blackboard: AsyncMock,
    mock_capability_registry: MagicMock,
    mock_controller: AsyncMock,
) -> None:
    step = make_plan_step()
    classifier = MockStepClassifier()

    mock_capability_registry.validate_call.side_effect = ValueError("Invalid parameters.")

    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_capability_registry,
        mock_controller,
        default_max_retries=1,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_retry_delay=0.01,
    )

    await executor.execute_step(step)

    mock_capability_registry.validate_call.assert_called_once()

    mock_controller.execute.assert_not_called()

    assert step.status == StepStatus.FAILED

    trace_calls = mock_blackboard.add_trace_entry.call_args_list
    assert len(trace_calls) == 2

    validation_trace = trace_calls[0][0][0]
    assert validation_trace.event_type == "VALIDATION_FAILED"
    assert "Invalid parameters" in validation_trace.reasoning_text

    final_trace = trace_calls[-1][0][0]
    assert final_trace.event_type == "STEP_FAILED"


@pytest.mark.asyncio
async def test_execute_step_aborted_by_emergency_stop(
    mock_blackboard: AsyncMock,
    mock_capability_registry: MagicMock,
    mock_controller: AsyncMock,
) -> None:
    step = make_plan_step()
    classifier = MockStepClassifier()

    # Simulate emergency stop flag set
    mock_blackboard.get_safety_flag = AsyncMock(return_value=True)

    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_capability_registry,
        mock_controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )

    await executor.execute_step(step)

    assert step.status == StepStatus.FAILED
    # No controller calls should happen
    mock_controller.execute.assert_not_called()
    # Trace entry should indicate aborted execution
    trace_calls = mock_blackboard.add_trace_entry.call_args_list
    assert any(call[0][0].event_type == "EXECUTION_ABORTED" for call in trace_calls)


@pytest.mark.asyncio
async def test_execute_step_validation_returns_unexpected_shape(
    mock_blackboard: AsyncMock,
    mock_capability_registry: MagicMock,
    mock_controller: AsyncMock,
) -> None:
    step = make_plan_step()
    classifier = MockStepClassifier()

    # Return a wrong shape
    mock_capability_registry.validate_call.return_value = "not_a_tuple"

    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_capability_registry,
        mock_controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )

    await executor.execute_step(step)

    assert step.status == StepStatus.FAILED
    mock_controller.execute.assert_not_called()
    trace_calls = mock_blackboard.add_trace_entry.call_args_list
    assert any(call[0][0].event_type == "VALIDATION_FAILED" for call in trace_calls)


@pytest.mark.asyncio
async def test_execute_step_controller_failure(
    mock_blackboard: AsyncMock,
    mock_capability_registry: MagicMock,
    mock_controller: AsyncMock,
) -> None:
    step = make_plan_step()
    classifier = MockStepClassifier()

    mock_controller.execute.side_effect = RuntimeError("Controller error")

    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_capability_registry,
        mock_controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )

    await executor.execute_step(step)

    assert step.status == StepStatus.FAILED
    trace_calls = mock_blackboard.add_trace_entry.call_args_list
    assert any(call[0][0].event_type == "EXECUTION_FAILED" for call in trace_calls)


@pytest.mark.asyncio
async def test_execute_step_skips_non_pending_step(
    mock_blackboard: AsyncMock,
    mock_capability_registry: MagicMock,
    mock_controller: AsyncMock,
) -> None:
    step = make_plan_step(status=StepStatus.COMPLETED)
    classifier = MockStepClassifier()

    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_capability_registry,
        mock_controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )

    await executor.execute_step(step)

    # Should remain unchanged
    assert step.status == StepStatus.COMPLETED
    mock_controller.execute.assert_not_called()
    mock_capability_registry.validate_call.assert_not_called()


@pytest.mark.asyncio
async def test_executor_loop_halts_on_emergency_stop(
    mock_blackboard: AsyncMock,
    mock_capability_registry: MagicMock,
    mock_controller: AsyncMock,
) -> None:
    classifier = MockStepClassifier()
    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_capability_registry,
        mock_controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )

    mock_blackboard.get_safety_flag = AsyncMock(return_value=True)

    executor._loop_running = True
    task = asyncio.create_task(executor._executor_loop())
    await asyncio.sleep(0.1)
    executor._loop_running = False
    await task

    trace_calls = mock_blackboard.add_trace_entry.call_args_list
    assert any(call[0][0].event_type == "EXECUTOR_HALTED" for call in trace_calls)
