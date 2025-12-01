from typing import Any, Protocol, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.plan import PlanStep, StepStatus
from kitchenwatch.edge.step_executor import StepExecutor


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
    registry.validate_call.return_value = None
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
    bb.is_anomaly_present.return_value = False
    bb.get_current_step.return_value = None
    bb.add_trace_entry.return_value = None
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
        action=cast(
            Any,
            {
                "action_name": action_name,
                "arguments": arguments or {"x": 10, "y": 20},
                "status": "PENDING",
            },
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
