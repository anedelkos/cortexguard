from typing import Any, Protocol
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from kitchenwatch.edge.models.action import Action, ActionStatus
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.plan import PlanStep, StepStatus
from kitchenwatch.edge.step_executor import StepExecutor

# --- Protocols & Mocks ---


class ControllerProtocol(Protocol):
    async def execute(self, primitive_name: str, parameters: dict[str, Any]) -> None: ...


@pytest.fixture
def mock_controller() -> AsyncMock:
    """Mocks the low-level hardware controller."""
    ctrl = AsyncMock()
    # Simulate instant execution
    ctrl.execute = AsyncMock(return_value=None)
    return ctrl


@pytest.fixture
def mock_action_registry(mock_controller: AsyncMock) -> MagicMock:
    """
    Mocks ActionRegistry to return primitive definitions.
    Crucially, it exposes the controller so StepExecutor can call it.
    """
    registry = MagicMock()
    registry._controller = mock_controller

    # Define a standard response for capabilities
    def get_primitives(capability_name: str) -> list[dict[str, Any]]:
        if capability_name == "MOVE_ITEM":
            return [{"primitive": "move_arm", "parameters": {"x": 10}}]
        if capability_name == "SET_TEMP":
            return [{"primitive": "set_thermostat", "parameters": {"t": 100}}]
        if capability_name == "COMPLEX_SEQ":
            return [
                {"primitive": "op_1", "parameters": {}},
                {"primitive": "op_2", "parameters": {}},
            ]
        raise KeyError(f"Capability {capability_name} not found")

    registry.get_primitives_for_capability.side_effect = get_primitives
    return registry


class MockStepClassifier:
    """Controls the outcome of the step (Success vs Failure) for testing logic."""

    def __init__(self, outcome: StepStatus = StepStatus.COMPLETED):
        self.outcome: StepStatus = outcome
        self.call_count: int = 0

    def classify_completion_status(self, step: PlanStep) -> StepStatus:
        self.call_count += 1
        return self.outcome


@pytest.fixture
def mock_blackboard() -> AsyncMock:
    bb = AsyncMock(spec=Blackboard)
    bb.is_anomaly_present.return_value = False  # Default safe
    return bb


# --- Helper to create valid Pydantic Models ---


def make_plan_step(
    id: str = "s1", capability: str = "MOVE_ITEM", status: StepStatus = StepStatus.PENDING
) -> PlanStep:
    return PlanStep(
        id=id,
        description="Test step description",
        action=Action(tool_id="test_tool", capability=capability, arguments={}),
        status=status,
    )


# --- Tests ---


@pytest.mark.asyncio
async def test_execute_step_success(
    mock_blackboard: AsyncMock, mock_action_registry: MagicMock, mock_controller: AsyncMock
) -> None:
    """Verify a step executes its primitive and updates status to COMPLETED."""
    step = make_plan_step()
    classifier = MockStepClassifier(StepStatus.COMPLETED)

    executor = StepExecutor(mock_blackboard, classifier, mock_action_registry)  # type: ignore

    await executor.execute_step(step)

    # 1. Primitive was executed
    mock_controller.execute.assert_called_once_with("move_arm", {"x": 10})

    # 2. Action status updated
    assert step.action.status == ActionStatus.COMPLETED

    # 3. Step status updated
    assert step.status == StepStatus.COMPLETED


@pytest.mark.asyncio
async def test_execute_step_retry_logic(
    mock_blackboard: AsyncMock, mock_action_registry: MagicMock, mock_controller: AsyncMock
) -> None:
    """Verify that the executor retries on failure before giving up."""
    step = make_plan_step()

    # Classifier essentially says "Physical action happened, but goal not met"
    # causing a retry loop.
    class FlakyClassifier:
        def __init__(self) -> None:
            self.attempts: int = 0

        def classify_completion_status(self, s: PlanStep) -> StepStatus:
            self.attempts += 1
            if self.attempts < 2:
                return StepStatus.FAILED

            return StepStatus.COMPLETED

    classifier = FlakyClassifier()

    # Configure executor for fast retries
    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_action_registry,  # type: ignore
        default_max_retries=3,
        default_retry_delay=0,
    )

    await executor.execute_step(step)

    # 1. Should have succeeded eventually
    assert step.status == StepStatus.COMPLETED

    # 2. Controller should be called twice (Initial + 1 Retry)
    assert mock_controller.execute.call_count == 2


@pytest.mark.asyncio
async def test_execute_step_permanent_failure(
    mock_blackboard: AsyncMock, mock_action_registry: MagicMock
) -> None:
    """Verify that exhausting retries leads to FAILED status."""
    step = make_plan_step()

    # Always fails
    classifier = MockStepClassifier(StepStatus.FAILED)

    executor = StepExecutor(
        mock_blackboard,
        classifier,
        mock_action_registry,  # type: ignore
        default_max_retries=2,
        default_retry_delay=0,
    )

    await executor.execute_step(step)

    # Should be FAILED after 2 attempts
    assert step.status == StepStatus.FAILED
    assert classifier.call_count == 2


@pytest.mark.asyncio
async def test_execute_step_blocked_by_anomaly(
    mock_blackboard: AsyncMock, mock_action_registry: MagicMock, mock_controller: AsyncMock
) -> None:
    """Verify execution halts immediately if anomaly is present."""
    step = make_plan_step()
    classifier = MockStepClassifier(StepStatus.COMPLETED)

    # Simulate Active Anomaly
    mock_blackboard.is_anomaly_present.return_value = True

    executor = StepExecutor(mock_blackboard, classifier, mock_action_registry)  # type: ignore

    await executor.execute_step(step)

    # Should perform NO actions
    mock_controller.execute.assert_not_called()

    # Should NOT change status (remains PENDING or whatever logic dictates, strictly not COMPLETED)
    assert step.status != StepStatus.COMPLETED


@pytest.mark.asyncio
async def test_execute_step_missing_capability(
    mock_blackboard: AsyncMock, mock_action_registry: MagicMock
) -> None:
    """Verify handling of unknown capabilities."""
    step = make_plan_step(capability="UNKNOWN_CAPABILITY")
    classifier = MockStepClassifier()

    executor = StepExecutor(mock_blackboard, classifier, mock_action_registry)  # type: ignore

    await executor.execute_step(step)

    assert step.status == StepStatus.FAILED
    assert step.action.status == ActionStatus.FAILED


@pytest.mark.asyncio
async def test_execute_complex_sequence(
    mock_blackboard: AsyncMock, mock_action_registry: MagicMock, mock_controller: AsyncMock
) -> None:
    """Verify it executes a list of primitives in order."""
    step = make_plan_step(capability="COMPLEX_SEQ")
    classifier = MockStepClassifier(StepStatus.COMPLETED)

    executor = StepExecutor(mock_blackboard, classifier, mock_action_registry)  # type: ignore

    await executor.execute_step(step)

    assert mock_controller.execute.call_count == 2
    mock_controller.execute.assert_has_calls([call("op_1", {}), call("op_2", {})])
    assert step.status == StepStatus.COMPLETED
