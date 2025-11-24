import asyncio
from typing import Protocol
from unittest.mock import AsyncMock, patch

import py_trees
import pytest

from kitchenwatch.core.action_registry import ActionRegistry, PrimitiveLeaf
from kitchenwatch.core.interfaces.base_controller import BaseController
from kitchenwatch.edge.models.plan import PlanStep, StepStatus
from kitchenwatch.edge.step_executor import (
    EXECUTOR_IDLE_INTERVAL,
    EXECUTOR_POLL_INTERVAL,
    StepExecutor,
)


class ControllerMockProtocol(BaseController, Protocol):
    execute: AsyncMock


@pytest.fixture
def mock_controller() -> AsyncMock:
    ctrl = AsyncMock()
    # Execute yields control once to allow task scheduling logic to complete
    ctrl.execute = AsyncMock(side_effect=lambda *a, **k: asyncio.sleep(0))
    return ctrl


@pytest.fixture
def registry(mock_controller: BaseController) -> ActionRegistry:
    reg = ActionRegistry(controller=mock_controller)
    reg._actions["flip_burger"] = [{"primitive": "flip_burger"}]
    reg._actions["noop"] = [{"primitive": "noop"}]
    return reg


# --- Helper Mocks for StepExecutor Tests ---


class DummyClassifier:
    """Default classifier that always returns COMPLETED."""

    def classify_completion_status(self, step: PlanStep) -> StepStatus:
        return StepStatus.COMPLETED


@pytest.fixture
def setup_executor(registry: ActionRegistry) -> tuple[StepExecutor, AsyncMock]:
    blackboard = AsyncMock()
    # Default behavior: no anomaly, current step is none.
    blackboard.is_anomaly_present = AsyncMock(return_value=False)
    # The return value for get_current_step can be PlanStep or None
    blackboard.get_current_step = AsyncMock(return_value=None)

    executor = StepExecutor(blackboard, DummyClassifier(), registry)
    return executor, blackboard


# --- Existing Tests ---


@pytest.mark.asyncio
async def test_build_sequence_returns_pytrees_sequence(registry: ActionRegistry) -> None:
    seq = registry.build("flip_burger")
    assert isinstance(seq, py_trees.composites.Sequence)
    assert len(seq.children) == 1
    leaf = seq.children[0]
    assert isinstance(leaf, PrimitiveLeaf)
    assert leaf.name == "flip_burger"


@pytest.mark.asyncio
async def test_primitive_leaf_executes_controller(mock_controller: ControllerMockProtocol) -> None:
    leaf = PrimitiveLeaf(name="flip_burger", controller=mock_controller)
    leaf.setup()

    # Phase 1: Start the task (returns RUNNING)
    status1 = leaf.update()
    assert status1 == py_trees.common.Status.RUNNING
    mock_controller.execute.assert_called_once_with("flip_burger", {})

    # Phase 2: Wait for task completion
    await asyncio.sleep(0.01)

    # Phase 3: Detect completion (returns SUCCESS)
    status2 = leaf.update()
    assert status2 == py_trees.common.Status.SUCCESS
    assert mock_controller.execute.call_count == 1


@pytest.mark.asyncio
async def test_execute_step_success(
    registry: ActionRegistry,
    mock_controller: ControllerMockProtocol,
) -> None:
    blackboard = AsyncMock()
    blackboard.is_anomaly_present = AsyncMock(return_value=False)

    step = PlanStep(id="1", name="flip_burger", status=StepStatus.PENDING)
    executor = StepExecutor(blackboard, DummyClassifier(), registry)

    await executor.execute_step(step)

    assert mock_controller.execute.call_count == 1
    assert step.status == StepStatus.COMPLETED
    mock_controller.execute.assert_called_once_with("flip_burger", {})


@pytest.mark.asyncio
async def test_execute_step_failure_and_retry(
    registry: ActionRegistry,
    mock_controller: ControllerMockProtocol,
) -> None:
    class RetryingClassifier:
        def __init__(self) -> None:
            self.calls: int = 0

        def classify_completion_status(self, step: PlanStep) -> StepStatus:
            self.calls += 1
            # Fails first time, succeeds second time
            return StepStatus.FAILED if self.calls == 1 else StepStatus.COMPLETED

    blackboard = AsyncMock()
    blackboard.is_anomaly_present = AsyncMock(return_value=False)

    step = PlanStep(id="1", name="flip_burger", status=StepStatus.PENDING)
    # Set max retries to 2 for 2 total attempts (1 fail + 1 success) to pass the classifier logic.
    executor = StepExecutor(
        blackboard, RetryingClassifier(), registry, default_max_retries=2, default_retry_delay=0
    )

    # We rely on the mock_controller's side_effect of asyncio.sleep(0) for instant, safe yielding.
    await executor.execute_step(step)

    # Should call controller twice: first attempt failed, second attempt succeeded
    assert mock_controller.execute.call_count == 2
    assert step.status == StepStatus.COMPLETED


@pytest.mark.asyncio
async def test_execute_step_permanent_failure() -> None:
    """Tests the step hitting the maximum retry limit and failing permanently."""

    class AlwaysFailingClassifier:
        def classify_completion_status(self, step: PlanStep) -> StepStatus:
            return StepStatus.FAILED

    # Setup Mocks
    blackboard = AsyncMock()
    blackboard.is_anomaly_present = AsyncMock(return_value=False)

    registry = AsyncMock(spec=ActionRegistry)
    mock_root = py_trees.composites.Sequence(name="mock", children=[], memory=True)
    registry.build.return_value = mock_root

    step = PlanStep(id="1", name="mock_fail", status=StepStatus.PENDING)

    # Since the loop is `range(1, max + 1)`, a value of 1 results in range(1, 2),
    # which runs exactly once. A value of 0 results in an empty loop.
    executor = StepExecutor(
        blackboard,
        AlwaysFailingClassifier(),
        registry,
        default_max_retries=1,
        default_retry_delay=0,
    )

    # Mock the method that execute_step *should* call
    mock_run_tree = AsyncMock(return_value=py_trees.common.Status.SUCCESS)

    with patch.object(executor, "_run_tree_to_completion", new=mock_run_tree):
        await executor.execute_step(step)

    # Check if the execution attempt happened once
    assert mock_run_tree.call_count == 1
    # Check if the status correctly transitioned to FAILED after 1 attempt
    assert step.status == StepStatus.FAILED


@pytest.mark.asyncio
async def test_execute_step_action_not_registered() -> None:
    """Tests failure when action_registry.build raises KeyError."""
    blackboard = AsyncMock()
    blackboard.is_anomaly_present = AsyncMock(return_value=False)
    registry = AsyncMock(spec=ActionRegistry)
    registry.build.side_effect = KeyError("Action not found")

    step = PlanStep(id="1", name="nonexistent_action", status=StepStatus.PENDING)
    executor = StepExecutor(blackboard, DummyClassifier(), registry)

    await executor.execute_step(step)

    assert step.status == StepStatus.FAILED


@pytest.mark.asyncio
async def test_execute_step_anomaly_interrupts_execution(
    registry: ActionRegistry,
    mock_controller: ControllerMockProtocol,
) -> None:
    """Tests the case where a MEDIUM+ anomaly appears during the tree's execution."""
    blackboard = AsyncMock()
    # Mock anomaly to appear *during* the tree tick loop
    blackboard.is_anomaly_present = AsyncMock(
        side_effect=[
            False,
            False,
            True,
        ]  # The check inside _run_tree_to_completion will catch the True
    )

    step = PlanStep(id="1", name="flip_burger", status=StepStatus.PENDING)
    executor = StepExecutor(blackboard, DummyClassifier(), registry)

    # FIX: Parameterize the generic Tuple
    async def mock_run_tree_anomaly_interrupt(*args: tuple[object, ...]) -> py_trees.common.Status:
        # This mock must return INVALID to signify interruption, as per StepExecutor logic
        return py_trees.common.Status.INVALID

    with patch.object(executor, "_run_tree_to_completion", new=mock_run_tree_anomaly_interrupt):
        await executor.execute_step(step)

    # Step status should revert to PENDING if interrupted by anomaly
    assert step.status == StepStatus.PENDING


@pytest.mark.asyncio
async def test_execute_step_blocked_by_anomaly_at_start(
    setup_executor: tuple[StepExecutor, AsyncMock],
) -> None:
    """Tests the safety check that prevents step execution if an anomaly is already present."""
    executor, blackboard = setup_executor
    # Set anomaly to be present immediately before the step starts
    blackboard.is_anomaly_present.return_value = True

    step = PlanStep(id="1", name="flip_burger", status=StepStatus.PENDING)

    # Use explicit mock variable to track calls
    mock_run_tree = AsyncMock()

    with patch.object(executor, "_run_tree_to_completion", new=mock_run_tree):
        await executor.execute_step(step)

    # Step status should remain PENDING
    assert step.status == StepStatus.PENDING
    # The execution flow should have been blocked, so run_tree_to_completion should not have been called
    mock_run_tree.assert_not_called()


@pytest.mark.asyncio
async def test_executor_control_methods_and_loop_flow(
    setup_executor: tuple[StepExecutor, AsyncMock],
) -> None:
    """Tests start, stop, pause, resume, and the main execution loop."""
    executor, blackboard = setup_executor

    # FIX: Use patch.object to mock the method, preventing "Cannot assign to a method" error (Line 245)
    with patch.object(executor, "execute_step", new=AsyncMock()) as mock_execute_step:
        pending_step = PlanStep(id="1", name="test", status=StepStatus.PENDING)

        # FIX: Explicitly type the side_effect list to resolve the "list-item" error
        side_effect_list: list[PlanStep | None] = [pending_step] * 5 + [None]

        # Provide enough steps to cover all execution periods (Start + Resume)
        # blackboard.get_current_step can return PlanStep or None.
        blackboard.get_current_step.side_effect = side_effect_list

        # --- 1. Start and Basic Tick ---
        await executor.start()
        assert executor._loop_running is True

        # Allow the loop to run a few times (3.5x the poll interval)
        await asyncio.sleep(EXECUTOR_POLL_INTERVAL * 3.5)

        # Should have attempted to execute the step multiple times
        assert mock_execute_step.call_count >= 2

        # --- 2. Pause ---
        await executor.pause()
        initial_count = mock_execute_step.call_count

        # Wait for longer than the poll interval, but it should hit the idle interval if paused
        await asyncio.sleep(EXECUTOR_IDLE_INTERVAL * 2)

        # Execution should halt
        assert mock_execute_step.call_count == initial_count  # No change while paused

        # --- 3. Resume ---
        await executor.resume()

        # Allow time for execution to resume
        await asyncio.sleep(EXECUTOR_POLL_INTERVAL * 3.5)

        # Execution should resume (call count must strictly increase)
        assert mock_execute_step.call_count > initial_count

        # --- 4. Anomaly Pause ---
        blackboard.is_anomaly_present.return_value = True  # Simulate critical anomaly
        initial_count = mock_execute_step.call_count
        await asyncio.sleep(EXECUTOR_IDLE_INTERVAL * 2)

        # Execution should halt due to anomaly
        assert mock_execute_step.call_count == initial_count

        # --- 5. Stop ---
        await executor.stop()
        assert executor._loop_running is False
        assert executor._task.done()


@pytest.mark.asyncio
async def test_execute_step_skips_non_pending_steps(
    setup_executor: tuple[StepExecutor, AsyncMock],
) -> None:
    """Tests that execute_step returns immediately if status is not PENDING."""
    executor, blackboard = setup_executor

    step_running = PlanStep(id="1", name="test", status=StepStatus.RUNNING)
    step_completed = PlanStep(id="2", name="test", status=StepStatus.COMPLETED)

    with patch.object(executor, "_run_tree_to_completion", new=AsyncMock()) as mock_run_tree:
        await executor.execute_step(step_running)
        await executor.execute_step(step_completed)

    # Execution logic should be skipped entirely for non-pending steps
    mock_run_tree.assert_not_called()
