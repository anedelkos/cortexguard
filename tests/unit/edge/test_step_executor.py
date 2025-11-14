import asyncio
from typing import Protocol
from unittest.mock import AsyncMock

import py_trees
import pytest

from kitchenwatch.core.action_registry import ActionRegistry, PrimitiveLeaf
from kitchenwatch.core.interfaces.base_controller import BaseController
from kitchenwatch.edge.models.plan import PlanStep, StepStatus
from kitchenwatch.edge.step_executor import StepExecutor


class ControllerMockProtocol(BaseController, Protocol):
    execute: AsyncMock


@pytest.fixture
def mock_controller() -> AsyncMock:
    ctrl = AsyncMock()
    ctrl.execute = AsyncMock()
    return ctrl


@pytest.fixture
def registry(mock_controller: BaseController) -> ActionRegistry:
    reg = ActionRegistry(controller=mock_controller)
    # Instead of register_action, populate _actions directly
    reg._actions["flip_burger"] = [{"primitive": "flip_burger"}]
    reg._actions["noop"] = [{"primitive": "noop"}]
    return reg


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
    status = leaf.update()
    await asyncio.sleep(0.01)  # allow async task to run
    mock_controller.execute.assert_called_once_with("flip_burger", {})
    assert status == py_trees.common.Status.SUCCESS


@pytest.mark.asyncio
async def test_execute_step_success(
    monkeypatch: pytest.MonkeyPatch,
    registry: ActionRegistry,
    mock_controller: ControllerMockProtocol,
) -> None:
    # Patch classifier to always report success
    class DummyClassifier:
        def classify_completion_status(self, step: PlanStep) -> StepStatus:
            return StepStatus.COMPLETED

    blackboard = AsyncMock()
    step = PlanStep(id="1", name="flip_burger", status=StepStatus.PENDING)
    executor = StepExecutor(blackboard, DummyClassifier(), registry)

    await executor.execute_step(step)

    assert step.status == StepStatus.COMPLETED
    mock_controller.execute.assert_called_once_with("flip_burger", {})


@pytest.mark.asyncio
async def test_execute_step_failure_and_retry(
    monkeypatch: pytest.MonkeyPatch,
    registry: ActionRegistry,
    mock_controller: ControllerMockProtocol,
) -> None:
    # Classifier fails first, then succeeds
    class DummyClassifier:
        def __init__(self) -> None:
            self.calls = 0

        def classify_completion_status(self, step: PlanStep) -> StepStatus:
            self.calls += 1
            if self.calls == 1:
                return StepStatus.FAILED
            return StepStatus.COMPLETED

    blackboard = AsyncMock()
    step = PlanStep(id="1", name="flip_burger", status=StepStatus.PENDING)
    executor = StepExecutor(blackboard, DummyClassifier(), registry, default_max_retries=2)

    await executor.execute_step(step)

    # Should eventually succeed
    assert step.status == StepStatus.COMPLETED
    assert mock_controller.execute.call_count == 2
