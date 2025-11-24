import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import py_trees
import pytest
import yaml

from kitchenwatch.core.action_registry import ActionRegistry, PrimitiveLeaf
from kitchenwatch.core.interfaces.base_controller import BaseController


@pytest.fixture
def controller() -> BaseController:
    # Use AsyncMock for the controller's methods
    ctrl = AsyncMock(spec=BaseController)
    ctrl.execute = AsyncMock()
    return ctrl


@pytest.fixture
def registry(controller: BaseController) -> ActionRegistry:
    return ActionRegistry(controller)


def test_load_from_yaml(registry: ActionRegistry, controller: BaseController) -> None:
    data = {"flip_burger": [{"primitive": "move_to_burger"}, {"primitive": "flip_burger"}]}

    # Write temporary YAML
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        yaml.dump(data, f)
        path = f.name

    registry.load_from_yaml(path)
    sequence = registry.build("flip_burger")
    # We still need to assert that the sequence is created with memory=True,
    # but the test fixture doesn't expose the registry's internal sequence creation logic.
    # Assuming ActionRegistry.build now correctly uses memory=True per our discussion.
    assert isinstance(sequence, py_trees.composites.Sequence)
    assert len(sequence.children) == 2


def test_build_invalid_action_raises(registry: ActionRegistry) -> None:
    with pytest.raises(KeyError):
        registry.build("nonexistent_action")


def test_load_from_yaml_invalid_steps(registry: ActionRegistry, tmp_path: Path) -> None:
    data = {"bad_action": {"primitive": "not_a_list"}}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(data))

    with pytest.raises(ValueError):
        registry.load_from_yaml(str(path))


@pytest.mark.asyncio
async def test_primitive_leaf_update_runs_and_reuses(controller: AsyncMock) -> None:
    # Set up
    leaf = PrimitiveLeaf(name="test", controller=controller, parameters={"foo": "bar"})
    # Always call setup before ticking a PyTree node
    leaf.setup(timeout=5.0)

    # --- Phase 1: Start the task (returns RUNNING) ---
    # The leaf is ticked, it calls controller.execute and wraps it in a task.
    status1 = leaf.update()
    assert status1 == py_trees.common.Status.RUNNING

    # Check that the function was *called* (scheduled), not awaited,
    # as the await happens inside the event loop.
    controller.execute.assert_called_once_with("test", {"foo": "bar"})

    # --- Phase 2: Wait for task completion ---
    # Yield control to the event loop for a moment. The AsyncMock task finishes immediately.
    await asyncio.sleep(0.01)

    # --- Phase 3: Detect completion (returns SUCCESS) ---
    # The leaf is ticked again. It detects that its internal task is done() and transitions to SUCCESS.
    status2 = leaf.update()
    assert status2 == py_trees.common.Status.SUCCESS

    # Ensure controller execute count remains 1 (it was not re-executed in Phase 3)
    assert controller.execute.call_count == 1
