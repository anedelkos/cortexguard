import tempfile
from typing import Any
from unittest.mock import Mock  # Import Mock for creating a concrete mock object

import pytest
import yaml

from kitchenwatch.core.action_registry import ActionRegistry
from kitchenwatch.core.interfaces.base_controller import BaseController


@pytest.fixture
def controller() -> BaseController:
    """
    Fixture for a mock BaseController instance.
    We use Mock(spec=BaseController) to create an object that satisfies
    the type hint without trying to instantiate the abstract Protocol,
    which caused the previous TypeError.
    """
    return Mock(spec=BaseController)


@pytest.fixture
def registry(controller: BaseController) -> ActionRegistry:
    """Fixture for a clean ActionRegistry instance."""
    return ActionRegistry(controller)


def create_yaml_file(data: dict[str, Any]) -> str:
    """Helper to write test data to a temporary YAML file and return its path."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        yaml.dump(data, f)
        return f.name


def test_load_from_yaml_success(registry: ActionRegistry) -> None:
    """Tests if the registry correctly loads capabilities and primitives from a YAML file."""
    data = {
        "flip_burger": [
            {"primitive": "move_to_burger", "parameters": {"x": 0.1, "y": 0.2}},
            {"primitive": "lower_tool"},
        ],
        "chop_onion": [
            {"primitive": "pick_onion"},
            {"primitive": "chop", "parameters": {"slices": 5}},
        ],
    }

    path = create_yaml_file(data)
    registry.load_from_yaml(path)

    # Check if the registry loaded the data internally (by checking a successful retrieval)
    primitives = registry.get_primitives_for_capability("flip_burger")
    assert len(primitives) == 2
    assert primitives[0]["primitive"] == "move_to_burger"
    assert primitives[1]["primitive"] == "lower_tool"


def test_get_primitives_for_capability_key_error(registry: ActionRegistry) -> None:
    """Tests that accessing an unregistered capability raises a KeyError."""
    data = {"only_one": [{"primitive": "test"}]}
    path = create_yaml_file(data)
    registry.load_from_yaml(path)

    with pytest.raises(KeyError, match="nonexistent_capability"):
        registry.get_primitives_for_capability("nonexistent_capability")


def test_load_from_yaml_invalid_steps_raises(registry: ActionRegistry) -> None:
    """Tests that loading data where steps are not a list raises a ValueError."""
    data = {"bad_action": "not_a_list"}
    path = create_yaml_file(data)

    with pytest.raises(ValueError, match="Steps for capability bad_action must be a list"):
        registry.load_from_yaml(path)
