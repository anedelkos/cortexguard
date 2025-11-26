import logging
from typing import Any

import yaml

from kitchenwatch.core.interfaces.base_controller import BaseController

logger = logging.getLogger(__name__)


class ActionRegistry:
    """
    Registry of high-level actions mapped to sequences of low-level primitives.

    This class provides the primitive sequence (the "recipe" of low-level calls)
    corresponding to a high-level capability requested by an Action model.
    """

    def __init__(self, controller: BaseController):
        # We keep a reference to the controller to satisfy the __init__ signature,
        # but the registry itself no longer uses it for execution (the StepExecutor does).
        self._controller = controller
        # Format: {"capability_name": [{"primitive": "name", "parameters": {}}, ...]}
        self._actions: dict[str, list[dict[str, Any]]] = {}
        logger.info("ActionRegistry initialized.")

    def load_from_yaml(self, path: str) -> None:
        """
        Load capabilities and their primitive sequences from a YAML file.
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Action registry YAML file not found at {path}")
            return
        except yaml.YAMLError as e:
            logger.error(f"Error loading YAML action registry: {e}")
            return

        for action_name, steps in data.items():
            if not isinstance(steps, list):
                raise ValueError(f"Steps for capability {action_name} must be a list")
            self._actions[action_name] = steps

        logger.info(f"Loaded {len(self._actions)} capabilities into ActionRegistry.")

    def get_primitives_for_capability(self, capability_name: str) -> list[dict[str, Any]]:
        """
        Retrieves the sequence of primitives associated with a given capability name.
        This sequence is then executed directly by the StepExecutor.

        Args:
            capability_name: The name of the high-level capability (e.g., 'flip_burger').

        Returns:
            A list of dictionaries defining the primitive sequence.

        Raises:
            KeyError: If the capability name is not registered.
        """
        if capability_name not in self._actions:
            raise KeyError(f"Capability '{capability_name}' is not registered.")

        return self._actions[capability_name]
