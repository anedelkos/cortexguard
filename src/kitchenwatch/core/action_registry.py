from typing import Any

import yaml
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.composites import Sequence

from kitchenwatch.core.interfaces.base_controller import BaseController


class ActionRegistry:
    """
    Registry of high-level actions mapped to sequences of low-level primitives.
    """

    def __init__(self, controller: BaseController):
        self._controller = controller
        self._actions: dict[str, list[dict[str, Any]]] = {}

    def load_from_yaml(self, path: str) -> None:
        """
        Load actions from a YAML file.
        Format:
        flip_burger:
          - primitive: move_to_burger
            parameters: {x: 0.1, y: 0.2}
          - primitive: lower_tool
          - primitive: grasp_burger
          - primitive: flip_motion
          - primitive: release_burger
          - primitive: retreat
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        for action_name, steps in data.items():
            if not isinstance(steps, list):
                raise ValueError(f"Steps for action {action_name} must be a list")
            self._actions[action_name] = steps

    def build(self, action_name: str) -> Behaviour:
        """
        Return a PyTrees Sequence for the high-level action.
        Each leaf will call the controller with its primitive.
        """
        if action_name not in self._actions:
            raise KeyError(f"Action '{action_name}' not registered")

        children = []
        for step in self._actions[action_name]:
            primitive = step.get("primitive")
            if not primitive:
                raise ValueError(
                    f"Missing or invalid primitive in action '{action_name}' step: {step}"
                )
            parameters: dict[str, Any] = step.get("parameters", {})

            # Create a leaf behaviour that executes this primitive via controller
            leaf = PrimitiveLeaf(name=primitive, controller=self._controller, parameters=parameters)
            children.append(leaf)

        sequence = Sequence(name=action_name, memory=False, children=children)
        return sequence


class PrimitiveLeaf(Behaviour):
    """
    Leaf node that executes a single primitive via the controller.
    """

    def __init__(
        self,
        name: str,
        controller: BaseController,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name)
        self._controller = controller
        self._parameters: dict[str, Any] = parameters or {}
        self._executed = False

    def setup(self, **kwargs: Any) -> None:
        self._executed = False

    def update(self) -> Status:
        if self._executed:
            return Status.SUCCESS

        import asyncio

        asyncio.create_task(self._controller.execute(self.name, self._parameters))
        self._executed = True
        return Status.SUCCESS
