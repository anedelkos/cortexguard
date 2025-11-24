import asyncio
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
          # ...
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

        sequence = Sequence(name=action_name, memory=True, children=children)
        return sequence


class PrimitiveLeaf(Behaviour):
    """
    Leaf node that executes a single primitive via the controller asynchronously.
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
        # Store the running task
        self._task: asyncio.Task[Any] | None = None

    def setup(self, **kwargs: Any) -> None:
        """Reset state on tree setup."""
        self._task = None

    def update(self) -> Status:
        """Called continuously by the StepExecutor's ticking loop."""

        # 1. Start the task on the first tick (when _task is None)
        if self._task is None:
            # CRITICAL: Start the asynchronous execution call.
            self.logger.debug(f"Starting async action: {self.name} with {self._parameters}")

            # The PyTree is executed in an async context, so we can create a task.
            self._task = asyncio.create_task(self._controller.execute(self.name, self._parameters))
            return Status.RUNNING

        # 2. Check task status on subsequent ticks
        elif not self._task.done():
            # Action is still running in the background
            return Status.RUNNING

        # 3. Task is complete, check result and transition status
        else:
            try:
                # Retrieve the result (which will raise any exception the task raised)
                self._task.result()
                self.logger.debug(f"Action {self.name} completed successfully.")
                return Status.SUCCESS
            except asyncio.CancelledError:
                self.logger.warning(f"Action {self.name} was cancelled.")
                return Status.FAILURE
            except Exception as e:
                self.logger.error(f"Action {self.name} failed: {e}")
                return Status.FAILURE

    def terminate(self, new_status: Status) -> None:
        """
        Ensure the running task is cancelled if the tree terminates early (e.g., interrupted by anomaly).
        """
        if self._task and not self._task.done():
            self._task.cancel()
            self.logger.warning(f"Action {self.name} was terminated prematurely.")
        self._task = None
