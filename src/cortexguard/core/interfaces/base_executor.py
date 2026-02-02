from typing import Protocol

from cortexguard.edge.models.plan import PlanStep


class BaseExecutor(Protocol):
    """
    Protocol for executors that run plan steps.

    Executors observe the blackboard for the current step, execute it,
    and update status back to the blackboard. Must support preemption and resumption.
    """

    async def start(self) -> None:
        """
        Start the executor loop.
        Continuously monitors the blackboard and executes steps.
        """
        ...

    async def stop(self) -> None:
        """
        Stop the executor loop gracefully.
        """
        ...

    async def execute_step(self, step: PlanStep) -> None:
        """
        Execute a single step. Should update the step status on the blackboard.
        Handle errors and report step failure.
        """
        ...

    async def pause(self) -> None:
        """
        Pause the currently running step if any. Used for preemption.
        """
        ...

    async def resume(self) -> None:
        """
        Resume the paused step.
        """
        ...
