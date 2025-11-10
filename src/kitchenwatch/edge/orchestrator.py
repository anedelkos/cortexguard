import asyncio
import logging

from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.plan import Plan, PlanStatus, StepStatus
from kitchenwatch.edge.utils.async_priority_queue import AsyncPriorityQueue

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator-centric async plan scheduler for KitchenWatch Edge.

    Responsibilities:
    - Manage plan queue and priorities
    - Observe blackboard for step completion/failure
    - Handle urgent remediation plans
    - Advance, pause, or resume plans based on blackboard state
    """

    def __init__(self, blackboard: Blackboard, logger: logging.Logger = logger) -> None:
        self._blackboard = blackboard
        self._plan_queue = AsyncPriorityQueue[Plan]()
        self._current_plan: Plan | None = None
        self._loop_running: bool = False
        self.__logger = logger

    # ---------------------
    # Plan Queue Methods
    # ---------------------
    async def add_plan(self, plan: Plan) -> None:
        """Add a new plan to the priority queue."""
        try:
            await self._plan_queue.put(plan.priority, plan)
            self.__logger.debug(f"Plan added: {plan.plan_id} with priority {plan.priority}")
        except Exception as e:
            self.__logger.exception(f"Failed to add plan {getattr(plan, 'plan_id', '?')}: {e}")

    async def _clear_current_plan(self) -> None:
        self._current_plan = None
        await self._blackboard.set_current_plan(None)
        await self._blackboard.set_current_step(None)

    async def _start_next_plan(self) -> None:
        """Pop the next plan from the queue and set it as current."""
        if not self._plan_queue:
            await self._clear_current_plan()
            return

        try:
            next_plan = await self._plan_queue.pop()
            if next_plan.status == PlanStatus.PENDING:
                self.__logger.info(f"Starting plan {next_plan.plan_id}")
            elif next_plan.status == PlanStatus.PREEMPTED:
                self.__logger.info(f"Resuming plan {next_plan.plan_id}")
            else:
                self.__logger.info(
                    f"Plan found in unexpected status {next_plan.plan_id}-{next_plan.status}... exiting"
                )
                return

            if next_plan.steps:
                # Restore step from current_step_index if needed
                index = next_plan.current_step_index
                index = min(index, len(next_plan.steps) - 1)
                current_step = next_plan.steps[index]

                await self._blackboard.set_current_plan(next_plan)
                await self._blackboard.set_current_step(current_step)

                next_plan.status = PlanStatus.RUNNING
                self._current_plan = next_plan

        except Exception as e:
            self.__logger.exception(f"Failed to start next plan: {e}")
            await self._clear_current_plan()

    async def _check_for_preemption(self) -> None:
        if not self._current_plan:
            return

        # Peek at the next plan in the queue without removing it
        next_plan = await self._plan_queue.peek()
        if next_plan and next_plan.priority < self._current_plan.priority:
            self.__logger.info(
                f"Preempting current plan {self._current_plan.plan_id} "
                f"for higher-priority plan {next_plan.plan_id}"
            )
            await self._pause_current_plan(interrupted=True)
            await self._start_next_plan()

    # ---------------------
    # Blackboard Observation Loop
    # ---------------------
    async def run(self, tick_interval: float = 0.1) -> None:
        """
        Main orchestrator loop observing blackboard for step completion or urgent plans.
        Runs until stopped via `stop()`.
        """
        self._loop_running = True
        self.__logger.info("Orchestrator started.")
        try:
            while self._loop_running:
                try:
                    # Start a plan if none running
                    if not self._current_plan and self._plan_queue:
                        await self._start_next_plan()

                    # Check if any urgent plans have been added, then if current step is completed or failed
                    if self._current_plan:
                        await self._check_for_preemption()

                        current_step = await self._blackboard.get_current_step()
                        step_status = (
                            getattr(current_step, "status", None) if current_step else None
                        )

                        if step_status is not None and step_status in (
                            StepStatus.COMPLETED,
                            StepStatus.FAILED,
                        ):
                            await self._advance_plan_or_handle_failure()

                    await asyncio.sleep(tick_interval)

                except asyncio.CancelledError:
                    self.__logger.info("Orchestrator loop cancelled gracefully.")
                    raise
                except Exception as loop_err:
                    self.__logger.exception(f"Unexpected error in orchestrator tick: {loop_err}")

        except Exception as e:
            self.__logger.critical(f"Fatal orchestrator error: {e}", exc_info=True)
        finally:
            self.__logger.info("Orchestrator stopped.")
            self._loop_running = False

    async def _advance_plan_or_handle_failure(self) -> None:
        """Advance to next step or insert urgent plan based on blackboard."""
        if not self._current_plan:
            return

        try:
            current_step = await self._blackboard.get_current_step()
            if not current_step:
                return

            step_status = getattr(current_step, "status", None)

            if step_status == StepStatus.FAILED:
                self.__logger.warning(
                    f"Step failed: {current_step.id}. Pausing plan for remediation."
                )
                await self._pause_current_plan()
            else:
                self._current_plan.current_step_index += 1
                if self._current_plan.current_step_index < len(self._current_plan.steps):
                    # Advance plan
                    next_step = self._current_plan.steps[self._current_plan.current_step_index]
                    await self._blackboard.set_current_step(next_step)
                else:
                    # Complete plan
                    self.__logger.info(f"Plan completed: {self._current_plan.plan_id} successfully")
                    self._current_plan.status = PlanStatus.COMPLETED
                    await self._clear_current_plan()
                    await self._start_next_plan()

        except Exception as e:
            self.__logger.exception(f"Error advancing plan: {e}")

    async def _pause_current_plan(self, interrupted: bool = False) -> None:
        """Pause the currently running plan and re-queue it by priority."""
        if not self._current_plan:
            return

        try:
            if interrupted:
                # Plan will resume after interrupt plan finishes executing
                self._current_plan.status = PlanStatus.PREEMPTED
                self.__logger.info(f"Preempted plan {self._current_plan.plan_id}")
                await self.add_plan(self._current_plan)
            else:
                # Plan has failed, place it on blackboard to be remediated
                self._current_plan.status = PlanStatus.PAUSED
                self.__logger.info(f"Paused plan {self._current_plan.plan_id}")
                await self._blackboard.set_paused_plan(self._current_plan)

            await self._clear_current_plan()

        except Exception as e:
            self.__logger.exception(f"Failed to pause and requeue plan: {e}")

    async def stop(self) -> None:
        """Stop orchestrator loop and persist state if needed."""
        self.__logger.info("Stop signal received. Flushing state...")
        self._loop_running = False

        try:
            if self._current_plan:
                await self._blackboard.set("last_active_plan", self._current_plan.serialize())
        except Exception as e:
            self.__logger.warning(f"Failed to flush orchestrator state: {e}")
