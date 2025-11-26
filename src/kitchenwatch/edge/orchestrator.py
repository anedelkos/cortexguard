import asyncio
import logging

from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.plan import Plan, PlanStatus, StepStatus
from kitchenwatch.edge.utils.async_priority_queue import AsyncPriorityQueue

logger = logging.getLogger(__name__)

DEFAULT_TICK_INTERVAL = 0.1


class Orchestrator:
    """
    Orchestrator-centric async plan scheduler for KitchenWatch Edge.

    Responsibilities:
    - Manage plan queue and priorities
    - Observe blackboard for step completion/failure
    - Handle urgent remediation plans
    - Advance, pause, or resume plans based on blackboard state
    """

    def __init__(self, blackboard: Blackboard) -> None:
        self._blackboard = blackboard
        self._plan_queue = AsyncPriorityQueue[Plan]()
        self._current_plan: Plan | None = None
        self._loop_running: bool = False
        self._task: asyncio.Task[None] | None = None

    # ---------------------
    # Plan Queue Methods
    # ---------------------
    async def add_plan(self, plan: Plan) -> None:
        """Add a new plan to the priority queue."""
        try:
            await self._plan_queue.put(plan.context.priority, plan)
            logger.debug(f"Plan added: {plan.plan_id} with priority {plan.context.priority}")
        except Exception as e:
            logger.exception(f"Failed to add plan {getattr(plan, 'plan_id', '?')}: {e}")

    async def _clear_current_plan(self) -> None:
        self._current_plan = None
        await self._blackboard.clear_current_plan()

    async def _start_next_plan(self, plan: Plan | None = None) -> None:
        """Pop the next plan from the queue and set it as current."""
        try:
            next_plan = plan or await self._plan_queue.pop(block=False)
            if next_plan is None:
                await self._clear_current_plan()
                return

            if next_plan.status == PlanStatus.PENDING:
                logger.info(f"Starting plan {next_plan.plan_id}")
            elif next_plan.status == PlanStatus.PREEMPTED:
                logger.info(f"Resuming plan {next_plan.plan_id}")
            else:
                logger.error(
                    f"Plan found in unexpected status {next_plan.plan_id}-{next_plan.status}... exiting"
                )
                return

            if next_plan.steps:
                # === CRITICAL CHANGE: Retrieve index from Blackboard's execution state ===
                # This assumes the Blackboard tracks the index separate from the Plan object.
                index = await self._blackboard.get_step_index_for_plan(next_plan.plan_id) or 0

                index = min(index, len(next_plan.steps) - 1)
                current_step = next_plan.steps[index]

                next_plan.status = PlanStatus.RUNNING
                self._current_plan = next_plan
                await self._blackboard.set_current_plan(next_plan)
                await self._blackboard.set_current_step(current_step)
                # You would also update the index on the blackboard here if it changed
                await self._blackboard.set_step_index_for_plan(next_plan.plan_id, index)
            else:
                logger.warning(f"Plan {next_plan.plan_id} has no steps, skipping")

        except Exception as e:
            logger.exception(f"Failed to start next plan: {e}")
            await self._clear_current_plan()

    async def _check_for_preemption(self) -> Plan | None:
        if not self._current_plan:
            return None

        next_plan = await self._plan_queue.pop_if_priority_lower_than(
            self._current_plan.context.priority
        )
        if next_plan:
            logger.info(
                f"Preempting current plan {self._current_plan.plan_id} "
                f"for higher-priority plan {next_plan.plan_id}"
            )
            await self._pause_current_plan(interrupted=True)
            return next_plan

        return None

    # ---------------------
    # Main Loop
    # ---------------------
    async def _run_loop(self, tick_interval: float) -> None:
        """Main orchestrator loop observing blackboard."""
        logger.info("Orchestrator loop started.")
        try:
            while self._loop_running:
                try:
                    logger.debug(
                        f"Tick: current plan = {self._current_plan.plan_id if self._current_plan else 'None'}"
                    )

                    # Start a plan if none running
                    if not self._current_plan:
                        await self._start_next_plan()

                    # Check preemption and step status
                    if self._current_plan:
                        urgent_plan = await self._check_for_preemption()
                        if urgent_plan:
                            await self._start_next_plan(urgent_plan)
                            await asyncio.sleep(tick_interval)
                            continue

                        current_step = await self._blackboard.get_current_step()
                        step_status = (
                            getattr(current_step, "status", None) if current_step else None
                        )

                        if step_status and step_status in (StepStatus.COMPLETED, StepStatus.FAILED):
                            await self._advance_plan_or_handle_failure()

                    await asyncio.sleep(tick_interval)

                except asyncio.CancelledError:
                    logger.info("Orchestrator loop cancelled gracefully.")
                    raise
                except Exception as loop_err:
                    logger.exception(f"Unexpected error in orchestrator tick: {loop_err}")

        except Exception as e:
            logger.critical(f"Fatal orchestrator error: {e}", exc_info=True)
        finally:
            logger.info("Orchestrator stopped.")
            self._loop_running = False

    async def _advance_plan_or_handle_failure(self) -> None:
        """Advance to next step or handle plan state based on blackboard and step status."""

        # 1. Check current plan context
        if not self._current_plan:
            return

        plan_id = self._current_plan.plan_id

        if not self._current_plan.steps:
            logger.warning(f"Plan {plan_id} has no steps to advance")
            self._current_plan.status = PlanStatus.COMPLETED
            await self._blackboard.set_current_plan(self._current_plan)
            await self._blackboard.clear_step_index_for_plan(plan_id)
            await self._clear_current_plan()
            return

        # 2. Retrieve current step and index from Blackboard
        current_step = await self._blackboard.get_current_step()
        if not current_step:
            logger.warning(f"Orchestrator running plan {plan_id} but no current step is set.")
            return

        # Get the index (defaults to 0 if the plan just started and index hasn't been set yet)
        current_step_index = await self._blackboard.get_step_index_for_plan(plan_id)
        if current_step_index is None:
            # This should generally be set upon plan start, but handle the default case
            current_step_index = 0

        try:
            step_status = getattr(current_step, "status", None)

            if step_status == StepStatus.FAILED:
                # Failure scenario
                logger.warning(f"Step failed: {current_step.id}. Pausing plan for remediation.")
                self._current_plan.status = PlanStatus.FAILED
                # The index state (current_step_index) remains on the Blackboard for retry
                await self._pause_current_plan()

            elif step_status == StepStatus.COMPLETED:
                # Advance scenario
                next_index = current_step_index + 1

                if next_index < len(self._current_plan.steps):
                    # --- ADVANCE PLAN ---
                    # 3. Update the index on the Blackboard
                    await self._blackboard.set_step_index_for_plan(plan_id, next_index)

                    next_step = self._current_plan.steps[next_index]
                    await self._blackboard.set_current_step(next_step)
                    logger.info(
                        f"Advanced plan {plan_id} to step index {next_index}: {next_step.id}"
                    )
                else:
                    # --- COMPLETE PLAN ---
                    logger.info(f"Plan completed: {plan_id} successfully")
                    self._current_plan.status = PlanStatus.COMPLETED
                    await self._blackboard.set_current_plan(self._current_plan)
                    # 4. Clear the index state from the Blackboard
                    await self._blackboard.clear_step_index_for_plan(plan_id)
                    await self._clear_current_plan()

            # If status is None (Step not executed yet) or Running, do nothing and wait for executor
            # to update the step status.

        except Exception as e:
            logger.exception(f"Error advancing plan {plan_id}: {e}")
            self._current_plan.status = PlanStatus.FAILED
            await self._blackboard.set_failed_plan(self._current_plan)
            await self._blackboard.clear_step_index_for_plan(plan_id)
            await self._clear_current_plan()

    async def _pause_current_plan(self, interrupted: bool = False) -> None:
        """Pause the currently running plan and re-queue it by priority."""
        if not self._current_plan:
            return

        try:
            if interrupted:
                # Plan will resume after interrupt plan finishes executing
                self._current_plan.status = PlanStatus.PREEMPTED
                logger.info(f"Preempted plan {self._current_plan.plan_id}")
                await self.add_plan(self._current_plan)
            else:
                # Plan has failed, place it on blackboard to be remediated
                self._current_plan.status = PlanStatus.PAUSED
                logger.info(f"Paused plan {self._current_plan.plan_id}")
                await self._blackboard.set_paused_plan(self._current_plan)

            await self._clear_current_plan()

        except Exception as e:
            logger.exception(f"Failed to pause and requeue plan: {e}")

    # ---------------------
    # Lifecycle Methods
    # ---------------------
    async def start(self, tick_interval: float = DEFAULT_TICK_INTERVAL) -> None:
        """Start orchestrator in background."""
        if self._loop_running:
            logger.warning("Orchestrator already running")
            return
        if self._task and not self._task.done():
            logger.warning("Previous task still active")
            return

        self._loop_running = True
        self._task = asyncio.create_task(self._run_loop(tick_interval))
        logger.info("Orchestrator started.")

    async def stop(self) -> None:
        """Stop orchestrator loop and persist state."""
        logger.info("Stop signal received. Flushing state...")
        self._loop_running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("Orchestrator task cancelled")

        try:
            if self._current_plan:
                await self._blackboard.set("last_active_plan", self._current_plan.model_dump())
        except Exception as e:
            logger.exception(f"Failed to flush orchestrator state: {e}")
