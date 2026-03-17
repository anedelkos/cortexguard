from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from opentelemetry import trace

from cortexguard.edge.arbiter import Arbiter
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.plan import Plan, PlanStatus, PlanType, StepStatus
from cortexguard.edge.models.reasoning_trace_entry import TraceSeverity
from cortexguard.edge.models.remediation_policy import RemediationPolicy
from cortexguard.edge.safety_agent import SafetyAgent, SafetyCommand
from cortexguard.edge.utils.async_priority_queue import AsyncPriorityQueue
from cortexguard.edge.utils.metrics import component_duration_ms
from cortexguard.edge.utils.tracing import BaseTraceSink, TraceSink

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("cortexguard.orchestrator")

HIGH_PRIORITY = 0  # Highest priority for urgent remediation plans


class Orchestrator:
    """
    Orchestrator-centric async plan scheduler for CortexGuard Edge.

    Responsibilities:
    - Manage plan queue and priorities
    - Observe blackboard for step completion/failure
    - Handle urgent remediation plans
    - Advance, pause, or resume plans based on blackboard state
    """

    def __init__(
        self,
        blackboard: Blackboard,
        arbiter: Arbiter,
        safety_agent: SafetyAgent,
        tick_interval: float,
        trace_sink: BaseTraceSink | None = None,
    ) -> None:
        self._blackboard = blackboard
        self._arbiter = arbiter
        self._safety_agent = safety_agent
        self._tick_interval = tick_interval
        self._trace_sink: BaseTraceSink = (
            trace_sink if trace_sink is not None else TraceSink(blackboard=self._blackboard)
        )
        # The priority queue uses the Plan's context.priority (lower number = higher priority)
        self._plan_queue = AsyncPriorityQueue[Plan]()
        self._current_plan: Plan | None = None
        self._loop_running: bool = False
        self._task: asyncio.Task[None] | None = None

    # ---------------------
    # Plan Queue Methods
    # ---------------------
    async def add_plan(self, plan: Plan) -> None:
        """Add a new plan to the priority queue."""
        # Only PENDING or PREEMPTED plans should be added to the queue
        if plan.status not in (PlanStatus.PENDING, PlanStatus.PREEMPTED):
            logger.warning(
                f"Attempted to add plan {plan.plan_id} with status {plan.status.name} to queue. Skipping."
            )
            return

        try:
            await self._plan_queue.put(plan.context.priority, plan)
            logger.debug(f"Plan added: {plan.plan_id} with priority {plan.context.priority}")

            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="PLAN_QUEUED",
                reasoning_text=f"Plan queued {plan.plan_id}",
                metadata={"plan_id": plan.plan_id, "priority": plan.context.priority},
            )
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

            # Defensive check: Ignore plans found in the queue that are already finished
            if next_plan.status in (PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED):
                logger.warning(
                    f"Plan {next_plan.plan_id} retrieved from queue but is in terminal state ({next_plan.status.name}), discarding."
                )
                await self._clear_current_plan()
                return  # Discard and move on

            # Determine the trace event type
            event_type = (
                "PLAN_RESUMED" if next_plan.status == PlanStatus.PREEMPTED else "PLAN_STARTED"
            )

            if next_plan.status == PlanStatus.PENDING:
                logger.info(f"Starting plan {next_plan.plan_id}")
            elif next_plan.status == PlanStatus.PREEMPTED:
                logger.info(f"Resuming plan {next_plan.plan_id}")
            else:
                logger.error(
                    f"Plan found in unexpected status {next_plan.plan_id}-{next_plan.status.name}... discarding"
                )
                return

            if next_plan.steps:
                index = await self._blackboard.get_step_index_for_plan(next_plan.plan_id) or 0

                # Ensure index is within bounds for safety
                index = min(index, len(next_plan.steps) - 1)
                current_step = next_plan.steps[index]

                next_plan.status = PlanStatus.RUNNING
                self._current_plan = next_plan

                await self._blackboard.set_current_plan(next_plan)
                await self._blackboard.set_current_step(current_step)
                await self._blackboard.set_step_index_for_plan(next_plan.plan_id, index)

                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type=event_type,
                    reasoning_text=f"{'Resuming' if event_type == 'PLAN_RESUMED' else 'Starting'} plan {next_plan.plan_id}. Type: {next_plan.plan_type.name}.",
                    metadata={
                        "plan_id": next_plan.plan_id,
                        "plan_type": next_plan.plan_type.name,
                        "step_index": index,
                        "step_id": current_step.id,
                    },
                )
                # ------------------------------------------

            else:
                logger.warning(
                    f"Plan {next_plan.plan_id} has no steps, skipping and clearing current state."
                )
                await self._clear_current_plan()

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
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="PLAN_PREEMPTED",
                reasoning_text=f"Plan {self._current_plan.plan_id} preempted by higher priority plan {next_plan.plan_id}.",
                metadata={
                    "preempted_plan_id": self._current_plan.plan_id,
                    "new_plan_id": next_plan.plan_id,
                    "new_priority": next_plan.context.priority,
                },
            )
            # Pause the current plan and requeue it as PREEMPTED
            await self._pause_current_plan(interrupted=True)
            return next_plan

        return None

    async def _handle_remediation_policy(self) -> None:
        """
        Processes a RemediationPolicy posted by the PolicyAgent, converts it
        to a high-priority Plan, and queues it.
        """
        policy: RemediationPolicy | None = await self._blackboard.get_remediation_policy()

        if policy:
            logger.warning(
                f"Processing Remediation Policy {policy.policy_id} for anomaly: {policy.trigger_event.key}"
            )

            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="POLICY_RECEIVED",
                reasoning_text=f"Remediation Policy {policy.policy_id} received from PolicyAgent for anomaly {policy.trigger_event.key}.",
                metadata={
                    "policy_id": policy.policy_id,
                    "trigger_anomaly_key": policy.trigger_event.key,
                    "escalation_required": policy.escalation_required,
                },
            )
            # --------------------------------------------

            # Create a goal context with the highest priority and necessary fields
            remediation_goal = GoalContext(
                goal_id=str(uuid4()),
                user_prompt=f"[Auto-generated for Remediation: {policy.trigger_event.key}]",
                # The intent comes from the anomaly description itself
                intent=f"Remediate anomaly: {policy.trigger_event.key}",
                priority=HIGH_PRIORITY,
            )

            remediation_plan = Plan(
                plan_id=str(uuid4()),  # Must have a unique ID
                plan_type=PlanType.REMEDIATION,
                context=remediation_goal,
                steps=policy.corrective_steps,
                status=PlanStatus.PENDING,
            )

            await self.add_plan(remediation_plan)
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="REMEDIATION_QUEUED",
                reasoning_text=f"Remediation plan queued for anomaly {policy.trigger_event.key}",
                metadata={"policy_id": policy.policy_id, "plan_id": remediation_plan.plan_id},
                severity=TraceSeverity.WARN,
            )

            logger.info(
                f"Queued remediation plan {remediation_plan.plan_id} with priority {HIGH_PRIORITY}."
            )

            # Critical: Clear the policy from the Blackboard once handled
            await self._blackboard.clear_remediation_policy()

    async def _check_safety(self) -> SafetyCommand | None:
        state_estimate = await self._blackboard.get_latest_state_estimate()
        if state_estimate is None:
            return None

        safety_cmd = await self._safety_agent.execute_safety_check(state_estimate)
        if safety_cmd.action != "NOMINAL":
            await self._arbiter.emergency_stop(reason=safety_cmd.reason or "hazard detected")

        return safety_cmd

    # ---------------------
    # Main Loop
    # ---------------------
    async def _run_loop(self, tick_interval: float) -> None:
        """Main orchestrator loop observing blackboard."""
        logger.info("Orchestrator loop started.")
        try:
            while self._loop_running:
                tick_start = time.perf_counter()
                with tracer.start_as_current_span("orchestrator.tick") as span:
                    plan_id = getattr(self._current_plan, "plan_id", None)
                    span.set_attribute(
                        "current_plan_id", plan_id if plan_id is not None else "None"
                    )
                    queue_size = await self._plan_queue.size()
                    span.set_attribute("queue_size", queue_size)

                    try:
                        logger.debug(
                            f"Tick: current plan = {self._current_plan.plan_id if self._current_plan else 'None'}"
                        )

                        # 1. Run safety check before any plan/step logic
                        with tracer.start_as_current_span(
                            "orchestrator.safety_check"
                        ) as safety_span:
                            safety_cmd = await self._check_safety()
                            action = getattr(safety_cmd, "action", None)
                            safety_span.set_attribute(
                                "safety.action", action if action is not None else "None"
                            )

                        if safety_cmd is None or safety_cmd.action != "NOMINAL":
                            await asyncio.sleep(tick_interval)
                            continue

                        # 2. Check for immediate, high-priority remediation actions
                        with tracer.start_as_current_span("orchestrator.handle_remediation"):
                            await self._handle_remediation_policy()

                        # Start a plan if none running
                        if not self._current_plan:
                            with tracer.start_as_current_span("orchestrator.start_plan"):
                                await self._start_next_plan()

                        # Check preemption and step status
                        if self._current_plan:
                            # Check preemption again, as remediation may have been added
                            with tracer.start_as_current_span("orchestrator.preemption_check"):
                                urgent_plan = await self._check_for_preemption()

                            if urgent_plan:
                                with tracer.start_as_current_span("orchestrator.start_plan"):
                                    await self._start_next_plan(urgent_plan)
                                await asyncio.sleep(tick_interval)
                                continue

                            current_step = await self._blackboard.get_current_step()
                            step_status = (
                                getattr(current_step, "status", None) if current_step else None
                            )

                            if step_status and step_status in (
                                StepStatus.COMPLETED,
                                StepStatus.FAILED,
                            ):
                                with tracer.start_as_current_span("orchestrator.advance_plan"):
                                    await self._advance_plan_or_handle_failure()

                        await asyncio.sleep(tick_interval)

                    except asyncio.CancelledError:
                        logger.info("Orchestrator loop cancelled gracefully.")
                        raise
                    except Exception as loop_err:
                        logger.exception(f"Unexpected error in orchestrator tick: {loop_err}")
                    finally:
                        duration_ms = (time.perf_counter() - tick_start) * 1000.0
                        component_duration_ms.labels(component="orchestrator_tick").observe(
                            duration_ms
                        )

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

            # --- TRACE LOGGING: Plan Completed (No Steps) ---
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="PLAN_COMPLETED",
                reasoning_text=f"Plan {plan_id} completed successfully (no steps executed).",
                metadata={"plan_id": plan_id, "status": PlanStatus.COMPLETED.name},
            )
            # ---------------------------------------------

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

                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="PLAN_FAILED",
                    reasoning_text=f"Plan {plan_id} failed on step {current_step.id} after {current_step.attempts} attempt(s). PolicyAgent triggered.",
                    metadata={"plan_id": plan_id, "failed_step_id": current_step.id},
                )

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
                    await self._trace_sink.post_trace_entry(
                        source=self,
                        event_type="STEP_ADVANCED",
                        reasoning_text=f"Advanced plan {plan_id} from {current_step_index} to {next_index}",
                        metadata={
                            "plan_id": plan_id,
                            "from_index": current_step_index,
                            "to_index": next_index,
                            "step_id": next_step.id,
                        },
                    )
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

                    await self._trace_sink.post_trace_entry(
                        source=self,
                        event_type="PLAN_COMPLETED",
                        reasoning_text=f"Plan {plan_id} completed successfully after executing {len(self._current_plan.steps)} steps.",
                        metadata={"plan_id": plan_id, "status": PlanStatus.COMPLETED.name},
                    )

                    await self._clear_current_plan()

            # If status is None (Step not executed yet) or Running, do nothing and wait for executor
            # to update the step status.

        except Exception as e:
            logger.exception(f"Error advancing plan {plan_id}: {e}")
            self._current_plan.status = PlanStatus.FAILED
            # Revert to pause logic on critical error
            await self._pause_current_plan()

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
                # Plan has failed or encountered a critical error, place it on blackboard to be remediated
                self._current_plan.status = PlanStatus.PAUSED
                logger.info(f"Paused plan {self._current_plan.plan_id}")
                await self._blackboard.set_paused_plan(self._current_plan)

            await self._clear_current_plan()

        except Exception as e:
            logger.exception(f"Failed to pause and requeue plan: {e}")

    # ---------------------
    # Lifecycle Methods
    # ---------------------
    async def start(self, tick_interval: float | None = None) -> None:
        """Start orchestrator in background."""
        if self._loop_running:
            logger.warning("Orchestrator already running")
            return
        if self._task and not self._task.done():
            logger.warning("Previous task still active")
            return

        interval = tick_interval if tick_interval is not None else self._tick_interval
        self._loop_running = True
        self._task = asyncio.create_task(self._run_loop(interval))
        logger.info("Orchestrator started.")
        await self._trace_sink.post_trace_entry(
            source=self,
            event_type="ORCHESTRATOR_STARTED",
            reasoning_text="Orchestrator started",
            metadata={"process_id": str(uuid4()), "tick_interval": interval},
        )

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
