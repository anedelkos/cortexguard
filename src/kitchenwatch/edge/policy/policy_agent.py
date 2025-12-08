import asyncio
import logging
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any, ClassVar
from uuid import uuid4

from kitchenwatch.core.interfaces.base_policy_engine import BasePolicyEngine
from kitchenwatch.edge.mayday_agent import MaydayAgent
from kitchenwatch.edge.models.agent_tool_call import AgentToolCall
from kitchenwatch.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.capability_registry import CapabilityRegistry
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.mayday_packet import SystemHealth
from kitchenwatch.edge.models.plan import Plan, PlanSource, PlanStep, StepStatus
from kitchenwatch.edge.models.remediation_policy import RemediationPolicy
from kitchenwatch.edge.models.state_estimate import StateEstimate
from kitchenwatch.edge.utils.tracing import BaseTraceSink, TraceSink

logger = logging.getLogger(__name__)

PolicyFunc = Callable[[AnomalyEvent, StateEstimate], RemediationPolicy]


class PolicyAgent:
    """
    Local Policy Agent for safety-critical anomaly remediation.

    The Agent delegates policy generation for unknown or complex anomalies to the
    injected BasePolicyEngine (LLM), while handling critical and known anomalies
    via rules-based dispatch.
    """

    # Configuration constants
    DEFAULT_TICK_INTERVAL = 0.5
    REMEDIATION_TOOL_ID: ClassVar[str] = "system_controls"
    THERMAL_TOOL_ID: ClassVar[str] = "thermal_management"
    MAX_ANOMALIES_PER_TICK = 3
    PROCESSED_CACHE_SIZE = 1000

    def __init__(
        self,
        blackboard: Blackboard,
        capability_registry: CapabilityRegistry,
        policy_engine: BasePolicyEngine,
        trace_sink: BaseTraceSink | None = None,
        mayday_agent: MaydayAgent | None = None,
    ) -> None:
        """Initialize Policy Agent."""
        self._blackboard = blackboard
        self._trace_sink: BaseTraceSink = (
            trace_sink if trace_sink is not None else TraceSink(blackboard=self._blackboard)
        )
        self._capability_registry = capability_registry
        self._policy_engine = policy_engine
        self._mayday_agent = mayday_agent

        self._loop_running: bool = False
        self._task: asyncio.Task[Any] | None = None

        self._processed_anomalies: deque[str] = deque(maxlen=self.PROCESSED_CACHE_SIZE)

        # Metrics for observability
        self._policies_generated = 0
        self._anomalies_processed = 0
        self._escalations_triggered = 0

        # Policy Dispatch Table: Maps ANOMALY KEY to a specific handler function.
        # Handlers are now guaranteed to return RemediationPolicy
        self._policy_dispatcher: dict[str, PolicyFunc] = {
            "overheat_warning": self._policy_for_overheat,
        }

        logger.info(f"Policy Agent initialized. LLM Engine: {self._policy_engine.model_name()}")

    def _scene_graph_summary_to_prompt(self, summary: list[dict[str, Any]]) -> str:
        if not summary:
            return "No scene graph summary available."
        lines = []
        for item in summary[:10]:
            obj_id = item.get("id", "unknown")
            label = item.get("label", "unknown")
            lines.append(f"- {obj_id}: {label}")
        return "Scene objects:\n" + "\n".join(lines)

    def _scene_graph_to_compact_prompt(self, sg: Any, max_objects: int = 20) -> str:
        """
        Convert a full SceneGraph into a compact, capped prompt fragment.
        sg is expected to have 'objects' and 'relationships' attributes.
        """
        if not sg:
            return "No scene graph available."

        objs = getattr(sg, "objects", []) or []
        rels = getattr(sg, "relationships", []) or []

        lines: list[str] = []
        # Objects: include id, label, and a small set of numeric props (distance/confidence)
        for o in objs[:max_objects]:
            obj_id = getattr(o, "id", "unknown")
            label = getattr(o, "label", "unknown")
            props = getattr(o, "properties", {}) or {}
            dist = props.get("distance_m")
            conf = props.get("confidence")
            dist_str = f"{float(dist):.2f}m" if isinstance(dist, (int, float)) else "N/A"
            conf_str = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "N/A"
            lines.append(f"- {obj_id}: {label}; distance={dist_str}; conf={conf_str}")

        # Relationships: include simple triples
        for r in rels[:max_objects]:
            src = getattr(r, "source_id", "unknown")
            rel = getattr(r, "relationship", "unknown")
            tgt = getattr(r, "target_id", "unknown")
            lines.append(f"- REL: {src} {rel} {tgt}")

        if len(objs) > max_objects:
            lines.append(f"... and {len(objs) - max_objects} more objects omitted")

        return (
            "SceneGraph facts:\n" + "\n".join(lines) if lines else "No scene graph facts available."
        )

    async def _fetch_context(self) -> tuple[StateEstimate | None, Plan | None]:
        """Fetch latest StateEstimate and Active Plan (Working Memory)."""
        state_estimate = await self._blackboard.get_latest_state_estimate()
        active_plan = await self._blackboard.get_current_plan()
        return state_estimate, active_plan

    def _validate_action(self, action: AgentToolCall) -> bool:
        """
        Validate that the action's capability (function_name) exists in the
        CapabilityRegistry.
        """
        try:
            self._capability_registry.get_function_schema(action.action_name)
            return True
        except KeyError:
            return False

    def _supervise_policy_actions(self, policy: RemediationPolicy) -> bool:
        """
        [FORMALIZED SUPERVISOR ROLE]
        Iterates over all steps in the generated policy and validates their actions.
        Mutates the policy if invalid actions are found.

        Returns: True if all actions are valid, False otherwise.
        """
        validation_successful = True

        for step in policy.corrective_steps:
            if step.action and not self._validate_action(step.action):
                tool_id = step.action.arguments.get("tool_id", "N/A")
                capability = step.action.action_name
                logger.error(
                    f"Supervisor Alert: Invalid action '{tool_id}.{capability}' detected in policy {policy.policy_id}"
                )

                # If validation fails, force escalation and update policy fields
                policy.escalation_required = True
                policy.risk_assessment = "CRITICAL - Invalid Action Generated (Validation Failure)"
                policy.reasoning_trace += (
                    f" [VALIDATION FAILURE: Capability '{capability}' is unregistered/forbidden.]"
                    " WARNING: Generated plan contained invalid actions"
                )
                validation_successful = False

        return validation_successful

    def _get_llm_action_catalog(self) -> str:
        """
        Fetches the complete list of physical tools and their capability schemas
        from the CapabilityRegistry, formatted as a JSON string for the LLM prompt.
        """
        try:
            return self._capability_registry.get_llm_tool_catalog()
        except Exception as e:
            logger.error(f"Failed to fetch tool catalog from CapabilityRegistry: {e}")
            return "[]"

    # --- RULES-BASED POLICIES (Now return RemediationPolicy objects) ---

    def _generate_critical_safety_policy(
        self, anomaly: AnomalyEvent, context: StateEstimate
    ) -> RemediationPolicy:
        """
        Generate policy for HIGH severity anomalies (emergency shutdown).
        Always returns RemediationPolicy.
        """
        reasoning = (
            f"Critical safety failure detected: {anomaly.key}. "
            f"System must enter safe idle immediately. Policy is non-negotiable shutdown."
        )

        policy_steps = [
            PlanStep(
                id="1",
                description="Initiate critical shutdown of all drive motors.",
                status=StepStatus.PENDING,
                action=AgentToolCall(
                    action_name="shutdown_motors",
                    arguments={"tool_id": self.REMEDIATION_TOOL_ID, "force": True},
                ),
            ),
            PlanStep(
                id="2",
                description="Move system to a safe, home position using safe halt mode.",
                status=StepStatus.PENDING,
                action=AgentToolCall(
                    action_name="move_to_home",
                    arguments={"tool_id": self.REMEDIATION_TOOL_ID, "mode": "safe_halt"},
                ),
            ),
        ]

        return RemediationPolicy(
            policy_id=str(uuid4()),
            trigger_event=anomaly,
            reasoning_trace=reasoning,
            risk_assessment="LOW - Emergency Halt Only",
            corrective_steps=policy_steps,
            escalation_required=True,
            created_at=datetime.now(),
        )

    def _policy_for_overheat(
        self, anomaly: AnomalyEvent, state_estimate: StateEstimate
    ) -> RemediationPolicy:
        """Specific handler for the 'overheat_warning' anomaly (Rules-Based)."""
        temp = anomaly.metadata.get("temp", "N/A")
        reasoning = (
            f"Medium severity overheat detected (Temp: {temp} C). "
            f"We will pause and initiate a cooldown cycle."
        )

        policy_steps = [
            PlanStep(
                id="1",
                description="Temporarily pause the currently executing plan.",
                status=StepStatus.PENDING,
                action=AgentToolCall(
                    action_name="pause_execution",
                    arguments={
                        "tool_id": self.REMEDIATION_TOOL_ID,
                    },
                ),
            ),
            PlanStep(
                id="2",
                description="Run the active cooling system for 30 seconds.",
                status=StepStatus.PENDING,
                action=AgentToolCall(
                    action_name="cooldown_cycle",
                    arguments={"tool_id": self.THERMAL_TOOL_ID, "duration": 30},
                ),
            ),
        ]

        return RemediationPolicy(
            policy_id=str(uuid4()),
            trigger_event=anomaly,
            reasoning_trace=reasoning,
            risk_assessment="MEDIUM - Requires Active Management",
            corrective_steps=policy_steps,
            escalation_required=False,
            created_at=datetime.now(),
        )

    # --- LLM DELEGATION ---

    async def _generate_unknown_medium_policy(
        self,
        anomaly: AnomalyEvent,
        context: StateEstimate,
        active_plan: Plan | None,
        snapshot: FusionSnapshot | None = None,
    ) -> RemediationPolicy:
        """
        Delegates to LLM for unknown medium severity anomalies, providing
        the current active plan as 'Working Memory' context.
        """
        logger.info(
            f"Delegating unknown MEDIUM policy generation for anomaly {anomaly.key} "
            f"to LLM Engine: {self._policy_engine.model_name()}"
        )

        # 1. Fetch the authoritative tool catalog
        action_catalog_json = self._get_llm_action_catalog()

        # 2. Pass Working Memory (Active Plan) to the Policy Engine for context
        # Defensive check for 'current_step_id' which was reported as a missing attribute.
        current_step_id = getattr(active_plan, "current_step_id", "Unknown")

        active_plan_context = (
            f"Active Plan ID: {active_plan.plan_id}, Current Step: {current_step_id}"
            if active_plan
            else "No active plan currently running."
        )

        # 3. Generate the policy
        start = datetime.now()
        await self._trace_sink.post_trace_entry(
            source=self,
            event_type="POLICY_LLM_REQUEST",
            reasoning_text="LLM policy request",
            metadata={"model": self._policy_engine.model_name()},
            refs={"anomaly_id": anomaly.id},
        )

        if snapshot and snapshot.sensors.get("scene_graph_summary"):
            vision_context = self._scene_graph_summary_to_prompt(
                snapshot.sensors["scene_graph_summary"]
            )
        else:
            sg = await self._blackboard.get_scene_graph()
            vision_context = (
                self._scene_graph_to_compact_prompt(sg) if sg else "No vision context available."
            )

        try:
            policy = await self._policy_engine.generate_policy(
                event=anomaly,
                context=context,
                # Arguments passed to LLM engine must match the required signature (LSP)
                action_catalog_json=action_catalog_json,
                active_plan_context=active_plan_context,
                vision_context=vision_context,
            )
            duration_ms = int((datetime.now() - start).total_seconds() * 1000)
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="POLICY_LLM_RESPONSE",
                reasoning_text="LLM policy response",
                metadata={"duration_ms": duration_ms},
                refs={"anomaly_id": anomaly.id},
            )
        except Exception as e:
            # Log and trace the LLM failure, then return a conservative fail-safe policy
            logger.exception("LLM policy generation failed: %s", e)
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="POLICY_LLM_ERROR",
                reasoning_text="LLM policy generation failed; returning fail-safe policy",
                metadata={"error": str(e)},
                refs={"anomaly_id": anomaly.id},
            )

            # Construct a deterministic fail-safe RemediationPolicy
            fail_reason = (
                "CRITICAL PARSING/LLM FAILURE: Unable to generate policy from LLM. "
                "Issuing fail-safe emergency remediation to preserve system safety."
            )
            fail_steps = [
                PlanStep(
                    id="1",
                    description="Perform emergency shutdown of hazardous actuators.",
                    status=StepStatus.PENDING,
                    action=AgentToolCall(
                        action_name="EMERGENCY_SHUTDOWN",
                        arguments={"tool_id": self.REMEDIATION_TOOL_ID},
                    ),
                )
            ]
            policy = RemediationPolicy(
                policy_id=str(uuid4()),
                trigger_event=anomaly,
                reasoning_trace=fail_reason,
                risk_assessment="HIGH - System safety cannot be guaranteed.",
                corrective_steps=fail_steps,
                escalation_required=True,
                created_at=datetime.now(),
            )

        return policy

    # --- MAIN DISPATCHER ---

    async def _dispatch_policy_generation(
        self,
        anomaly: AnomalyEvent,
        context: StateEstimate,
        active_plan: Plan | None,
        snapshot: FusionSnapshot | None,
    ) -> RemediationPolicy | None:
        """
        Dispatches the anomaly to the correct policy generation function.
        Guarantees a return of RemediationPolicy or None.
        """
        # 1. Highest Priority: Critical Safety Shutdown (Rules-Based)
        if anomaly.severity == AnomalySeverity.HIGH:
            return self._generate_critical_safety_policy(anomaly, context)

        # 2. Medium Priority: Specific Known Remediation (Rules-Based)
        handler = self._policy_dispatcher.get(anomaly.key)

        if handler:
            # Type checker now accepts this call because PolicyFunc no longer
            # includes the implicit 'self'
            return handler(anomaly, context)

        # 3. Default Medium: Unknown Anomaly (LLM-Assisted Plan)
        if anomaly.severity == AnomalySeverity.MEDIUM:
            # We pass the active_plan here to close the Working Memory gap
            return await self._generate_unknown_medium_policy(
                anomaly, context, active_plan, snapshot
            )

        # 4. Low Priority: Ignored
        return None

    async def _handle_anomaly_event(self, anomaly: AnomalyEvent) -> None:
        """
        Orchestrate policy generation, validation, and publishing.
        """
        if anomaly.id in self._processed_anomalies:
            return

        self._processed_anomalies.append(anomaly.id)
        self._anomalies_processed += 1

        # 1. Fetch Context (State Estimate + Working Memory / Active Plan)
        context, active_plan = await self._fetch_context()

        if not context:
            logger.warning(
                f"No StateEstimate context available for anomaly {anomaly.key}. Skipping policy generation."
            )
            return

        # Defensive access for active_plan_id logging
        active_plan_id = (
            active_plan.plan_id if active_plan and hasattr(active_plan, "plan_id") else "None"
        )
        logger.info(
            f"Policy Agent triggered by anomaly '{anomaly.key}' "
            f"(Severity: {anomaly.severity.name}). Active Plan: {active_plan_id}"
        )

        # 2. Generate Policy (Standardized Output: RemediationPolicy | None)
        snapshot = await self._blackboard.get_fusion_snapshot()
        policy = await self._dispatch_policy_generation(anomaly, context, active_plan, snapshot)

        if not policy:
            logger.info(f"No policy generated for anomaly {anomaly.key} (likely LOW severity).")
            return

        policy_source = "LLM" if policy.policy_id.startswith("llm-") else "Rules"

        # 3. Supervisor Step: Validate Actions (Safety Gate)
        # This checks the policy regardless of source (Rules or LLM)
        self._supervise_policy_actions(policy)

        # 4. Publish to Blackboard
        await self._blackboard.set_remediation_policy(policy)
        self._policies_generated += 1
        escalation = policy.escalation_required  # Read final escalation state

        if escalation:
            self._escalations_triggered += 1

        # 5. Log to Trace
        await self._trace_sink.post_trace_entry(
            source=self,
            event_type="POLICY_GENERATED",
            reasoning_text=f"Remediation Policy {policy.policy_id} generated by {policy_source}. Escalation: {escalation}. Steps: {len(policy.corrective_steps)}.",
            metadata={
                "policy_id": policy.policy_id,
                "anomaly_id": anomaly.id,
                "source": policy_source,
                "risk_assessment": policy.risk_assessment,
                "escalation_required": escalation,
                "full_reasoning_trace": policy.reasoning_trace,
                "corrective_actions": [
                    {
                        "action": step.action.action_name,
                        "tool_id": step.action.arguments.get("tool_id"),
                    }
                    for step in policy.corrective_steps
                    if step.action
                ],
            },
        )

        logger.warning(
            f"Generated policy {policy.policy_id} for anomaly {anomaly.key}. "
            f"Source: {policy_source}. "
            f"Escalation: {escalation}"
        )

        if escalation and self._mayday_agent is not None:
            try:
                # Build a compact health snapshot (implement or fetch from your health monitor)
                health = SystemHealth(
                    cpu_load_pct=None,
                    net_rtt_ms=None,
                    packet_loss_pct=None,
                    disk_pressure_pct=None,
                )

                # Build the MaydayPacket from the policy + blackboard context
                packet = await self._mayday_agent.build_packet_from_policy(
                    policy=policy,
                    blackboard=self._blackboard,
                    health=health,
                    trace_id=policy.policy_id,
                )

                cloud_plan = await self._mayday_agent.send_escalation(packet)

                if cloud_plan:
                    cloud_plan.source = PlanSource.CLOUD_AGENT
                    cloud_plan.trace_id = packet.trace_id

                    await self._blackboard.set_current_plan(cloud_plan)

                    await self._trace_sink.post_trace_entry(
                        source=self,
                        event_type="CLOUD_PLAN_RECEIVED",
                        reasoning_text=f"Cloud remediation plan received for policy {policy.policy_id}",
                        metadata={"policy_id": policy.policy_id, "plan_id": cloud_plan.plan_id},
                    )
                else:
                    await self._trace_sink.post_trace_entry(
                        source=self,
                        event_type="CLOUD_NO_PLAN",
                        reasoning_text=f"Cloud returned no plan for policy {policy.policy_id}; falling back to local remediation.",
                        metadata={"policy_id": policy.policy_id},
                    )

            except Exception as exc:
                logger.exception(
                    "Mayday escalation failed for policy %s: %s", policy.policy_id, exc
                )
                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="MAYDAY_ERROR",
                    reasoning_text="Failed to send Mayday escalation; falling back to local remediation",
                    metadata={"policy_id": policy.policy_id, "error": str(exc)},
                )

    async def _run_loop(self, tick_interval: float) -> None:
        """Main Policy Agent loop: polls blackboard for active anomalies."""
        logger.info("Policy Agent loop started (Polling for Anomalies)")
        try:
            while self._loop_running:
                try:
                    active_anomalies = await self._blackboard.get_active_anomalies()

                    if active_anomalies:
                        # Sort by severity (highest first)
                        sorted_anomalies = sorted(
                            active_anomalies.values(),
                            key=lambda a: a.severity.value,
                            reverse=True,
                        )

                        processed_this_tick = 0
                        for anomaly in sorted_anomalies:
                            if processed_this_tick >= self.MAX_ANOMALIES_PER_TICK:
                                break

                            # Only process unhandled HIGH/MEDIUM severity anomalies
                            if (
                                anomaly.id not in self._processed_anomalies
                                and anomaly.severity
                                in (
                                    AnomalySeverity.HIGH,
                                    AnomalySeverity.MEDIUM,
                                )
                            ):
                                await self._handle_anomaly_event(anomaly)
                                processed_this_tick += 1

                                # Break immediately after HIGH severity to allow orchestrator to react
                                if anomaly.severity == AnomalySeverity.HIGH:
                                    logger.warning(
                                        "HIGH severity anomaly processed, "
                                        "allowing Orchestrator to react"
                                    )
                                    break

                    await asyncio.sleep(tick_interval)

                except asyncio.CancelledError:
                    raise
                except Exception as loop_err:
                    logger.exception(f"Unexpected error in Policy Agent tick: {loop_err}")

        except Exception as e:
            logger.critical(f"Fatal Policy Agent error: {e}", exc_info=True)
        finally:
            logger.info("Policy Agent stopped")
            self._loop_running = False

    async def start(self, tick_interval: float = DEFAULT_TICK_INTERVAL) -> None:
        """Start Policy Agent in background."""
        if self._loop_running:
            logger.warning("Policy Agent already running")
            return
        if self._task and not self._task.done():
            logger.warning("Previous task still active")
            return

        self._loop_running = True
        self._task = asyncio.create_task(self._run_loop(tick_interval))
        logger.info("Policy Agent started")

    async def stop(self) -> None:
        """Stop Policy Agent loop gracefully."""
        logger.info("Policy Agent stop signal received")
        self._loop_running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("Policy Agent task cancelled")

    def get_metrics(self) -> dict[str, int]:
        """Get agent metrics for monitoring."""
        return {
            "policies_generated": self._policies_generated,
            "anomalies_processed": self._anomalies_processed,
            "escalations_triggered": self._escalations_triggered,
            "processed_cache_size": len(self._processed_anomalies),
        }
