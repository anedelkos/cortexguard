import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any, ClassVar
from uuid import uuid4

from opentelemetry import trace

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
from kitchenwatch.edge.utils.metrics import (
    anomalies_total,
    component_duration_ms,
    policy_escalations_total,
)
from kitchenwatch.edge.utils.tracing import BaseTraceSink, TraceSink

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("cortexguard.policy_agent")

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
    _MAX_ANOMALIES_PER_TICK = 3
    _PROCESSED_CACHE_SIZE = 1000

    def __init__(
        self,
        blackboard: Blackboard,
        capability_registry: CapabilityRegistry,
        policy_engine: BasePolicyEngine,
        mayday_agent: MaydayAgent,
        trace_sink: BaseTraceSink | None = None,
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

        self._processed_anomalies: deque[str] = deque(maxlen=self._PROCESSED_CACHE_SIZE)

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

    def _validate_policy_actions(
        self, policy: RemediationPolicy, *, remove_invalid_steps: bool = True
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate all actions in a remediation policy.

        Returns (is_valid, details) where details contains lists of invalid step indices
        and a short message suitable for traces/logging.

        - If remove_invalid_steps is True, invalid corrective steps are removed from the policy.
        """
        with tracer.start_as_current_span("policy_agent.validation") as span:
            span.set_attribute("policy.id", policy.policy_id)

            invalid_indices: list[int] = []
            invalid_descriptions: list[str] = []

            for idx, step in enumerate(list(policy.corrective_steps)):
                action = getattr(step, "action", None)
                if not action:
                    continue

                try:
                    valid = self._validate_action(action)
                except Exception as exc:
                    valid = False
                    logger.exception(
                        "Exception while validating action for policy %s step %d: %s",
                        policy.policy_id,
                        idx,
                        exc,
                    )

                if not valid:
                    tool_id = action.arguments.get("tool_id", "N/A")
                    capability = action.action_name
                    invalid_indices.append(idx)
                    invalid_descriptions.append(f"{tool_id}.{capability}")
                    logger.error(
                        "Supervisor Alert: Invalid action '%s.%s' detected in policy %s (step %d)",
                        tool_id,
                        capability,
                        policy.policy_id,
                        idx,
                    )

            if not invalid_indices:
                return True, {"message": "all_actions_valid"}

            # Mutate policy once: mark escalation, update risk and reasoning trace
            policy.escalation_required = True
            policy.risk_assessment = "CRITICAL - Invalid Action Generated (Validation Failure)"
            policy.reasoning_trace = (
                (policy.reasoning_trace or "")
                + " [VALIDATION FAILURE: Invalid actions detected: "
                + ", ".join(invalid_descriptions)
                + "]"
            )

            # Optionally remove or neutralize invalid steps
            if remove_invalid_steps:
                # Remove by index in reverse order to keep indices valid
                for idx in sorted(invalid_indices, reverse=True):
                    try:
                        policy.corrective_steps.pop(idx)
                    except Exception:
                        logger.exception(
                            "Failed to remove invalid corrective step %d from policy %s",
                            idx,
                            policy.policy_id,
                        )

            details = {
                "invalid_indices": invalid_indices,
                "invalid_actions": invalid_descriptions,
                "removed_invalid_steps": remove_invalid_steps,
            }

            return False, details

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
        with tracer.start_as_current_span("policy_agent.llm_request") as span:
            span.set_attribute("model", self._policy_engine.model_name())

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
                    self._scene_graph_to_compact_prompt(sg)
                    if sg
                    else "No vision context available."
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
                span.set_attribute("llm.duration_ms", duration_ms)

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
                span.add_event("LLM_FALLBACK", {"error": str(e)})
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

    def _policy_for_value_freeze(
        self, anomaly: AnomalyEvent, state_estimate: StateEstimate
    ) -> RemediationPolicy:
        policy = RemediationPolicy(
            policy_id=str(uuid4()),
            trigger_event=anomaly,
            reasoning_trace="Sensor freeze detected; reducing system speed to 50%.",
            risk_assessment="LOW",
            corrective_steps=[
                PlanStep(
                    id="1",
                    description="Reduce system speed to 50% due to sensor freeze.",
                    status=StepStatus.PENDING,
                    action=AgentToolCall(
                        action_name="set_speed_limit",
                        arguments={"tool_id": self.REMEDIATION_TOOL_ID, "speed_factor": 0.5},
                    ),
                )
            ],
            escalation_required=False,
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
        with tracer.start_as_current_span("policy_agent.dispatch") as span:
            span.set_attribute("anomaly.key", anomaly.key)
            span.set_attribute("anomaly.severity", anomaly.severity.name)

            # 1. Highest Priority: Critical Safety Shutdown (Rules-Based)
            if anomaly.severity == AnomalySeverity.HIGH:
                return self._generate_critical_safety_policy(anomaly, context)

            # 2. S2.3 Value Freeze → degraded mode policy
            if anomaly.key.endswith("_value_freeze"):
                return self._policy_for_value_freeze(anomaly, context)

            # 3. Medium Priority: Specific Known Remediation (Rules-Based)
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

    async def _escalate_to_mayday_agent(self, policy: RemediationPolicy) -> None:
        policy_escalations_total.inc()

        with tracer.start_as_current_span("policy_agent.escalation") as span:
            span.set_attribute("policy.id", policy.policy_id)

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
                    span.add_event("CLOUD_PLAN_RECEIVED")
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
                    span.add_event("CLOUD_NO_PLAN")

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

    def _build_safe_fallback_policy(self, anomaly: AnomalyEvent) -> RemediationPolicy:
        """
        Build a minimal, safe remediation policy: pause system, notify operator.
        Adapt to your RemediationPolicy constructor.
        """

        pause_action = AgentToolCall(
            action_name="pause_system", arguments={"reason": "policy_validation_failed"}
        )
        notify_action = AgentToolCall(
            action_name="notify_operator", arguments={"anomaly_id": anomaly.id}
        )
        steps = [
            PlanStep(action=pause_action, description="Pause system for safety"),
            PlanStep(action=notify_action, description="Notify operator for manual review"),
        ]
        policy_id = f"fallback-{anomaly.id}"
        return RemediationPolicy(
            policy_id=policy_id,
            corrective_steps=steps,
            escalation_required=True,
            risk_assessment="fallback",
            reasoning_trace="fallback generated due to policy validation failure",
            trigger_event=anomaly,
        )

    async def _fetch_context_for_anomaly(
        self, anomaly: AnomalyEvent
    ) -> tuple[StateEstimate | None, Plan | None]:
        context, active_plan = await self._fetch_context()

        if not context:
            logger.warning(
                f"No StateEstimate context available for anomaly {anomaly.key}. "
                "Skipping policy generation."
            )
            return None, None

        active_plan_id = getattr(active_plan, "plan_id", "None")
        logger.info(
            f"Policy Agent triggered by anomaly '{anomaly.key}' "
            f"(Severity: {anomaly.severity.name}). Active Plan: {active_plan_id}"
        )

        return context, active_plan

    async def _generate_policy_for_anomaly(
        self,
        anomaly: AnomalyEvent,
        context: StateEstimate,
        active_plan: Plan | None,
        snapshot: FusionSnapshot | None,
    ) -> RemediationPolicy | None:
        start = time.perf_counter()

        policy = await self._dispatch_policy_generation(anomaly, context, active_plan, snapshot)

        if not policy:
            logger.info(f"No policy generated for anomaly {anomaly.key} (likely LOW severity).")
            return None

        duration_ms = (time.perf_counter() - start) * 1000.0
        component_duration_ms.labels(component="policy_generation").observe(duration_ms)

        return policy

    async def _validate_or_fallback(
        self, anomaly: AnomalyEvent, policy: RemediationPolicy
    ) -> RemediationPolicy:
        is_valid, details = self._validate_policy_actions(policy, remove_invalid_steps=True)

        if is_valid:
            return policy

        await self._trace_sink.post_trace_entry(
            source=self,
            event_type="POLICY_VALIDATION_FAILED",
            reasoning_text=f"Policy {policy.policy_id} failed supervisor validation.",
            metadata={"policy_id": policy.policy_id, "anomaly_id": anomaly.id, **details},
        )

        fallback = self._build_safe_fallback_policy(anomaly)
        fallback.risk_assessment += " Failed policy risk assessment: " + policy.risk_assessment
        fallback.reasoning_trace += " Failed policy reasoning trace:" + policy.reasoning_trace

        return fallback

    async def _publish_policy(self, anomaly: AnomalyEvent, policy: RemediationPolicy) -> None:
        component_duration_ms.labels(component="policy_publish").observe(0)

        await self._blackboard.set_remediation_policy(policy)
        self._policies_generated += 1

        escalation = policy.escalation_required
        if escalation:
            self._escalations_triggered += 1

        event_type = "ESCALATE" if escalation else "POLICY_GENERATED"

        await self._trace_sink.post_trace_entry(
            source=self,
            event_type=event_type,
            reasoning_text=(
                f"Remediation Policy {policy.policy_id} generated. "
                f"Escalation: {escalation}. Steps: {len(policy.corrective_steps)}."
            ),
            metadata={
                "policy_id": policy.policy_id,
                "anomaly_id": anomaly.id,
                "source": "LLM" if policy.policy_id.startswith("llm-") else "Rules",
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
            f"Escalation: {escalation}"
        )

    async def _handle_anomaly_event(self, anomaly: AnomalyEvent) -> None:
        """
        Orchestrate policy generation, validation, and publishing.
        """
        anomalies_total.inc()

        with tracer.start_as_current_span("policy_agent.handle_anomaly") as span:
            span.set_attribute("anomaly.key", anomaly.key)
            span.set_attribute("anomaly.severity", anomaly.severity.name)
            span.set_attribute("anomaly.id", anomaly.id)

            if anomaly.id in self._processed_anomalies:
                return

            self._processed_anomalies.append(anomaly.id)
            self._anomalies_processed += 1

            # 1. Fetch context
            context, active_plan = await self._fetch_context_for_anomaly(anomaly)
            if context is None:
                return

            # 2. Generate policy
            snapshot = await self._blackboard.get_fusion_snapshot()
            policy = await self._generate_policy_for_anomaly(
                anomaly, context, active_plan, snapshot
            )
            if policy is None:
                return

            # 3. Validate or fallback
            policy = await self._validate_or_fallback(anomaly, policy)

            # 4. Publish
            await self._publish_policy(anomaly, policy)

            # 5. Escalate if needed
            if policy.escalation_required and self._mayday_agent is not None:
                await self._escalate_to_mayday_agent(policy)

    async def _process_active_anomalies_tick(self) -> list[str]:
        """
        Perform one policy-agent tick: fetch active anomalies, process up to
        MAX_ANOMALIES_PER_TICK, and return the list of anomaly ids processed.
        """
        tick_start = time.perf_counter()
        processed_ids: list[str] = []

        with tracer.start_as_current_span("policy_agent.tick") as span:
            try:
                active_anomalies = await self._blackboard.get_active_anomalies()
                span.set_attribute("active_anomalies.count", len(active_anomalies or {}))
            except Exception as exc:
                logger.exception("Failed to fetch active anomalies: %s", exc)
                return processed_ids

        if not active_anomalies:
            return processed_ids

        sorted_anomalies = sorted(
            active_anomalies.values(),
            key=lambda a: a.severity.value,
            reverse=True,
        )

        processed_this_tick = 0
        for anomaly in sorted_anomalies:
            if processed_this_tick >= self._MAX_ANOMALIES_PER_TICK:
                break

            if anomaly.id not in self._processed_anomalies and anomaly.severity in (
                AnomalySeverity.HIGH,
                AnomalySeverity.MEDIUM,
            ):
                try:
                    await self._handle_anomaly_event(anomaly)
                except Exception as handle_exc:
                    logger.exception("Error handling anomaly %s: %s", anomaly.id, handle_exc)
                    # continue to next anomaly rather than aborting the tick
                else:
                    processed_ids.append(anomaly.id)
                    processed_this_tick += 1

                    if anomaly.severity == AnomalySeverity.HIGH:
                        logger.warning(
                            "HIGH severity anomaly processed, allowing Orchestrator to react"
                        )
                        break

        duration_ms = (time.perf_counter() - tick_start) * 1000.0
        component_duration_ms.labels(component="policy_agent_tick").observe(duration_ms)

        return processed_ids

    async def _run_loop(self, tick_interval: float) -> None:
        """Main Policy Agent loop: polls blackboard for active anomalies."""
        logger.info("Policy Agent loop started (Polling for Anomalies)")
        try:
            while self._loop_running:
                try:
                    await self._process_active_anomalies_tick()
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
