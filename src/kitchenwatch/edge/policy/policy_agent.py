import asyncio
import logging
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any, ClassVar, cast

from kitchenwatch.core.interfaces.base_policy_engine import BasePolicyEngine
from kitchenwatch.edge.models.agent_tool_call import AgentToolCall
from kitchenwatch.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.capability_registry import CapabilityRegistry
from kitchenwatch.edge.models.plan import PlanStep, StepStatus
from kitchenwatch.edge.models.reasoning_trace_entry import ReasoningTraceEntry
from kitchenwatch.edge.models.remediation_policy import RemediationPolicy
from kitchenwatch.edge.models.state_estimate import StateEstimate

logger = logging.getLogger(__name__)


# Type alias for rules-based policy generation functions.
# These handlers return simple tuples which are then assembled into a RemediationPolicy.
PolicyFunc = Callable[
    ["PolicyAgent", AnomalyEvent, StateEstimate],
    tuple[str, list[PlanStep], str, bool],  # (reasoning, steps, risk_assessment, escalation)
]


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
    ) -> None:
        """Initialize Policy Agent."""
        self._blackboard = blackboard
        self._capability_registry = capability_registry
        self._policy_engine = policy_engine

        self._loop_running: bool = False
        self._task: asyncio.Task[Any] | None = None

        self._processed_anomalies: deque[str] = deque(maxlen=self.PROCESSED_CACHE_SIZE)

        # Metrics for observability
        self._policies_generated = 0
        self._anomalies_processed = 0
        self._escalations_triggered = 0

        # Policy Dispatch Table: Maps ANOMALY KEY to a specific handler function.
        self._policy_dispatcher: dict[str, PolicyFunc] = {
            "overheat_warning": cast(PolicyFunc, self._policy_for_overheat),
        }

        logger.info(f"Policy Agent initialized. LLM Engine: {self._policy_engine.model_name()}")

    async def _post_trace_entry(
        self, source: str, event_type: str, reasoning_text: str, metadata: dict[str, Any]
    ) -> None:
        """Helper method to create and post a ReasoningTraceEntry."""
        entry = ReasoningTraceEntry(
            timestamp=datetime.now(),
            source=source,
            event_type=event_type,
            reasoning_text=reasoning_text,
            metadata=metadata,
        )
        await self._blackboard.add_trace_entry(entry)

    async def _fetch_context(self) -> StateEstimate | None:
        """Fetch latest StateEstimate for decision context."""
        return await self._blackboard.get_latest_state_estimate()

    def _validate_action(self, action: AgentToolCall) -> bool:
        """
        Validate that the action's capability (function_name) exists in the
        CapabilityRegistry by attempting to retrieve its function schema.
        """
        try:
            self._capability_registry.get_function_schema(action.action_name)
            return True
        except KeyError:
            logger.error(
                f"Action validation failed: Capability '{action.action_name}' "
                f"is not registered in the system's CapabilityRegistry."
            )
            return False

    def _get_llm_tool_catalog(self) -> str:
        """
        Fetches the complete list of physical tools and their capability schemas
        from the CapabilityRegistry, formatted as a JSON string for the LLM prompt.
        """
        try:
            # Uses the correct method 'get_llm_tool_catalog'
            return self._capability_registry.get_llm_tool_catalog()
        except Exception as e:
            logger.error(f"Failed to fetch tool catalog from CapabilityRegistry: {e}")
            # Return an empty list JSON structure as a safe fallback, ensuring type is 'str'
            return "[]"

    def _generate_critical_safety_policy(
        self, anomaly: AnomalyEvent, context: StateEstimate
    ) -> tuple[str, list[PlanStep], str, bool]:
        """
        Generate policy for HIGH severity anomalies (emergency shutdown).
        These policies are rules-based, non-negotiable, and always preempt.
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

        risk_assessment = "LOW - Emergency Halt Only"
        escalation_required = True

        return reasoning, policy_steps, risk_assessment, escalation_required

    def _policy_for_overheat(
        self, anomaly: AnomalyEvent, state_estimate: StateEstimate
    ) -> tuple[str, list[PlanStep], str, bool]:
        """Specific handler for the 'overheat_warning' anomaly (Rules-Based)."""
        reasoning = (
            f"Medium severity overheat detected "
            f"(Temp: {anomaly.metadata.get('temp', 'N/A')} C). "
            f"Since the intent was '{state_estimate.source_intent}', we will pause and "
            f"initiate a cooldown cycle."
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

        risk_assessment = "MEDIUM - Requires Active Management"
        escalation = False

        return reasoning, policy_steps, risk_assessment, escalation

    async def _generate_unknown_medium_policy(
        self, anomaly: AnomalyEvent, context: StateEstimate
    ) -> RemediationPolicy:
        """
        Handler for unknown MEDIUM severity anomalies, delegating plan generation
        entirely to the injected policy engine (LLM).
        """
        logger.info(
            f"Delegating unknown MEDIUM policy generation for anomaly {anomaly.key} "
            f"to LLM Engine: {self._policy_engine.model_name()}"
        )

        # 1. Fetch the authoritative tool catalog from the CapabilityRegistry
        tool_catalog_json = self._get_llm_tool_catalog()

        # 2. Pass the catalog along with the anomaly and state to the Policy Engine
        policy = await self._policy_engine.generate_policy(
            event=anomaly,
            context=context,
            tool_catalog_json=tool_catalog_json,
        )

        return policy

    async def _dispatch_policy_generation(
        self, anomaly: AnomalyEvent, context: StateEstimate
    ) -> RemediationPolicy | tuple[str, list[PlanStep], str, bool]:
        """
        Dispatches the anomaly to the correct policy generation function.

        Returns:
            Either a complete RemediationPolicy (if generated by LLM) or
            a tuple for a rules-based policy.
        """
        # 1. Highest Priority: Critical Safety Shutdown (Rules-Based, Non-Negotiable)
        if anomaly.severity == AnomalySeverity.HIGH:
            return self._generate_critical_safety_policy(anomaly, context)

        # 2. Medium Priority: Specific Known Remediation (Rules-Based)
        handler = self._policy_dispatcher.get(anomaly.key)

        if handler:
            return handler(self, anomaly, context)

        # 3. Default Medium: Unknown Anomaly (LLM-Assisted Plan)
        if anomaly.severity == AnomalySeverity.MEDIUM:
            # For LLM-assisted plans, we return the fully constructed RemediationPolicy object
            return await self._generate_unknown_medium_policy(anomaly, context)

        # 4. Low Priority: Ignored
        return "", [], "LOW - Ignored", False  # Returns the tuple format for ignored policies

    async def _handle_anomaly_event(self, anomaly: AnomalyEvent) -> None:
        """
        Generate remediation policy for an anomaly event using the dispatcher.
        """
        if anomaly.id in self._processed_anomalies:
            return

        self._processed_anomalies.append(anomaly.id)
        self._anomalies_processed += 1

        context = await self._fetch_context()

        if not context:
            logger.warning(
                f"No StateEstimate context available for anomaly {anomaly.key}. Skipping policy generation."
            )
            return

        logger.info(
            f"Policy Agent triggered by anomaly '{anomaly.key}' "
            f"(Severity: {anomaly.severity.name}). Current Intent: {context.source_intent}"
        )

        policy_result = await self._dispatch_policy_generation(anomaly, context)

        policy_source = "Rules"

        # Handle rules-based policy generation (returns a tuple)
        if isinstance(policy_result, tuple):
            reasoning, policy_steps, risk_assessment, escalation = policy_result

            if not policy_steps:
                return  # Low priority/Ignored

            # Validate actions before creating policy (Safety Gate)
            for step in policy_steps:
                # Actions are now AgentToolCall objects
                if step.action and not self._validate_action(step.action):
                    tool_id = step.action.arguments.get("tool_id", "N/A")
                    capability = step.action.action_name
                    logger.error(
                        f"Action validation failed in policy generation for: {tool_id}.{capability}"
                    )
                    escalation = True
                    validation_successful = False

            # Create policy from rules-based components
            policy = RemediationPolicy(
                trigger_event=anomaly,
                reasoning_trace=reasoning,
                risk_assessment=risk_assessment,
                corrective_steps=policy_steps,
                escalation_required=escalation,
            )

        # Handle LLM-based policy generation (returns a RemediationPolicy object)
        elif isinstance(policy_result, RemediationPolicy):
            policy = policy_result
            policy_steps = policy.corrective_steps
            escalation = policy.escalation_required
            policy_source = "LLM"  # Set source for trace logging

            # Re-validate LLM-generated actions as a final safety measure
            validation_successful = True
            for step in policy_steps:
                # LLM-generated steps should also use AgentToolCall
                if isinstance(step.action, AgentToolCall) and not self._validate_action(
                    step.action
                ):
                    tool_id = step.action.arguments.get("tool_id", "N/A")
                    capability = step.action.action_name
                    logger.error(f"LLM-generated Action validation failed: {tool_id}.{capability}")
                    # If LLM produces invalid action, we must escalate
                    policy.escalation_required = True
                    validation_successful = False

            # If validation failed, log and update reasoning/risk on the object
            if not validation_successful:
                policy.reasoning_trace += (
                    " WARNING: Generated plan contained invalid actions; required full review."
                )
                policy.risk_assessment = "CRITICAL - Invalid LLM Action"

        else:
            logger.error(f"Policy generation returned unexpected type: {type(policy_result)}")
            return

        # Push policy to orchestrator via blackboard
        await self._blackboard.set_remediation_policy(policy)
        self._policies_generated += 1

        if escalation:
            self._escalations_triggered += 1

        await self._post_trace_entry(
            source="PolicyAgent",
            event_type="POLICY_GENERATED",
            reasoning_text=f"Remediation Policy {policy.policy_id} generated by {policy_source}. Escalation: {policy.escalation_required}. Steps: {len(policy.corrective_steps)}.",
            metadata={
                "policy_id": policy.policy_id,
                "anomaly_id": anomaly.id,
                "source": policy_source,
                "risk_assessment": policy.risk_assessment,
                "escalation_required": policy.escalation_required,
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
