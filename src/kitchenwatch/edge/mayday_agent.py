from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, cast

from kitchenwatch.core.interfaces.base_cloud_agent_client import BaseCloudAgentClient
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.mayday_packet import MaydayPacket, SystemHealth
from kitchenwatch.edge.models.plan import Plan
from kitchenwatch.edge.models.reasoning_trace_entry import TraceSeverity
from kitchenwatch.edge.models.remediation_policy import RemediationPolicy

# Import the tracing protocol (BaseTraceSink) and TraceSink concrete class
from kitchenwatch.edge.utils.tracing import BaseTraceSink  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


class _NoopTraceSink(BaseTraceSink):
    """Minimal no-op trace sink used when no real sink is provided (tests / headless)."""

    async def post_trace_entry(
        self,
        source: object | str,
        event_type: str,
        reasoning_text: str,
        metadata: dict[str, Any] | None = None,
        *,
        severity: TraceSeverity = TraceSeverity.INFO,
        refs: dict[str, str] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        return None

    def nonblocking_post_trace_entry(
        self,
        source: object | str,
        event_type: str,
        reasoning_text: str,
        metadata: dict[str, Any] | None = None,
        *,
        severity: TraceSeverity = TraceSeverity.INFO,
        refs: dict[str, str] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        return None


class MaydayAgent:
    """
    Edge-side escalation agent. Builds MaydayPacket from Blackboard/context
    and delegates transport to an injected BaseCloudAgentClient implementation.

    Adds bounded timeouts and simple retry/backoff to make cloud calls resilient.
    Emits structured trace entries for observability and auditing.
    """

    def __init__(
        self,
        cloud_agent_client: BaseCloudAgentClient,
        device_id: str,
        *,
        timeout_seconds: float = 5.0,
        max_retries: int = 1,
        backoff_factor: float = 2.0,
        trace_sink: BaseTraceSink | None = None,
    ) -> None:
        self._cloud_agent_client = cloud_agent_client
        self._device_id = device_id

        # Resilience configuration
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._backoff_factor = float(backoff_factor)

        # Metrics
        self._escalations_sent = 0  # number of send_escalation() calls
        self._attempts_sent = 0  # number of actual transport attempts (includes retries)
        self._responses_received = 0

        # Tracing: prefer provided sink; if none, use a no-op sink to avoid constructing TraceSink without Blackboard
        self._trace_sink: BaseTraceSink = trace_sink if trace_sink is not None else _NoopTraceSink()

    def _serialize_state_estimate(self, state: Any) -> dict[str, object] | None:
        """
        Return a dict[str, object] representation of a StateEstimate or None.
        Uses model_dump if available, falls back to to_dict or __dict__.
        Casts to the narrower dict[str, object] type for mypy.
        """
        if state is None:
            return None

        # Prefer pydantic-style model_dump
        dump_fn = getattr(state, "model_dump", None)
        if callable(dump_fn):
            try:
                raw = dump_fn()
                return cast(dict[str, object], raw)
            except Exception as exc:
                logger.debug("model_dump failed in _serialize_state_estimate", exc_info=exc)

        # Fallback to to_dict
        to_dict = getattr(state, "to_dict", None)
        if callable(to_dict):
            try:
                raw = to_dict()
                return cast(dict[str, object], raw)
            except Exception as exc:
                logger.debug("to_dict failed in _serialize_state_estimate", exc_info=exc)

        # Fallback to __dict__
        if hasattr(state, "__dict__"):
            try:
                return cast(dict[str, object], dict(state.__dict__))
            except Exception as exc:
                logger.debug("__dict__ access failed in _serialize_state_estimate", exc_info=exc)

        return None

    async def build_packet_from_policy(
        self,
        policy: RemediationPolicy,
        blackboard: Blackboard,
        health: SystemHealth,
        trace_id: str | None = None,
    ) -> MaydayPacket:
        """
        Compose a compact MaydayPacket using Blackboard helpers and the given policy.
        Uses model_dump / compact serializers where available on models.
        Emits PACKET_BUILT and PACKET_REDACTED traces.
        """
        start = datetime.now()

        # Gather compact context from blackboard
        state = await blackboard.get_latest_state_estimate()
        sg = await blackboard.get_scene_graph()
        anomalies = await blackboard.get_active_anomalies()
        current_plan = await blackboard.get_current_plan()
        current_step = await blackboard.get_current_step()

        # recent actions and traces: use existing blackboard methods if present
        # fall back to empty lists if not implemented
        try:
            recent_actions = await blackboard.get("recent_actions", [])
        except Exception:
            recent_actions = []

        try:
            reasoning_trace = [
                e.model_dump() for e in (await blackboard.get("reasoning_trace", []))
            ]
        except Exception:
            reasoning_trace = []

        # Use SceneGraph.to_compact_dict if available; fall back defensively
        scene_graph_compact = None
        scene_graph_count = 0
        if sg is not None:
            fn = getattr(sg, "to_compact_dict", None)
            if callable(fn):
                try:
                    scene_graph_compact = fn()
                    scene_graph_count = (
                        len(scene_graph_compact.get("objects", []))
                        if isinstance(scene_graph_compact, dict)
                        else 0
                    )
                except Exception:
                    scene_graph_compact = None

        # remediation_policy serialization: prefer model_dump, fall back to dict() or vars()
        try:
            if hasattr(policy, "model_dump") and callable(policy.model_dump):
                remediation_policy_serialized = policy.model_dump(exclude_none=True)
            elif hasattr(policy, "dict") and callable(policy.dict):
                remediation_policy_serialized = policy.dict()
            else:
                remediation_policy_serialized = dict(vars(policy))
        except Exception:
            remediation_policy_serialized = {"policy_id": getattr(policy, "policy_id", None)}

        # replay_data: prefer explicit attribute, otherwise check metadata for embedded replay
        replay_data = None
        trigger = getattr(policy, "trigger_event", None)
        if trigger is not None:
            # prefer explicit attribute if present and accessible
            if hasattr(trigger, "replay"):
                try:
                    replay_data = trigger.replay
                except Exception:
                    replay_data = None
            # otherwise look in metadata (safe for Pydantic models)
            if replay_data is None:
                try:
                    md = getattr(trigger, "metadata", {}) or {}
                    replay_data = md.get("replay")
                except Exception:
                    replay_data = None

        policy_id_val = getattr(policy, "policy_id", "")
        trace_id_val: str = trace_id if isinstance(trace_id, str) else str(policy_id_val or "")
        packet = MaydayPacket(
            trace_id=trace_id_val,
            device_id=self._device_id,
            timestamp=datetime.now(),
            anomalies=list(anomalies.values()) if anomalies else [],
            current_plan_id=getattr(current_plan, "plan_id", None),
            current_step=getattr(current_step, "id", None),
            last_actions=[a for a in recent_actions],
            health=health,
            state_estimate=self._serialize_state_estimate(state),
            scene_graph_compact=scene_graph_compact,
            reasoning_trace=reasoning_trace or [getattr(policy, "reasoning_trace", "")],
            remediation_policy=remediation_policy_serialized,
            replay_data=replay_data,
        )

        duration_ms = int((datetime.now() - start).total_seconds() * 1000)
        # PACKET_BUILT trace
        try:
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="PACKET_BUILT",
                reasoning_text="Mayday packet composed",
                metadata={
                    "trace_id": packet.trace_id,
                    "policy_id": getattr(policy, "policy_id", None),
                    "device_id": self._device_id,
                    "packet_size_bytes": (
                        len(packet.model_dump_json())
                        if hasattr(packet, "model_dump_json")
                        else None
                    ),
                    "scene_graph_objects_count": scene_graph_count,
                    "duration_ms": duration_ms,
                },
            )
        except Exception:
            logger.debug("Failed to post PACKET_BUILT trace", exc_info=True)

        # If redaction or truncation occurred, emit PACKET_REDACTED (best-effort)
        try:
            redacted = False
            removed_fields: list[str] = []
            if sg is not None and scene_graph_compact is None:
                redacted = True
                removed_fields.append("scene_graph")
            if len(reasoning_trace) > 50:
                redacted = True
                removed_fields.append("reasoning_trace_truncated")
            if redacted:
                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="PACKET_REDACTED",
                    reasoning_text="Mayday packet redaction applied",
                    metadata={
                        "trace_id": packet.trace_id,
                        "policy_id": getattr(policy, "policy_id", None),
                        "removed_fields": removed_fields,
                    },
                    severity=TraceSeverity.WARN,
                )
        except Exception:
            logger.debug("Failed to post PACKET_REDACTED trace", exc_info=True)

        return packet

    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        """
        Send the MaydayPacket to the cloud via the injected BaseCloudAgentClient.

        Behavior:
        - Increment _escalations_sent once per logical call.
        - Attempt up to (max_retries + 1) transport attempts.
        - Each attempt is bounded by timeout_seconds via asyncio.wait_for.
        - On transient failure (timeout or exception) retry with exponential backoff.
        - Return Plan on success, or None on exhaustion/fatal error.
        Emits ESCALATION_ATTEMPT, ESCALATION_SUCCESS, ESCALATION_NO_PLAN, ESCALATION_ATTEMPT_TIMEOUT,
        ESCALATION_ATTEMPT_ERROR, and ESCALATION_FAILURE traces.
        """
        logger.info("MaydayAgent: sending escalation %s", packet.trace_id)
        self._escalations_sent += 1

        last_exc: Exception | None = None
        attempt = 0
        while attempt <= self._max_retries:
            attempt += 1
            self._attempts_sent += 1

            # Trace attempt start
            try:
                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="ESCALATION_ATTEMPT",
                    reasoning_text=f"Escalation attempt {attempt} started",
                    metadata={
                        "trace_id": packet.trace_id,
                        "attempt_index": attempt,
                        "timeout_s": self._timeout_seconds,
                        "backoff_factor": self._backoff_factor,
                    },
                )
            except Exception:
                logger.debug("Failed to post ESCALATION_ATTEMPT trace", exc_info=True)

            attempt_start = datetime.now()
            try:
                coro = self._cloud_agent_client.send_escalation(packet)
                plan = await asyncio.wait_for(coro, timeout=self._timeout_seconds)
                duration_ms = int((datetime.now() - attempt_start).total_seconds() * 1000)

                if plan:
                    self._responses_received += 1
                    logger.info(
                        "MaydayAgent: received plan %s", getattr(plan, "plan_id", "<no-id>")
                    )
                    # Success trace
                    try:
                        await self._trace_sink.post_trace_entry(
                            source=self,
                            event_type="ESCALATION_SUCCESS",
                            reasoning_text="Cloud returned remediation plan",
                            metadata={
                                "trace_id": packet.trace_id,
                                "plan_id": getattr(plan, "plan_id", None),
                                "duration_ms": duration_ms,
                                "attempt_index": attempt,
                            },
                        )
                    except Exception:
                        logger.debug("Failed to post ESCALATION_SUCCESS trace", exc_info=True)
                else:
                    logger.warning("MaydayAgent: cloud returned no plan for %s", packet.trace_id)
                    try:
                        await self._trace_sink.post_trace_entry(
                            source=self,
                            event_type="ESCALATION_NO_PLAN",
                            reasoning_text="Cloud returned no plan",
                            metadata={
                                "trace_id": packet.trace_id,
                                "duration_ms": duration_ms,
                                "attempt_index": attempt,
                            },
                            severity=TraceSeverity.WARN,
                        )
                    except Exception:
                        logger.debug("Failed to post ESCALATION_NO_PLAN trace", exc_info=True)

                return plan

            except TimeoutError as te:
                last_exc = te
                logger.warning(
                    "MaydayAgent: escalation attempt %d timed out after %.2fs for %s",
                    attempt,
                    self._timeout_seconds,
                    packet.trace_id,
                )
                try:
                    await self._trace_sink.post_trace_entry(
                        source=self,
                        event_type="ESCALATION_ATTEMPT_TIMEOUT",
                        reasoning_text="Escalation attempt timed out",
                        metadata={
                            "trace_id": packet.trace_id,
                            "attempt_index": attempt,
                            "timeout_s": self._timeout_seconds,
                        },
                        severity=TraceSeverity.WARN,
                    )
                except Exception:
                    logger.debug("Failed to post ESCALATION_ATTEMPT_TIMEOUT trace", exc_info=True)

            except Exception as exc:
                last_exc = exc
                logger.exception(
                    "MaydayAgent: escalation attempt %d failed for %s: %s",
                    attempt,
                    packet.trace_id,
                    exc,
                )
                try:
                    await self._trace_sink.post_trace_entry(
                        source=self,
                        event_type="ESCALATION_ATTEMPT_ERROR",
                        reasoning_text="Escalation attempt failed",
                        metadata={
                            "trace_id": packet.trace_id,
                            "attempt_index": attempt,
                            "error": str(exc),
                        },
                        severity=TraceSeverity.HIGH,
                    )
                except Exception:
                    logger.debug("Failed to post ESCALATION_ATTEMPT_ERROR trace", exc_info=True)

            # If we have more retries left, backoff then retry
            if attempt <= self._max_retries:
                backoff = (self._backoff_factor ** (attempt - 1)) * 0.1
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    logger.info("MaydayAgent: retry sleep cancelled")
                    break

        # Exhausted retries: emit final failure trace and return None
        try:
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="ESCALATION_FAILURE",
                reasoning_text="Exhausted retries for escalation",
                metadata={
                    "trace_id": packet.trace_id,
                    "attempts": self._attempts_sent,
                    "last_error": repr(last_exc),
                },
                severity=TraceSeverity.CRITICAL,
            )
        except Exception:
            logger.debug("Failed to post ESCALATION_FAILURE trace", exc_info=True)

        return None

    def get_metrics(self) -> dict[str, int]:
        return {
            "escalations_sent": self._escalations_sent,
            "attempts_sent": self._attempts_sent,
            "responses_received": self._responses_received,
        }
