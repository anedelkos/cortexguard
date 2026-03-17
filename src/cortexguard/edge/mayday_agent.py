from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any, cast

from opentelemetry import trace

from cortexguard.core.interfaces.base_cloud_agent_client import BaseCloudAgentClient
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth
from cortexguard.edge.models.plan import Plan
from cortexguard.edge.models.reasoning_trace_entry import TraceSeverity
from cortexguard.edge.models.remediation_policy import RemediationPolicy
from cortexguard.edge.utils.metrics import (
    component_duration_ms,
    policy_escalations_total,
)

# Import the tracing protocol (BaseTraceSink) and TraceSink concrete class
from cortexguard.edge.utils.tracing import BaseTraceSink  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("cortexguard.mayday")


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
        self._consecutive_failures = 0
        self._timeouts_total = 0
        self._errors_total = 0
        self._retries_total = 0
        self._failures_total = 0
        self._health_state = 0  # 0=ok, 1=degraded
        self._health_degrade_threshold = 3

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
        start = datetime.now(UTC)

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
            timestamp=datetime.now(UTC),
            anomalies=list(anomalies.values()) if anomalies else [],
            current_plan_id=getattr(current_plan, "plan_id", None),
            current_step=getattr(current_step, "id", None),
            last_actions=[a for a in recent_actions],
            health=health,
            state_estimate=self._serialize_state_estimate(state),
            scene_graph_compact=scene_graph_compact,
            reasoning_trace=reasoning_trace or [],
            remediation_policy=remediation_policy_serialized,
            replay_data=replay_data,
        )

        duration_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
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

    async def _attempt_escalation(
        self, packet: MaydayPacket, attempt: int
    ) -> tuple[Plan | None, Exception | None]:
        """
        One escalation attempt, including tracing, cloud call, and error handling.
        """
        attempt_start = time.perf_counter()

        with tracer.start_as_current_span("mayday.send_attempt") as attempt_span:
            attempt_span.set_attribute("attempt.index", attempt)
            attempt_span.set_attribute("timeout_seconds", self._timeout_seconds)
            attempt_span.set_attribute("backoff_factor", self._backoff_factor)

            self._attempts_sent += 1

            # Trace attempt start
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

            attempt_start = time.perf_counter()

            try:
                plan = await self._call_cloud(packet, attempt)
                duration_ms = (time.perf_counter() - attempt_start) * 1000.0
                component_duration_ms.labels(component="mayday_attempt").observe(duration_ms)

                if plan:
                    self._responses_received += 1
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
                else:
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

                self._consecutive_failures = 0
                self._health_state = 0
                return plan, None

            except asyncio.CancelledError:
                logger.info("MaydayAgent: escalation cancelled")
                raise

            except TimeoutError as te:
                self._timeouts_total += 1
                self._retries_total += 1
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._health_degrade_threshold:
                    self._health_state = 1

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
                return None, te

            except Exception as exc:
                self._errors_total += 1
                self._retries_total += 1
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._health_degrade_threshold:
                    self._health_state = 1

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
                return None, exc

    async def _call_cloud(self, packet: MaydayPacket, attempt: int) -> Plan | None:
        """
        Isolated cloud call wrapped in its own span.
        """
        with tracer.start_as_current_span("cloud.send_escalation") as cloud_span:
            cloud_span.set_attribute("packet.trace_id", packet.trace_id)
            cloud_span.set_attribute("attempt.index", attempt)

            coro = self._cloud_agent_client.send_escalation(packet)
            return await asyncio.wait_for(coro, timeout=self._timeout_seconds)

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
        policy_escalations_total.inc()
        start = time.perf_counter()

        with tracer.start_as_current_span("mayday.escalation") as span:
            span.set_attribute("packet.trace_id", packet.trace_id)
            span.set_attribute("device.id", self._device_id)
            span.set_attribute("max_retries", self._max_retries)
            span.set_attribute("timeout_seconds", self._timeout_seconds)

            logger.info("MaydayAgent: sending escalation %s", packet.trace_id)
            self._escalations_sent += 1

            last_exc: Exception | None = None

            for attempt in range(1, self._max_retries + 2):
                plan, last_exc = await self._attempt_escalation(packet, attempt)

                if plan is not None:
                    return plan

                # retry if attempts remain
                if attempt <= self._max_retries:
                    backoff = (self._backoff_factor ** (attempt - 1)) * 0.1
                    try:
                        await asyncio.sleep(backoff)
                    except asyncio.CancelledError:
                        logger.info("MaydayAgent: retry sleep cancelled")
                        return None

            # Exhausted retries
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

            self._failures_total += 1
            duration_ms = (time.perf_counter() - start) * 1000.0
            component_duration_ms.labels(component="mayday").observe(duration_ms)
            return None

    def get_metrics(self) -> dict[str, int]:
        return {
            "escalations_sent": self._escalations_sent,
            "attempts_sent": self._attempts_sent,
            "responses_received": self._responses_received,
            "retries_total": self._retries_total,
            "timeouts_total": self._timeouts_total,
            "errors_total": self._errors_total,
            "failures_total": self._failures_total,
            "health_state": self._health_state,
        }
