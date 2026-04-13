from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from opentelemetry import trace

from cortexguard.core.interfaces.base_controller import BaseController
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.capability_registry import CapabilityRegistry
from cortexguard.edge.models.reasoning_trace_entry import ReasoningTraceEntry, TraceSeverity
from cortexguard.edge.utils.metrics import emergency_stop_active

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("cortexguard.arbiter")


class Arbiter:
    """
    Central authority for action authorization, preemption, and audit.

    Responsibilities:
    - Authorize or deny action requests (request_action).
    - Perform emergency stop (emergency_stop).
    - Record compact ReasoningTraceEntry objects for every decision.
    - Publish a short safety flag to the Blackboard for subscribers.
    """

    def __init__(
        self,
        blackboard: Blackboard,
        capability_registry: CapabilityRegistry,
        controller: BaseController,
        audit_capacity: int = 1000,
    ) -> None:
        self._blackboard = blackboard
        self._capability_registry = capability_registry
        self._controller = controller

        self._audit: deque[ReasoningTraceEntry] = deque(maxlen=audit_capacity)
        self._lock = asyncio.Lock()

    async def request_action(
        self, _caller_id: str, action: AgentToolCall, reason: str | None = None
    ) -> bool:
        """
        Authorize and (best-effort) execute an action.

        Returns True if the action was authorized and execution was attempted successfully.
        Always records a ReasoningTraceEntry and publishes a compact safety flag to the Blackboard.

        The lock is held only during fast validation and audit append — never during
        controller.execute(), so emergency_stop() can always acquire it promptly.
        """
        trace_id = f"trace-{uuid4().hex[:8]}"
        ts = datetime.now(UTC)

        event_type = "ACTION_REQUEST"
        metadata: dict[str, Any] = {
            "action_name": action.action_name,
            "arguments": action.arguments,
        }
        refs: dict[str, str] = {}

        # ------------------------------------------------------------------
        # Phase 1: Validation (under lock — fast, no I/O)
        # Returns the authorized reasoning text and risk, or records a deny
        # entry and returns False immediately.
        # ------------------------------------------------------------------
        authorized_reasoning: str | None = None

        async with self._lock:
            try:
                # 1) Capability existence
                try:
                    _ = self._capability_registry.get_function_schema(action.action_name)
                except KeyError:
                    entry = ReasoningTraceEntry(
                        id=trace_id,
                        timestamp=ts,
                        source="arbiter",
                        event_type=event_type,
                        reasoning_text="Unknown capability",
                        metadata=metadata,
                        refs=refs,
                        duration_ms=None,
                        severity=TraceSeverity.HIGH,
                    )
                    self._append_and_publish(entry)
                    return False

                # 2) Validate call against schema / risk
                try:
                    validate_result = self._capability_registry.validate_call(
                        action.action_name, action.arguments or {}
                    )
                except Exception as e:
                    entry = ReasoningTraceEntry(
                        id=trace_id,
                        timestamp=ts,
                        source="arbiter",
                        event_type=event_type,
                        reasoning_text=f"Validation error: {e}",
                        metadata=metadata,
                        refs=refs,
                        duration_ms=None,
                        severity=TraceSeverity.HIGH,
                    )
                    self._append_and_publish(entry)
                    return False

                if validate_result is None:
                    entry = ReasoningTraceEntry(
                        id=trace_id,
                        timestamp=ts,
                        source="arbiter",
                        event_type=event_type,
                        reasoning_text="Validation API returned no result; denying by default",
                        metadata=metadata,
                        refs=refs,
                        duration_ms=None,
                        severity=TraceSeverity.HIGH,
                    )
                    self._append_and_publish(entry)
                    return False

                valid, risk = validate_result

                if not valid:
                    entry = ReasoningTraceEntry(
                        id=trace_id,
                        timestamp=ts,
                        source="arbiter",
                        event_type=event_type,
                        reasoning_text=f"Validation failed (risk={risk})",
                        metadata=metadata,
                        refs=refs,
                        duration_ms=None,
                        severity=TraceSeverity.HIGH,
                    )
                    self._append_and_publish(entry)
                    return False

                # 3) Risk gating
                if getattr(risk, "name", str(risk)).upper() == "HIGH":
                    entry = ReasoningTraceEntry(
                        id=trace_id,
                        timestamp=ts,
                        source="arbiter",
                        event_type=event_type,
                        reasoning_text="High risk action; requires escalation",
                        metadata=metadata,
                        refs=refs,
                        duration_ms=None,
                        severity=TraceSeverity.HIGH,
                    )
                    self._append_and_publish(entry)
                    return False

                # Validation passed — record authorization decision for Phase 2
                authorized_reasoning = f"Authorized (risk={getattr(risk, 'name', str(risk))})"

            except Exception as e:
                entry = ReasoningTraceEntry(
                    id=trace_id,
                    timestamp=ts,
                    source="arbiter",
                    event_type="ACTION_REQUEST_ERROR",
                    reasoning_text=f"Arbiter internal error: {e}",
                    metadata=metadata,
                    refs=refs,
                    duration_ms=None,
                    severity=TraceSeverity.CRITICAL,
                )
                self._append_and_publish(entry)
                return False

        # ------------------------------------------------------------------
        # Phase 2: Execute controller (lock released — may be slow/blocking)
        # ------------------------------------------------------------------
        if authorized_reasoning is None:  # pragma: no cover — invariant: always set above
            return False
        exec_start = datetime.now(UTC)
        exec_entry: ReasoningTraceEntry
        succeeded = False
        try:
            if hasattr(self._controller, "execute"):
                await self._controller.execute(action.action_name, action.arguments or {})
            elif hasattr(self._controller, "execute_action"):
                await self._controller.execute_action(action)
            else:
                raise RuntimeError("Controller has no execute method")
            exec_duration = int((datetime.now(UTC) - exec_start).total_seconds() * 1000)
            exec_entry = ReasoningTraceEntry(
                id=trace_id,
                timestamp=ts,
                source="arbiter",
                event_type=event_type,
                reasoning_text=authorized_reasoning,
                metadata=metadata,
                refs=refs,
                duration_ms=exec_duration,
                severity=TraceSeverity.INFO,
            )
            succeeded = True
        except Exception as exec_err:
            exec_entry = ReasoningTraceEntry(
                id=trace_id,
                timestamp=ts,
                source="arbiter",
                event_type=event_type,
                reasoning_text=f"Execution failed: {exec_err}",
                metadata=metadata,
                refs=refs,
                duration_ms=None,
                severity=TraceSeverity.HIGH,
            )

        # ------------------------------------------------------------------
        # Phase 3: Audit append (under lock — fast)
        # ------------------------------------------------------------------
        async with self._lock:
            self._append_and_publish(exec_entry)

        if succeeded:
            try:
                await self._blackboard.set_safety_flag("last_action_authorized", True)
            except Exception as exc:
                logger.debug("Failed to set last_action_authorized safety flag", exc_info=exc)
            try:
                asyncio.create_task(self._blackboard.add_trace_entry(exec_entry))
            except Exception as exc:
                logger.debug("Failed to schedule trace post (add_trace_entry)", exc_info=exc)

        return succeeded

    async def emergency_stop(self, reason: str, trace_id: str | None = None) -> None:
        """
        Force immediate stop. Records a CRITICAL trace, publishes a safety flag,
        and calls the controller's emergency stop (best-effort).
        """
        tid = trace_id or f"trace-{uuid4().hex[:8]}"
        ts = datetime.now(UTC)
        event_type = "EMERGENCY_STOP"
        reasoning_text = f"Emergency stop requested: {reason}"
        metadata = {"reason": reason}
        entry = ReasoningTraceEntry(
            id=tid,
            timestamp=ts,
            source="arbiter",
            event_type=event_type,
            reasoning_text=reasoning_text,
            metadata=metadata,
            refs={},
            duration_ms=None,
            severity=TraceSeverity.CRITICAL,
        )

        with tracer.start_as_current_span("arbiter.emergency_stop") as span:
            span.set_attribute("reason", reason)
            span.set_attribute("trace_id", tid)
            async with self._lock:
                # record and publish
                self._append_and_publish(entry)

                emergency_stop_active.set(1)
                try:
                    await self._blackboard.set_safety_flag("emergency_stop", True)
                except Exception as exc:
                    logger.debug("Failed to set emergency_stop safety flag", exc_info=exc)

                # call controller emergency stop if available
                try:
                    if hasattr(self._controller, "emergency_stop"):
                        await self._controller.emergency_stop()
                    else:
                        # fallback to executing a named primitive
                        if hasattr(self._controller, "execute"):
                            await self._controller.execute("EMERGENCY_STOP", {})
                except Exception as exc:
                    span.record_exception(exc)
                    # controller failure should not raise to callers
                    logger.exception(
                        "Controller emergency_stop call failed; continuing (best-effort)",
                        exc_info=exc,
                    )

    def _append_and_publish(self, entry: ReasoningTraceEntry) -> None:
        """Append to in-memory audit and attempt to publish to blackboard (called under lock)."""
        # store in-memory
        self._audit.append(entry)

        # Best-effort publish the full trace to the Blackboard's add_trace_entry API.
        try:
            # schedule async call so we don't await inside the lock
            asyncio.create_task(self._blackboard.add_trace_entry(entry))
        except Exception as exc:
            # swallow publish errors; audit remains in-memory
            logger.debug("Failed to schedule trace post (add_trace_entry)", exc_info=exc)

        # Lightweight toggle so subscribers can detect recent arbiter activity (best-effort).
        # Note: emergency_stop flag is set explicitly in emergency_stop() via direct await —
        # not here, to avoid unintended side-effects from internal errors.
        try:
            asyncio.create_task(self._blackboard.set_safety_flag("last_arbiter_event", True))
        except Exception as exc:
            logger.debug("Failed to schedule last_arbiter_event safety flag", exc_info=exc)

    async def get_latest_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent audit entries as dicts."""
        async with self._lock:
            items = list(self._audit)[-limit:]
            return [e.model_dump() for e in items]

    async def clear_audit(self) -> None:
        """Clear the in-memory audit buffer."""
        async with self._lock:
            self._audit.clear()
