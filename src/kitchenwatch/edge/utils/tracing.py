import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from opentelemetry import trace

from kitchenwatch.edge.models.anomaly_event import AnomalySeverity
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.reasoning_trace_entry import ReasoningTraceEntry, TraceSeverity

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("cortexguard.trace")


def _source_to_name(source: object | str) -> str:
    """Normalize source to a short, stable name for traces."""
    if isinstance(source, str):
        return source
    # Prefer the class name; fall back to type repr if unusual
    try:
        return source.__class__.__name__
    except Exception:
        return type(source).__name__


@runtime_checkable
class BaseTraceSink(Protocol):
    """Protocol describing the minimal tracing surface used by components."""

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
    ) -> None: ...

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
    ) -> None: ...


@dataclass
class TraceSink:
    """Concrete TraceSink backed by a Blackboard."""

    blackboard: Blackboard

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
        await post_trace_entry(
            self,
            source,
            event_type,
            reasoning_text,
            metadata,
            severity=severity,
            refs=refs,
            duration_ms=duration_ms,
        )

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
        nonblocking_post_trace_entry(
            self,
            source,
            event_type,
            reasoning_text,
            metadata,
            severity=severity,
            refs=refs,
            duration_ms=duration_ms,
        )


async def _safe_add_trace(blackboard: Blackboard, entry: ReasoningTraceEntry) -> None:
    """Best-effort add_trace_entry; never raise to callers."""
    try:
        await blackboard.add_trace_entry(entry)
    except Exception:
        logger.debug("Failed to post trace entry", exc_info=True)


def map_anomaly_to_trace_severity(anom_sev: AnomalySeverity) -> TraceSeverity:
    """Deterministic mapping from AnomalySeverity -> TraceSeverity."""
    if anom_sev == AnomalySeverity.HIGH:
        return TraceSeverity.CRITICAL
    if anom_sev == AnomalySeverity.MEDIUM:
        return TraceSeverity.HIGH
    return TraceSeverity.WARN


async def post_trace_entry(
    sink: BaseTraceSink,
    source: object | str,
    event_type: str,
    reasoning_text: str,
    metadata: dict[str, Any] | None = None,
    *,
    severity: TraceSeverity = TraceSeverity.INFO,
    refs: dict[str, str] | None = None,
    duration_ms: int | None = None,
) -> None:
    """
    Post a single ReasoningTraceEntry to the configured sink.

    This function is awaitable and best-effort: failures are swallowed and logged.
    """
    source_name = _source_to_name(source)
    entry = ReasoningTraceEntry(
        timestamp=datetime.now(UTC),
        source=source_name,
        event_type=event_type,
        reasoning_text=reasoning_text,
        metadata=metadata or {},
        refs=refs or {},
        duration_ms=duration_ms,
        severity=severity,
    )

    # Try to call the concrete sink implementation if it exposes a blackboard,
    # otherwise fall back to expecting the sink to implement post_trace itself.
    if isinstance(sink, TraceSink) and getattr(sink, "blackboard", None) is not None:
        await _safe_add_trace(sink.blackboard, entry)

        try:
            span = trace.get_current_span()
            if span is not None:
                span.add_event(
                    event_type,
                    {
                        "source": source_name,
                        "reasoning_text": reasoning_text,
                        **(metadata or {}),
                        "severity": severity.name,
                        "duration_ms": duration_ms if duration_ms is not None else 0,
                    },
                )
        except Exception:
            logger.debug("Failed to emit OTel span event", exc_info=True)
    else:
        # If a custom sink is provided (test double, remote exporter), try to call its post_trace.
        try:
            await sink.post_trace_entry(source, event_type, reasoning_text, metadata, severity=severity, refs=refs, duration_ms=duration_ms)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Failed to post trace via custom sink", exc_info=True)


def nonblocking_post_trace_entry(
    sink: BaseTraceSink,
    source: object | str,
    event_type: str,
    reasoning_text: str,
    metadata: dict[str, Any] | None = None,
    *,
    severity: TraceSeverity = TraceSeverity.INFO,
    refs: dict[str, str] | None = None,
    duration_ms: int | None = None,
) -> None:
    """
    Schedule a trace post without awaiting. Safe to call from cancellation handlers.
    """
    try:
        asyncio.create_task(
            post_trace_entry(
                sink,
                source,
                event_type,
                reasoning_text,
                metadata,
                severity=severity,
                refs=refs,
                duration_ms=duration_ms,
            )
        )
    except Exception:
        logger.debug("Failed to schedule nonblocking trace", exc_info=True)


@asynccontextmanager
async def timed_trace(
    sink: BaseTraceSink,
    source: str,
    event_type: str,
    reasoning_text: str,
    metadata: dict[str, Any] | None = None,
    *,
    severity: TraceSeverity = TraceSeverity.INFO,
    refs: dict[str, str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Async context manager that measures duration and posts a completed trace on exit.

    Usage:
        async with timed_trace(sink, "AnomalyDetector", "TICK", "tick work") as ctx:
            # do work
        # on exit a TICK_COMPLETED trace is posted with duration_ms
    """
    start = datetime.now(UTC)
    try:
        yield {"start": start}
    finally:
        duration_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
        # Use nonblocking_post_trace to avoid blocking in finally blocks if desired;
        # here we await to ensure the duration is captured, but callers can choose nonblocking_post_trace.
        try:
            asyncio.get_running_loop()  # ensure we're in an event loop
            # prefer awaiting the post so downstream consumers see the completed trace promptly
            asyncio.create_task(
                post_trace_entry(
                    sink,
                    source,
                    f"{event_type}_COMPLETED",
                    reasoning_text,
                    metadata or {},
                    severity=severity,
                    refs=refs,
                    duration_ms=duration_ms,
                )
            )
        except RuntimeError:
            # No running loop — best-effort synchronous fallback
            try:
                nonblocking_post_trace_entry(
                    sink,
                    source,
                    f"{event_type}_COMPLETED",
                    reasoning_text,
                    metadata or {},
                    severity=severity,
                    refs=refs,
                    duration_ms=duration_ms,
                )
            except Exception:
                logger.debug("Failed to post timed trace", exc_info=True)
