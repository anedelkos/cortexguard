"""Asynchronous orchestration of cloud planning requests and final results."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from opentelemetry import trace as _otel_trace

from cortexguard.cloud.graph.state import CloudPlanningState, ResumeInput
from cortexguard.cloud.graph.workflow import build_graph, run_planning_workflow
from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.cloud.persistence.repository import IncidentRepositoryProtocol
from cortexguard.cloud.planner.llm_client import LLMClientProtocol
from cortexguard.cloud.queue.sqs import SQSQueue
from cortexguard.cloud.retrieval.store import RetrievalStoreProtocol
from cortexguard.cloud.validation.plan_validator import PlanValidatorProtocol
from cortexguard.edge.models.mayday_packet import MaydayPacket
from cortexguard.edge.models.plan import Plan

_tracer = _otel_trace.get_tracer(__name__)

logger = logging.getLogger(__name__)


def _increment_cloud_metric(metric_name: str) -> None:
    try:
        from cortexguard.cloud import runtime as cloud_runtime
    except ImportError:
        logger.debug("Cloud runtime unavailable while incrementing metric %s", metric_name)
        return
    metric = getattr(cloud_runtime, metric_name, None)
    if metric is not None:
        metric.inc()


def _observe_cloud_metric_seconds(metric_name: str, duration_seconds: float) -> None:
    try:
        from cortexguard.cloud import runtime as cloud_runtime
    except ImportError:
        logger.debug("Cloud runtime unavailable while observing metric %s", metric_name)
        return
    metric = getattr(cloud_runtime, metric_name, None)
    if metric is not None:
        metric.observe(duration_seconds)


@dataclass
class PlanningResult:
    """Completed planning outcome, distinct from the ``"pending"`` sentinel and unknown IDs."""

    decision: str
    plan: Plan | None


class CloudOrchestrator:
    def __init__(
        self,
        repo: IncidentRepositoryProtocol,
        retrieval_store: RetrievalStoreProtocol | None,
        llm_client: LLMClientProtocol | None,
        validator: PlanValidatorProtocol | None = None,
        checkpoint_store: str | None = None,
        checkpoint_db_path: str | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        self._repo = repo
        self._graph = build_graph(
            repo,
            retrieval_store,
            llm_client,
            validator,
            checkpoint_store,
            checkpoint_db_path,
            checkpointer=checkpointer,
        )
        self._results: dict[str, PlanningResult | Literal["pending"]] = {}
        self._lock = asyncio.Lock()

    async def submit(self, packet: MaydayPacket) -> str:
        trace_id = packet.trace_id
        async with self._lock:
            self._results[trace_id] = "pending"
        _increment_cloud_metric("cloud_planning_requests_total")
        asyncio.create_task(self._run(trace_id, packet))
        return trace_id

    async def run_once(self, packet: MaydayPacket) -> bool:
        """Run the planning workflow and persist the result.

        Used by the SQS worker, the final decision is written to the incident store
        by :meth:`_finalise_incident`, making it visible to :class:`SQSCloudOrchestrator`.

        If the graph interrupts (needs_human), the checkpoint is saved and
        finalisation is skipped. Call :meth:`resume` with operator input to continue.

        Returns:
            ``True`` if the graph was interrupted (paused for operator), ``False`` otherwise.
        """
        _, interrupted = await self._execute(packet.trace_id, packet)
        return interrupted

    async def resume(self, thread_id: str, operator_response: dict[str, Any]) -> None:
        """Resume a paused graph with operator input.

        Args:
            thread_id: The incident_id that was used as the thread_id
                when ``run_once`` was called.
            operator_response: Dict with ``approved``, ``plan_override``,
                ``timeout``, or ``action`` keys.
        """
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        try:
            snap = self._graph.get_state(config)
        except Exception:
            logger.warning("resume: get_state failed for thread_id=%s", thread_id, exc_info=True)
            return
        if snap is None or not snap.next:
            logger.warning(
                "resume called for non-interrupted thread_id=%s (no pending tasks)",
                thread_id,
            )
            return
        state: ResumeInput = {
            "operator_response": operator_response,
        }
        final_state: CloudPlanningState = await self._graph.ainvoke(state, config)
        decision = final_state.get("decision") or "no_safe_plan"
        plan = final_state.get("candidate_plan")
        await self._finalise_incident(final_state, decision, plan)
        logger.info(
            "cloud_plan resume decision=%s incident_id=%s",
            decision,
            thread_id,
        )

    async def _run(self, trace_id: str, packet: MaydayPacket) -> None:
        final_state, interrupted = await self._execute(trace_id, packet)
        if final_state is not None:
            decision = final_state.get("decision") or "no_safe_plan"
            plan = final_state.get("candidate_plan")
            async with self._lock:
                self._results[trace_id] = PlanningResult(decision=decision, plan=plan)

    async def _execute(
        self, trace_id: str, packet: MaydayPacket
    ) -> tuple[CloudPlanningState | None, bool]:
        """Run the graph and detect interruption.

        Returns:
            (final_state or None, interrupted) where ``interrupted`` is
            ``True`` if the graph paused at ``route_decision`` for operator input.
        """
        final_state: CloudPlanningState | None = None
        decision: str = "no_safe_plan"
        plan: Plan | None = None
        confidence: float = 0.0
        rationale: str = ""
        t0 = time.monotonic()
        with _tracer.start_as_current_span("cloud_planning_workflow"):
            _span = _otel_trace.get_current_span()
            try:
                final_state = await run_planning_workflow(packet, self._graph, thread_id=trace_id)
                decision = final_state.get("decision") or "no_safe_plan"
                plan = final_state.get("candidate_plan")
                confidence = final_state.get("confidence") or 0.0
                rationale = final_state.get("rationale") or ""
                _span.set_attribute("cloud.decision", decision)
                _span.set_attribute("cloud.confidence", confidence)
                _span.set_attribute("cloud.step_count", len(plan.steps) if plan else 0)
                if rationale:
                    _span.set_attribute("cloud.rationale", rationale[:500])
            except Exception:
                logger.exception("Unhandled error in planning workflow trace_id=%s", trace_id)
                decision = "no_safe_plan"
                plan = None
        _observe_cloud_metric_seconds("cloud_planning_duration_seconds", time.monotonic() - t0)

        # Check if graph was interrupted (paused at route_decision for operator)
        interrupted = False
        try:
            config = {"configurable": {"thread_id": trace_id}}
            state_snapshot = self._graph.get_state(config)
            if state_snapshot is not None and state_snapshot.next:
                interrupted = True
        except Exception:
            logger.warning(
                "cloud_plan interrupted check failed trace_id=%s", trace_id, exc_info=True
            )

        if interrupted:
            logger.info(
                "cloud_plan interrupted (needs_human) trace_id=%s, awaiting operator",
                trace_id,
            )
        else:
            if final_state is not None:
                await self._finalise_incident(final_state, decision, plan)
            logger.info(
                "cloud_plan decision=%s confidence=%.2f steps=%d trace_id=%s | %s",
                decision,
                confidence,
                len(plan.steps) if plan else 0,
                trace_id,
                rationale[:200],
            )
        return final_state, interrupted

    async def _finalise_incident(
        self,
        final_state: object,
        decision: str,
        plan: Plan | None,
    ) -> None:
        """Write planning outcome fields back to the persisted incident record."""
        if not isinstance(final_state, dict):
            return
        incident_id = final_state.get("incident_id")
        if not isinstance(incident_id, str):
            return
        try:
            record = await self._repo.get_incident(incident_id)
            if record is None:
                return
            validation_result = final_state.get("validation_result")
            errors_json = (
                json.dumps(validation_result.errors) if validation_result is not None else "[]"
            )
            updated = _dc_replace(
                record,
                candidate_plan_json=plan.model_dump_json() if plan is not None else None,
                validation_errors_json=errors_json,
                decision=decision,
                rationale=final_state.get("rationale"),
                confidence=final_state.get("confidence"),
            )
            await self._repo.save_incident(updated)
        except Exception:
            logger.exception("Failed to finalise incident record incident_id=%s", incident_id)

    async def get_result(self, trace_id: str) -> PlanningResult | Literal["pending"] | None:
        """Return the result for *trace_id*.

        Returns:
            ``"pending"`` if still running, a :class:`PlanningResult` when done,
            or ``None`` if *trace_id* is unknown.
        """
        async with self._lock:
            if trace_id not in self._results:
                return None
            return self._results[trace_id]

    async def resolve_stale_checkpoints(self, max_hours: float = 1.0) -> int:
        """Auto-resolve checkpoints that have been waiting for operator input.

        Checks the checkpointer for interrupted threads and resumes any that
        have been paused beyond *max_hours* with ``operator_response=timeout``,
        resulting in a ``no_safe_plan`` decision.

        Works with any checkpointer that supports the ``alist`` API.
        Falls back to raw SQL for ``AsyncSqliteSaver`` when ``alist`` is
        unsupported.  MemorySaver returns 0 (in-memory only, no persistence).

        Returns:
            Number of threads resolved.
        """
        checkpointer = getattr(self._graph, "checkpointer", None)
        if checkpointer is None:
            return 0

        # Collect interrupted thread IDs
        cutoff = datetime.now(UTC) - timedelta(hours=max_hours)
        thread_ids: list[str] = []

        # Prefer the supported alist API
        try:
            seen: set[str] = set()
            async for cp in checkpointer.alist(None):
                tid = cp.config.get("configurable", {}).get("thread_id")
                if tid and tid not in seen:
                    seen.add(tid)
                    thread_ids.append(tid)
        except NotImplementedError:
            conn = getattr(checkpointer, "conn", None)
            if conn is None:
                return 0
            try:
                cursor = await conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
                rows = await cursor.fetchall()
                await cursor.close()
                thread_ids = [r[0] for r in rows]
            except Exception:
                return 0
        except Exception:
            logger.exception("TTL: error listing checkpoints")
            return 0

        resolved = 0
        for thread_id in thread_ids:
            try:
                config = {"configurable": {"thread_id": thread_id}}
                snap = self._graph.get_state(config)
                if snap is None or not snap.next:
                    continue
                if snap.created_at is not None:
                    try:
                        created_dt = datetime.fromisoformat(snap.created_at)
                        if created_dt >= cutoff:
                            continue
                    except (ValueError, TypeError):
                        continue
                await self.resume(thread_id, {"timeout": True})
                resolved += 1
                logger.info("TTL: auto-resolved stale checkpoint thread_id=%s", thread_id)
            except Exception:
                logger.exception("TTL: error resolving thread_id=%s", thread_id)
        return resolved


def _pending_incident_record(packet: MaydayPacket) -> IncidentRecord:
    anomaly_key = packet.anomalies[0].key if packet.anomalies else "UNKNOWN"
    severity = packet.anomalies[0].severity.value if packet.anomalies else "unknown"
    return IncidentRecord(
        incident_id=str(uuid.uuid4()),
        escalation_id=packet.trace_id,
        trace_id=packet.trace_id,
        device_id=packet.device_id,
        anomaly_key=anomaly_key,
        anomaly_type="detected",
        severity=severity,
        summary=f"device={packet.device_id} anomaly={anomaly_key} queued",
        raw_packet_json=packet.model_dump_json(),
        retrieved_incident_ids_json="[]",
        candidate_plan_json=None,
        validation_errors_json="[]",
        decision="pending",
        created_at=datetime.now(UTC),
    )


class SQSCloudOrchestrator:
    """Orchestrator that enqueues packets to SQS and reads results from Postgres.

    The API process writes a ``"pending"`` incident record to Postgres and
    pushes the packet onto the SQS queue.  One or more worker processes
    long-poll the queue, run the full planning workflow, and write the final
    decision back to Postgres.  ``get_result()`` polls Postgres directly, so
    it works correctly across multiple Fargate tasks.
    """

    def __init__(
        self,
        repo: IncidentRepositoryProtocol,
        sqs_queue: SQSQueue,
    ) -> None:
        self._repo = repo
        self._sqs = sqs_queue

    async def submit(self, packet: MaydayPacket) -> str:
        record = _pending_incident_record(packet)
        await self._repo.save_incident(record)
        await self._sqs.enqueue(
            {"trace_id": packet.trace_id, "packet": packet.model_dump(mode="json")}
        )
        _increment_cloud_metric("cloud_planning_requests_total")
        return packet.trace_id

    async def get_result(self, trace_id: str) -> PlanningResult | Literal["pending"] | None:
        record = await self._repo.get_incident_by_trace_id(trace_id)
        if record is None:
            return None
        if record.decision == "pending":
            return "pending"
        plan: Plan | None = None
        if record.candidate_plan_json:
            try:
                plan = Plan.model_validate_json(record.candidate_plan_json)
            except Exception:
                logger.warning("Failed to deserialise plan for trace_id=%s", trace_id)
        return PlanningResult(decision=record.decision, plan=plan)
