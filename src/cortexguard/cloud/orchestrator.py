"""Asynchronous orchestration of cloud planning requests and final results."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from datetime import UTC, datetime
from typing import Literal

from opentelemetry import trace as _otel_trace

from cortexguard.cloud.graph.state import CloudPlanningState
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
    """Completed planning outcome — distinct from the ``"pending"`` sentinel and unknown IDs."""

    decision: str
    plan: Plan | None


class CloudOrchestrator:
    def __init__(
        self,
        repo: IncidentRepositoryProtocol,
        retrieval_store: RetrievalStoreProtocol | None,
        llm_client: LLMClientProtocol | None,
        validator: PlanValidatorProtocol | None = None,
    ) -> None:
        self._repo = repo
        self._graph = build_graph(repo, retrieval_store, llm_client, validator)
        self._results: dict[str, PlanningResult | Literal["pending"]] = {}
        self._lock = asyncio.Lock()

    async def submit(self, packet: MaydayPacket) -> str:
        trace_id = packet.trace_id
        async with self._lock:
            self._results[trace_id] = "pending"
        _increment_cloud_metric("cloud_planning_requests_total")
        asyncio.create_task(self._run(trace_id, packet))
        return trace_id

    async def run_once(self, packet: MaydayPacket) -> None:
        """Run the planning workflow and persist the result without updating the in-process dict.

        Used by the SQS worker — the final decision is written to the incident store
        by :meth:`_finalise_incident`, making it visible to :class:`SQSCloudOrchestrator`.
        """
        await self._execute(packet.trace_id, packet)

    async def _run(self, trace_id: str, packet: MaydayPacket) -> None:
        decision, plan = await self._execute(trace_id, packet)
        async with self._lock:
            self._results[trace_id] = PlanningResult(decision=decision, plan=plan)

    async def _execute(self, trace_id: str, packet: MaydayPacket) -> tuple[str, Plan | None]:
        final_state: CloudPlanningState | None = None
        confidence: float = 0.0
        rationale: str = ""
        t0 = time.monotonic()
        with _tracer.start_as_current_span("cloud_planning_workflow"):
            _span = _otel_trace.get_current_span()
            try:
                final_state = await run_planning_workflow(packet, self._graph)
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
        await self._finalise_incident(final_state, decision, plan)
        logger.info(
            "cloud_plan decision=%s confidence=%.2f steps=%d trace_id=%s | %s",
            decision,
            confidence,
            len(plan.steps) if plan else 0,
            trace_id,
            rationale[:200],
        )
        return decision, plan

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
