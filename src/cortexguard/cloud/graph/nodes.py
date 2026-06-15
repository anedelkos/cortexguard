"""LangGraph nodes implementing cloud persistence, retrieval, planning, and routing."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import replace as _dc_replace
from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END
from langgraph.types import interrupt
from opentelemetry import trace as _otel_trace

from cortexguard.cloud.graph.state import CloudPlanningState, ValidationResult
from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.cloud.persistence.repository import IncidentRepositoryProtocol
from cortexguard.cloud.planner.llm_client import LLMClientProtocol, PlannerRequest
from cortexguard.cloud.retrieval.store import RetrievalStoreProtocol
from cortexguard.cloud.validation.plan_validator import PlanValidatorProtocol
from cortexguard.edge.models.plan import Plan, PlanSource

_tracer = _otel_trace.get_tracer(__name__)

logger = logging.getLogger(__name__)


def _observe_cloud_metric_seconds(metric_name: str, duration_seconds: float) -> None:
    try:
        from cortexguard.cloud import runtime as cloud_runtime
    except ImportError:
        logger.debug("Cloud runtime unavailable while observing metric %s", metric_name)
        return
    metric = getattr(cloud_runtime, metric_name, None)
    if metric is not None:
        metric.observe(duration_seconds)


def _observe_labeled_histogram(metric_name: str, value: float, **labels: str) -> None:
    try:
        from cortexguard.cloud import runtime as cloud_runtime
    except ImportError:
        logger.debug("Cloud runtime unavailable while observing labeled metric %s", metric_name)
        return
    metric = getattr(cloud_runtime, metric_name, None)
    if metric is not None:
        metric.labels(**labels).observe(value)


def _increment_cloud_metric(metric_name: str) -> None:
    try:
        from cortexguard.cloud import runtime as cloud_runtime
    except ImportError:
        logger.debug("Cloud runtime unavailable while incrementing metric %s", metric_name)
        return
    metric = getattr(cloud_runtime, metric_name, None)
    if metric is not None:
        metric.inc()


def _increment_cloud_labeled_metric(metric_name: str, **labels: str) -> None:
    try:
        from cortexguard.cloud import runtime as cloud_runtime
    except ImportError:
        logger.debug("Cloud runtime unavailable while incrementing labeled metric %s", metric_name)
        return
    metric = getattr(cloud_runtime, metric_name, None)
    if metric is not None:
        metric.labels(**labels).inc()


def _normalise_plan(plan: Plan | None, trace_id: str) -> Plan | None:
    """Force correct provenance fields and regenerate any non-UUID IDs."""
    if plan is None:
        return None
    steps = [
        step.model_copy(update={"id": str(uuid.uuid4())}) if not _is_uuid(step.id) else step
        for step in plan.steps
    ]
    return plan.model_copy(
        update={
            "plan_id": str(uuid.uuid4()) if not _is_uuid(plan.plan_id) else plan.plan_id,
            "source": PlanSource.CLOUD_AGENT,
            "trace_id": trace_id,
            "steps": steps,
        }
    )  # type: ignore[no-any-return]


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def make_persist_incident_node(
    repo: IncidentRepositoryProtocol,
) -> Callable[[CloudPlanningState], Coroutine[Any, Any, CloudPlanningState]]:
    async def persist_incident(state: CloudPlanningState) -> CloudPlanningState:
        with _tracer.start_as_current_span("persist_incident"):
            try:
                incident_id = str(uuid.uuid4())
                packet = state["request"]
                anomaly_key = packet.anomalies[0].key if packet.anomalies else "UNKNOWN"
                severity = packet.anomalies[0].severity.value if packet.anomalies else "unknown"
                summary = (
                    f"device={packet.device_id} anomaly={anomaly_key} severity={severity} "
                    f"plan={packet.current_plan_id or 'none'}"
                )
                record = IncidentRecord(
                    incident_id=incident_id,
                    escalation_id=packet.trace_id,
                    trace_id=packet.trace_id,
                    device_id=packet.device_id,
                    anomaly_key=anomaly_key,
                    anomaly_type="detected",
                    severity=severity,
                    summary=summary,
                    raw_packet_json=packet.model_dump_json(),
                    retrieved_incident_ids_json=json.dumps(
                        [r["incident_id"] for r in state.get("retrieved_incidents", [])]
                    ),
                    candidate_plan_json=None,
                    validation_errors_json="[]",
                    decision="pending",
                    created_at=datetime.now(UTC),
                )
                await repo.save_incident(record)
                return {
                    **state,
                    "incident_id": incident_id,
                    "retrieved_incidents": [],
                    "retrieved_incident_records": [],
                }
            except Exception as exc:
                errors = list(state["errors"]) + [str(exc)]
                return {
                    **state,
                    "errors": errors,
                    "decision": "needs_human",
                    "candidate_plan": None,
                }

    return persist_incident


def make_retrieve_similar_incidents_node(
    retrieval_store: RetrievalStoreProtocol | None,
    repo: IncidentRepositoryProtocol | None = None,
) -> Callable[[CloudPlanningState], Coroutine[Any, Any, CloudPlanningState]]:
    async def retrieve_similar_incidents(state: CloudPlanningState) -> CloudPlanningState:
        with _tracer.start_as_current_span("retrieve_similar_incidents"):
            try:
                if retrieval_store is None:
                    return {
                        **state,
                        "retrieved_incidents": [],
                        "retrieved_incident_records": [],
                    }
                t0 = time.monotonic()
                results: list[tuple[str, float, IncidentRecord | None]] = (
                    await retrieval_store.retrieve_similar(state["request"])
                )
                _observe_cloud_metric_seconds(
                    "cloud_retrieval_duration_seconds", time.monotonic() - t0
                )

                if results:
                    packet = state["request"]
                    anomaly_key = packet.anomalies[0].key if packet.anomalies else "UNKNOWN"
                    _observe_labeled_histogram(
                        "cloud_retrieval_similarity_score",
                        results[0][1],
                        anomaly_key=anomaly_key,
                    )

                retrieved_incidents: list[dict[str, Any]] = [
                    {"incident_id": rid, "similarity_score": score} for rid, score, _ in results
                ]
                # Records are already loaded by the store (for re-ranking). The store handles
                # re-ranking; the node handles hydration into state for the LLM prompt. Both
                # use the same loaded records, no second repo fetch needed.
                records: list[IncidentRecord] = [rec for _, _, rec in results if rec is not None]
                if repo is not None:
                    incident_id = state.get("incident_id")
                    if incident_id is not None:
                        current = await repo.get_incident(str(incident_id))
                        if current is not None:
                            updated = _dc_replace(
                                current,
                                retrieved_incident_ids_json=json.dumps(
                                    [rid for rid, _, _ in results]
                                ),
                                retrieved_incidents_json=json.dumps(retrieved_incidents),
                            )
                            await repo.save_incident(updated)
                return {
                    **state,
                    "retrieved_incidents": retrieved_incidents,
                    "retrieved_incident_records": records,
                }
            except Exception as exc:
                errors = list(state["errors"]) + [str(exc)]
                return {
                    **state,
                    "errors": errors,
                    "decision": "needs_human",
                    "candidate_plan": None,
                }

    return retrieve_similar_incidents


def make_generate_candidate_plan_node(
    llm_client: LLMClientProtocol | None,
    capability_catalog_json: str = "[]",
) -> Callable[[CloudPlanningState], Coroutine[Any, Any, CloudPlanningState]]:
    async def generate_candidate_plan(state: CloudPlanningState) -> CloudPlanningState:
        with _tracer.start_as_current_span("generate_candidate_plan"):
            try:
                if llm_client is None:
                    return {**state, "candidate_plan": None, "confidence": 0.5, "rationale": "stub"}
                packet = state["request"]
                anomaly_key = packet.anomalies[0].key if packet.anomalies else "UNKNOWN"
                severity = packet.anomalies[0].severity.value if packet.anomalies else "unknown"
                retrieved_summaries = [
                    r.summary for r in state.get("retrieved_incident_records", [])  # type: ignore[attr-defined]
                ]
                request = PlannerRequest(
                    escalation_summary=f"device={packet.device_id} anomaly={anomaly_key}",
                    state_summary=json.dumps(packet.state_estimate or {}),
                    retrieved_summaries=retrieved_summaries,
                    capability_catalog_json=capability_catalog_json,
                    anomaly_key=anomaly_key,
                    severity=severity,
                )
                t0 = time.monotonic()
                try:
                    planner_response = await llm_client.generate_structured_plan(request)
                except Exception as throttle_exc:
                    from cortexguard.cloud.planner.throttler import LLMThrottleError

                    if isinstance(throttle_exc, LLMThrottleError):
                        logger.warning(
                            "LLM unavailable (outcome=%s), routing to needs_human",
                            throttle_exc.outcome,
                        )
                        return {
                            **state,
                            "candidate_plan": None,
                            "decision": "needs_human",
                            "rationale": f"LLM provider unavailable: {throttle_exc.outcome}",
                        }
                    raise
                _observe_cloud_metric_seconds("cloud_llm_duration_seconds", time.monotonic() - t0)
                plan = _normalise_plan(planner_response.candidate_plan, packet.trace_id)
                return {
                    **state,
                    "candidate_plan": plan,
                    "confidence": planner_response.confidence,
                    "rationale": planner_response.rationale,
                    "needs_human_review": planner_response.needs_human_review,
                }
            except Exception as exc:
                errors = list(state["errors"]) + [str(exc)]
                return {
                    **state,
                    "errors": errors,
                    "decision": "needs_human",
                    "candidate_plan": None,
                }

    return generate_candidate_plan


def make_validate_candidate_plan_node(
    validator: PlanValidatorProtocol | None,
) -> Callable[[CloudPlanningState], Coroutine[Any, Any, CloudPlanningState]]:
    async def validate_candidate_plan(state: CloudPlanningState) -> CloudPlanningState:
        with _tracer.start_as_current_span("validate_candidate_plan"):
            try:
                if validator is None:
                    result = ValidationResult(passed=True, errors=[], risk_level="low")
                    return {**state, "validation_result": result}
                result = validator.validate(
                    state.get("candidate_plan"),
                    state.get("confidence") or 0.0,
                    state.get("needs_human_review", False),
                )
                if not result.passed:
                    _increment_cloud_metric("cloud_validation_failures_total")
                return {**state, "validation_result": result}
            except Exception as exc:
                errors = list(state["errors"]) + [str(exc)]
                return {
                    **state,
                    "errors": errors,
                    "decision": "needs_human",
                    "candidate_plan": None,
                }

    return validate_candidate_plan


async def route_decision(state: CloudPlanningState) -> CloudPlanningState:
    with _tracer.start_as_current_span("route_decision"):
        try:
            if state.get("errors"):
                return {**state, "decision": "needs_human"}

            candidate_plan = state.get("candidate_plan")
            validation_result = state.get("validation_result")

            if candidate_plan is None:
                decision = "no_safe_plan"
            elif validation_result is None or not validation_result.passed:
                decision = "needs_human"
            else:
                decision = "plan_ready"

            _increment_cloud_labeled_metric("cloud_decisions_total", decision=decision)
            return {**state, "decision": decision}
        except Exception as exc:
            errors = list(state["errors"]) + [str(exc)]
            return {**state, "errors": errors, "decision": "needs_human", "candidate_plan": None}


async def pause_for_operator(state: CloudPlanningState) -> CloudPlanningState:
    with _tracer.start_as_current_span("pause_for_operator"):
        op = state.get("operator_response") or {}
        if op.get("timeout"):
            return {**state, "decision": "no_safe_plan"}
        if op.get("approved") is True:
            return {
                **state,
                "candidate_plan": op.get("plan_override", state.get("candidate_plan")),
                "decision": "plan_ready",
            }
        if op.get("action") == "reject":
            return {**state, "decision": "no_safe_plan"}

        packet = state["request"]
        anomaly_key = packet.anomalies[0].key if packet.anomalies else "UNKNOWN"
        logger.warning(
            "Cloud planner decision=needs_human escalation_id=%s anomaly_key=%s",
            packet.trace_id,
            anomaly_key,
        )
        _increment_cloud_metric("cloud_needs_human_total")
        await interrupt(
            {
                "incident_id": state.get("incident_id"),
                "reason": "needs_human_review",
                "request": state["request"],
                "candidate_plan": state.get("candidate_plan"),
            }
        )
        return {**state, "decision": "no_safe_plan"}


def route_after_decision(state: CloudPlanningState) -> str:
    """Return next node name after route_decision.

    Only pause for operator when needs_human_review was requested by the LLM.
    Validation failures and node errors just return ``needs_human`` without
    pausing, the operator can see those via MCP directly.
    """
    decision = state.get("decision")
    if decision == "needs_human" and state.get("needs_human_review"):
        return "pause_for_operator"
    return END  # type: ignore[no-any-return]
