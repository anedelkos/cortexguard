"""Persistence-layer records stored by the cloud planner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class IncidentRecord:
    incident_id: str
    escalation_id: str
    trace_id: str
    device_id: str
    anomaly_key: str
    anomaly_type: str
    severity: str
    summary: str
    raw_packet_json: str
    retrieved_incident_ids_json: str
    candidate_plan_json: str | None
    validation_errors_json: str
    decision: str
    created_at: datetime
    rationale: str | None = None
    confidence: float | None = None
    parent_incident_id: str | None = None
    source: str = "edge"
    operator_resolution_json: str | None = None
    retrieved_incidents_json: str | None = None


@dataclass
class OutcomeRecord:
    outcome_id: str
    escalation_id: str
    decision_id: str
    device_id: str
    status: str
    completed_at: datetime
    notes: str | None
    failure_reason: str | None
    linked_at: datetime
