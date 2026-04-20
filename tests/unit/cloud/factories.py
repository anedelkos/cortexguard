from __future__ import annotations

import uuid
from datetime import UTC, datetime

from cortexguard.cloud.persistence.models import IncidentRecord, OutcomeRecord


def make_incident(**overrides: object) -> IncidentRecord:
    base = IncidentRecord(
        incident_id=str(uuid.uuid4()),
        escalation_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        device_id="robot-arm-01",
        anomaly_key="S1.1_MISGRASP",
        anomaly_type="repeated_failure",
        severity="high",
        summary="Repeated grip failures on part A",
        raw_packet_json="{}",
        retrieved_incident_ids_json="[]",
        candidate_plan_json=None,
        validation_errors_json="[]",
        decision="plan_ready",
        created_at=datetime.now(UTC),
        rationale=None,
        confidence=None,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def make_outcome(escalation_id: str = "", **overrides: object) -> OutcomeRecord:
    base = OutcomeRecord(
        outcome_id=str(uuid.uuid4()),
        escalation_id=escalation_id or str(uuid.uuid4()),
        decision_id=str(uuid.uuid4()),
        device_id="robot-arm-01",
        status="completed",
        completed_at=datetime.now(UTC),
        notes=None,
        failure_reason=None,
        linked_at=datetime.now(UTC),
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base
