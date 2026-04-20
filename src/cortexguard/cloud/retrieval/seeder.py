"""Seed loading utilities for bootstrapping retrieval before real incidents exist."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from cortexguard.cloud.persistence.models import IncidentRecord, OutcomeRecord
from cortexguard.cloud.persistence.repository import IncidentRepositoryProtocol
from cortexguard.cloud.retrieval.store import RetrievalStore

_SEEDS_DIR = Path(__file__).parents[1] / "data" / "seeds"


def _load_incident(data: dict[str, object]) -> IncidentRecord:
    return IncidentRecord(
        incident_id=str(data["incident_id"]),
        escalation_id=str(data["escalation_id"]),
        trace_id=str(data["trace_id"]),
        device_id=str(data["device_id"]),
        anomaly_key=str(data["anomaly_key"]),
        anomaly_type=str(data["anomaly_type"]),
        severity=str(data["severity"]),
        summary=str(data["summary"]),
        raw_packet_json=str(data["raw_packet_json"]),
        retrieved_incident_ids_json=str(data["retrieved_incident_ids_json"]),
        candidate_plan_json=(
            str(data["candidate_plan_json"]) if data.get("candidate_plan_json") else None
        ),
        validation_errors_json=str(data["validation_errors_json"]),
        decision=str(data["decision"]),
        created_at=datetime.fromisoformat(str(data["created_at"])),
        rationale=str(data["rationale"]) if data.get("rationale") else None,
        confidence=float(str(data["confidence"])) if data.get("confidence") is not None else None,
    )


def _load_outcome(data: dict[str, object]) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_id=str(data["outcome_id"]),
        escalation_id=str(data["escalation_id"]),
        decision_id=str(data["decision_id"]),
        device_id=str(data["device_id"]),
        status=str(data["status"]),
        completed_at=datetime.fromisoformat(str(data["completed_at"])),
        notes=str(data["notes"]) if data.get("notes") else None,
        failure_reason=str(data["failure_reason"]) if data.get("failure_reason") else None,
        linked_at=datetime.fromisoformat(str(data["linked_at"])),
    )


class SeedLoader:
    def __init__(self, seeds_dir: Path = _SEEDS_DIR) -> None:
        self._seeds_dir = seeds_dir

    async def seed_if_empty(self, store: RetrievalStore, repo: IncidentRepositoryProtocol) -> int:
        existing = await repo.list_recent_incidents(limit=1)
        if existing:
            return 0

        count = 0
        for seed_file in sorted(self._seeds_dir.glob("*.json")):
            raw = json.loads(seed_file.read_text())
            incident = _load_incident(raw["incident"])
            outcome = _load_outcome(raw["outcome"])
            await repo.save_incident(incident)
            await repo.save_outcome(outcome)
            await store.index_incident(incident)
            count += 1

        return count
