"""State definitions shared across the cloud planning workflow graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.edge.models.mayday_packet import MaydayPacket
from cortexguard.edge.models.plan import Plan


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    risk_level: str = "low"


class CloudPlanningState(TypedDict):
    request: MaydayPacket
    incident_id: str | None
    retrieved_incidents: list[IncidentRecord]
    candidate_plan: Plan | None
    validation_result: ValidationResult | None
    decision: str | None
    rationale: str | None
    confidence: float | None
    needs_human_review: bool
    errors: list[str]
