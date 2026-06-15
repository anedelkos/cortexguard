"""State definitions shared across the cloud planning workflow graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.edge.models.mayday_packet import MaydayPacket
from cortexguard.edge.models.plan import Plan


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    risk_level: str = "low"


class CloudPlanningState(TypedDict):
    request: NotRequired[MaydayPacket]
    incident_id: str | None
    retrieved_incidents: list[dict[str, Any]]
    retrieved_incident_records: list[IncidentRecord]
    candidate_plan: Plan | None
    validation_result: ValidationResult | None
    decision: str | None
    rationale: str | None
    confidence: float | None
    needs_human_review: bool
    errors: list[str]
    operator_response: NotRequired[dict[str, Any] | None]


class ResumeInput(TypedDict, total=False):
    """Partial state passed on resume -- only the operator response is set.

    LangGraph merges this with the checkpointed state, so missing keys
    are filled from the checkpoint.
    """

    operator_response: dict[str, Any] | None
