"""Response and payload schemas for cloud mayday and outcome endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class ExecutionOutcomeStatus(StrEnum):
    """Terminal status values reported by the edge after executing a cloud-issued plan."""

    completed = "completed"
    failed = "failed"
    aborted = "aborted"
    rejected_by_operator = "rejected_by_operator"


class MaydayAcceptedResponse(BaseModel):
    """Returned by the cloud API immediately after accepting a mayday escalation packet."""

    trace_id: str


class ExecutionOutcome(BaseModel):
    """Reported by the edge device after executing (or failing to execute) a cloud-issued plan."""

    escalation_id: str
    decision_id: str
    device_id: str
    status: ExecutionOutcomeStatus
    completed_at: datetime
    notes: str | None = None
    failure_reason: str | None = None


class RecentOutcomeResponse(BaseModel):
    """Read-side view of a recently recorded execution outcome."""

    outcome_id: str
    escalation_id: str
    decision_id: str
    device_id: str
    status: str
    completed_at: datetime
    notes: str | None = None
    failure_reason: str | None = None
    linked_at: datetime
