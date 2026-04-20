"""API routes for recording edge execution outcomes back into cloud storage."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, status

from cortexguard.cloud.persistence.models import OutcomeRecord
from cortexguard.cloud.persistence.repository import IncidentRepositoryProtocol
from cortexguard.cloud.schemas.responses import ExecutionOutcome

logger = logging.getLogger(__name__)


def get_outcomes_router(repo: IncidentRepositoryProtocol | None = None) -> APIRouter:
    router = APIRouter()

    @router.post("/outcomes", status_code=status.HTTP_200_OK)
    async def record_outcome(outcome: ExecutionOutcome) -> dict[str, object]:
        if repo is not None:
            record = OutcomeRecord(
                outcome_id=str(uuid.uuid4()),
                escalation_id=outcome.escalation_id,
                decision_id=outcome.decision_id,
                device_id=outcome.device_id,
                status=outcome.status.value,
                completed_at=outcome.completed_at,
                notes=outcome.notes,
                failure_reason=outcome.failure_reason,
                linked_at=datetime.now(UTC),
            )
            await repo.save_outcome(record)
        try:
            from cortexguard.cloud.runtime import cloud_outcome_status_total

            cloud_outcome_status_total.labels(status=outcome.status.value).inc()
        except ImportError:
            logger.debug("Cloud metrics unavailable while recording execution outcome")
        return {"ok": True, "escalation_id": outcome.escalation_id}

    return router
