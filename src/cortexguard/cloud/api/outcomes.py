"""API routes for recording edge execution outcomes back into cloud storage."""

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Query, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from cortexguard.cloud.persistence.models import OutcomeRecord
from cortexguard.cloud.persistence.repository import IncidentRepositoryProtocol
from cortexguard.cloud.schemas.responses import ExecutionOutcome, RecentOutcomeResponse

logger = logging.getLogger(__name__)


def get_outcomes_router(
    repo: "IncidentRepositoryProtocol | None" = None,
    outcome_rate_limit: str = "30/minute",
    _limiter: "Limiter | None" = None,
) -> APIRouter:
    """Create the outcomes router with injected repository and rate limit string."""
    lim = _limiter if _limiter is not None else Limiter(key_func=get_remote_address)
    router = APIRouter()

    @router.post("/outcomes", status_code=status.HTTP_200_OK)
    @lim.limit(outcome_rate_limit)  # type: ignore[misc]
    async def record_outcome(request: Request, outcome: ExecutionOutcome) -> dict[str, object]:
        """Record the result of an edge-executed plan back into cloud storage."""
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

    @router.get(
        "/outcomes/recent",
        status_code=status.HTTP_200_OK,
        response_model=list[RecentOutcomeResponse],
    )
    @lim.limit(outcome_rate_limit)  # type: ignore[misc]
    async def list_recent_outcomes(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> list[RecentOutcomeResponse]:
        """List recently recorded execution outcomes for operator visibility and debugging."""
        if repo is None:
            return []
        records = await repo.list_recent_outcomes(limit=limit)
        return [
            RecentOutcomeResponse(
                outcome_id=record.outcome_id,
                escalation_id=record.escalation_id,
                decision_id=record.decision_id,
                device_id=record.device_id,
                status=record.status,
                completed_at=record.completed_at,
                notes=record.notes,
                failure_reason=record.failure_reason,
                linked_at=record.linked_at,
            )
            for record in records
        ]

    return router
