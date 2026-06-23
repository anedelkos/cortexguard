"""API routes for ingesting step-telemetry records from edge devices."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Query, Request, status
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from cortexguard.cloud.telemetry.models import StoredTelemetryRecord
from cortexguard.cloud.telemetry.repository import TelemetryRepositoryProtocol
from cortexguard.edge.models.telemetry import TelemetryBatch

logger = logging.getLogger(__name__)


class TelemetryRecordResponse(BaseModel):
    """Read-side view of a stored telemetry record."""

    device_id: str
    key: str
    outcome: str
    timestamp: str
    sensor_snapshot_json: str


def get_telemetry_router(
    repo: TelemetryRepositoryProtocol | None = None,
    telemetry_rate_limit: str = "60/minute",
    _limiter: Limiter | None = None,
) -> APIRouter:
    """Create the telemetry ingestion router with injected repository."""
    lim = _limiter if _limiter is not None else Limiter(key_func=get_remote_address)
    router = APIRouter()

    @router.post("/telemetry", status_code=status.HTTP_200_OK)
    @lim.limit(telemetry_rate_limit)
    async def ingest_telemetry(request: Request, batch: TelemetryBatch) -> dict[str, object]:
        """Ingest a batch of step-telemetry records from an edge device."""
        if repo is not None:
            records = [
                StoredTelemetryRecord(
                    device_id=rec.device_id,
                    key=rec.key,
                    outcome=rec.outcome,
                    timestamp=datetime.fromisoformat(rec.timestamp),
                    sensor_snapshot_json=(
                        rec.sensor_snapshot.model_dump_json()
                        if rec.sensor_snapshot is not None
                        else "{}"
                    ),
                )
                for rec in batch.records
            ]
            await repo.save_records(records)

        try:
            from cortexguard.cloud.runtime import cloud_telemetry_records_total

            cloud_telemetry_records_total.inc(len(batch.records))
        except ImportError:
            logger.debug("Cloud metrics unavailable while ingesting telemetry")

        logger.info("Ingested %d telemetry records", len(batch.records))
        return {"ok": True, "count": len(batch.records)}

    @router.get(
        "/telemetry/recent",
        status_code=status.HTTP_200_OK,
        response_model=list[TelemetryRecordResponse],
    )
    @lim.limit(telemetry_rate_limit)
    async def list_recent_telemetry(
        request: Request,
        limit: int = Query(default=20, ge=1, le=500),
    ) -> list[TelemetryRecordResponse]:
        """List recent telemetry records for debugging and visualization."""
        if repo is None:
            return []
        records = await repo.list_recent_records(limit=limit)
        return [
            TelemetryRecordResponse(
                device_id=r.device_id,
                key=r.key,
                outcome=r.outcome,
                timestamp=r.timestamp.isoformat(),
                sensor_snapshot_json=r.sensor_snapshot_json,
            )
            for r in records
        ]

    return router
