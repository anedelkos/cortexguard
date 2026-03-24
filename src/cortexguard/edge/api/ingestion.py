from __future__ import annotations

import logging
import time

from fastapi import APIRouter, status

from cortexguard.edge.local_receiver import LocalReceiver
from cortexguard.edge.utils.metrics import http_request_duration_ms, http_requests_total
from cortexguard.simulation.models.fused_record import FusedRecord
from cortexguard.simulation.models.windowed_fused_record import WindowedFusedRecord

logger = logging.getLogger(__name__)


def get_ingestion_router(receiver: LocalReceiver) -> APIRouter:
    """
    Factory function to create the ingestion router with injected dependencies.

    The LocalReceiver dependency is injected from the EdgeRuntime composition root,
    ensuring all ingested data flows into the shared Blackboard instance.
    """
    router = APIRouter()

    # NOTE: The route handler must be async and use 'await' because LocalReceiver.ingest is an async method.
    @router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
    async def ingest_record(record: FusedRecord | WindowedFusedRecord) -> dict[str, int | str]:
        """
        Receive a fused record (single-frame or windowed) from the simulator.
        Automatically validated via Pydantic models from `cortexguard.simulation.models`.
        """
        start = time.perf_counter()
        status_code = "202"
        try:
            await receiver.ingest(record)
            return {
                "message": "record accepted",
                "received_count": receiver.received_count,
                "record_type": record.__class__.__name__,
            }
        except Exception:
            status_code = "500"
            raise
        finally:
            http_requests_total.labels(method="POST", status_code=status_code).inc()
            http_request_duration_ms.observe((time.perf_counter() - start) * 1000)

    return router
