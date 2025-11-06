import logging

from fastapi import APIRouter, status

from kitchenwatch.edge.local_receiver import LocalReceiver
from kitchenwatch.simulation.models.fused_record import FusedRecord
from kitchenwatch.simulation.models.windowed_fused_record import WindowedFusedRecord

router = APIRouter()
logger = logging.getLogger(__name__)

receiver = LocalReceiver(verbose=False, logger=logger)


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
def ingest_record(record: FusedRecord | WindowedFusedRecord) -> dict[str, int | str]:
    """
    Receive a fused record (single-frame or windowed) from the simulator.
    Automatically validated via Pydantic models from `kitchenwatch.simulation.models`.
    """
    receiver.ingest(record.model_dump())
    return {
        "message": "record accepted",
        "received_count": receiver.received_count,
        "record_type": record.__class__.__name__,
    }
