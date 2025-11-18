import logging

from fastapi import APIRouter, status

from kitchenwatch.edge.edge_fusion import EdgeFusion, VisionEmbedder
from kitchenwatch.edge.local_receiver import LocalReceiver
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.online_learner_state_estimator import OnlineLearnerStateEstimator
from kitchenwatch.edge.river_online_learner import RiverOnlineLearner
from kitchenwatch.simulation.models.fused_record import FusedRecord
from kitchenwatch.simulation.models.windowed_fused_record import WindowedFusedRecord

router = APIRouter()
logger = logging.getLogger(__name__)

blackboard = Blackboard()
vision_embedder = VisionEmbedder()
online_learner = RiverOnlineLearner()
state_estimator = OnlineLearnerStateEstimator(online_learner)
edge_fusion = EdgeFusion(blackboard, state_estimator=state_estimator, embedder=vision_embedder)
receiver = LocalReceiver(verbose=False, custom_logger=logger, edge_fusion=edge_fusion)


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
def ingest_record(record: FusedRecord | WindowedFusedRecord) -> dict[str, int | str]:
    """
    Receive a fused record (single-frame or windowed) from the simulator.
    Automatically validated via Pydantic models from `kitchenwatch.simulation.models`.
    """
    _ = receiver.ingest(record.model_dump())
    return {
        "message": "record accepted",
        "received_count": receiver._received_count,
        "record_type": record.__class__.__name__,
    }
