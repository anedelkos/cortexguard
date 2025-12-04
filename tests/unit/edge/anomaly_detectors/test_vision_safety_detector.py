import asyncio
from datetime import UTC, datetime
from typing import Any

from kitchenwatch.core.interfaces.base_detector import BaseDetector
from kitchenwatch.edge.detectors.vision.vision_safety_detector import (
    VisionSafetyDetector,
    VisionSafetyDetectorConfig,
)
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot


class DummySnapshot(FusionSnapshot):
    # simple helper to construct snapshots without full WindowedFusedRecord plumbing
    pass


async def _run_detect(det: BaseDetector, snap: FusionSnapshot) -> dict[str, Any]:
    return await det.detect(snap)


def test_human_proximity_violation():
    det = VisionSafetyDetector(VisionSafetyDetectorConfig(safety_radius_m=0.5, min_confidence=0.6))
    snap = FusionSnapshot(
        id="snap-1",
        timestamp=datetime.now(UTC),
        sensors={
            "vision_objects": [
                {
                    "label": "person",
                    "distance_m": 0.2,
                    "confidence": 0.9,
                    "bbox_id": "b1",
                    "camera_id": "camA",
                },
            ]
        },
        derived={},
    )
    anom = asyncio.run(_run_detect(det, snap))
    assert anom is not None
    assert anom["key"] == "HUMAN_PROXIMITY_VIOLATION"
    assert 0.0 <= anom["anomaly_score"] <= 1.0
    assert anom["severity"] == "high"
    assert anom["metadata"]["snapshot_id"] == "snap-1"


def test_no_anomaly_when_far():
    det = VisionSafetyDetector(VisionSafetyDetectorConfig(safety_radius_m=0.5))
    snap = FusionSnapshot(
        id="snap-2",
        timestamp=datetime.now(UTC),
        sensors={"vision_objects": [{"label": "person", "distance_m": 0.8, "confidence": 0.9}]},
        derived={},
    )
    anom = asyncio.run(_run_detect(det, snap))
    assert anom["key"] == "NO_ANOMALY"
    assert anom["anomaly_score"] == 0.0


def test_vision_occlusion_persistent():
    det = VisionSafetyDetector(
        VisionSafetyDetectorConfig(occlusion_area_threshold_pct=60.0, occlusion_min_duration_s=3.0)
    )
    snap = FusionSnapshot(
        id="snap-3",
        timestamp=datetime.now(UTC),
        sensors={"vision_occlusion": {"area_pct": 70.0, "duration_s": 5.0}},
        derived={},
    )
    anom = asyncio.run(_run_detect(det, snap))
    assert anom["key"] == "VISION_OCCLUSION_PERSISTENT"
    assert anom["severity"] == "medium"
    assert anom["metadata"]["area_pct"] == 70.0
