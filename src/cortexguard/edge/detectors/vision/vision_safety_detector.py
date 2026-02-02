from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from cortexguard.core.interfaces.base_detector import BaseDetector
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot

logger = logging.getLogger(__name__)

"""
Minimal, testable VisionSafetyDetector that consumes FusionSnapshot produced by EdgeFusion.

- Conforms to BaseDetector protocol: async def detect(snapshot: FusionSnapshot) -> dict[str, Any]
- Reads snapshot.id, snapshot.sensors, and snapshot.derived
- Returns dict with keys: key, anomaly_score, severity, metadata
- Lightweight, reflex-oriented: HUMAN_PROXIMITY_VIOLATION, VISION_OCCLUSION_PERSISTENT, NO_ANOMALY
"""


class VisionObject(TypedDict, total=False):
    """Normalized vision object record expected inside FusionSnapshot.sensors['vision_objects']."""

    label: str
    distance_m: float
    confidence: float
    bbox_id: str
    camera_id: str


@dataclass(slots=True)
class VisionSafetyDetectorConfig:
    safety_radius_m: float = 0.5
    occlusion_area_threshold_pct: float = 60.0
    occlusion_min_duration_s: float = 3.0
    min_confidence: float = 0.5


class VisionSafetyDetector(BaseDetector):
    """
    Minimal detector focused on human-in-scene safety and persistent occlusion.

    Expected FusionSnapshot shape:
      snapshot.id: str
      snapshot.sensors: {
        "vision_objects": list[dict(label, distance_m, confidence, bbox_id?, camera_id? )],
        "vision_occlusion": {"area_pct": float, "duration_s": float},
        "image_embedding": optional tensor or None,
        ...
      }
      snapshot.derived: dict (optional derived features)
    """

    def __init__(self, config: VisionSafetyDetectorConfig | None = None) -> None:
        self.config = config or VisionSafetyDetectorConfig()
        logger.info(
            "VisionSafetyDetector initialized: safety_radius=%.3fm occlusion>=%.1f%% for >=%.1fs min_conf=%.2f",
            self.config.safety_radius_m,
            self.config.occlusion_area_threshold_pct,
            self.config.occlusion_min_duration_s,
            self.config.min_confidence,
        )

    async def detect(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        """
        Analyze the FusionSnapshot and return a protocol-compliant dict.

        Always returns a dict with keys:
          - key: str
          - anomaly_score: float in [0.0, 1.0]
          - severity: "low" | "medium" | "high"
          - metadata: dict[str, Any]
        """
        # Defensive access to snapshot fields
        snapshot_id = getattr(snapshot, "id", None)
        sensors = getattr(snapshot, "sensors", {}) or {}
        derived = getattr(snapshot, "derived", {}) or {}

        # 1) Persistent occlusion (Tier 1)
        occlusion = sensors.get("vision_occlusion")
        if isinstance(occlusion, dict):
            try:
                area_pct = float(occlusion.get("area_pct", 0.0))
                duration_s = float(occlusion.get("duration_s", 0.0))
            except (TypeError, ValueError):
                area_pct = 0.0
                duration_s = 0.0

            if (
                area_pct >= self.config.occlusion_area_threshold_pct
                and duration_s >= self.config.occlusion_min_duration_s
            ):
                score = min(1.0, area_pct / 100.0)
                return {
                    "key": "VISION_OCCLUSION_PERSISTENT",
                    "anomaly_score": float(score),
                    "severity": "medium",
                    "metadata": {
                        "snapshot_id": snapshot_id,
                        "area_pct": area_pct,
                        "duration_s": duration_s,
                        # include a small derived hint if available
                        "vision_confidence_hint": derived.get("vision_confidence_hint"),
                    },
                }

        # 2) Human proximity (Tier 0 reflex)
        raw_objects: Sequence[VisionObject] = cast(
            Sequence[VisionObject], sensors.get("vision_objects", [])
        )
        if isinstance(raw_objects, list):
            for raw in raw_objects:
                if not isinstance(raw, dict):
                    continue
                label = str(raw.get("label", "")).lower()
                try:
                    confidence = float(raw.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                try:
                    distance_m = float(raw.get("distance_m", float("inf")))
                except (TypeError, ValueError):
                    distance_m = float("inf")

                if label == "person" and confidence >= self.config.min_confidence:
                    if distance_m < self.config.safety_radius_m:
                        radius = max(self.config.safety_radius_m, 1e-6)
                        score = max(0.0, min(1.0, 1.0 - (distance_m / radius)))
                        return {
                            "key": "HUMAN_PROXIMITY_VIOLATION",
                            "anomaly_score": float(score),
                            "severity": "high",
                            "metadata": {
                                "snapshot_id": snapshot_id,
                                "camera_id": raw.get("camera_id"),
                                "bbox_id": raw.get("bbox_id"),
                                "distance_m": distance_m,
                                "confidence": confidence,
                                # small derived hints for correlation
                                "ema_distance_hint": derived.get("person_distance_ema"),
                            },
                        }

        # 3) No anomaly found
        return {
            "key": "NO_ANOMALY",
            "anomaly_score": 0.0,
            "severity": "low",
            "metadata": {"snapshot_id": snapshot_id},
        }
