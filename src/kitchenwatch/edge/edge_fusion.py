from __future__ import annotations

import logging
import statistics
import uuid
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import torch
from PIL import Image
from torchvision import models, transforms

from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.scene_graph import SceneGraph, SceneObject, SceneRelationship
from kitchenwatch.edge.online_learner_state_estimator import OnlineLearnerStateEstimator
from kitchenwatch.simulation.models.windowed_fused_record import WindowedFusedRecord

# Default smoothing factor for EMA (0 < alpha <= 1)
# Lower alpha = more smoothing, higher alpha = more responsive
DEFAULT_ALPHA = 0.1

logger = logging.getLogger(__name__)


class VisionEmbedder:
    """Lightweight ResNet-based embedder for images."""

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._model = torch.nn.Sequential(*list(self._model.children())[:-1])  # Remove classifier
        self._model.eval().to(device)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def embed(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0).to(self._device)
        emb = cast(torch.Tensor, self._model(x).squeeze())
        return emb.cpu()


def _mock_vision_inference(
    record: WindowedFusedRecord, image_embedding: Any
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """
    Fast demo stub that synthesizes vision_objects and vision_occlusion.
    - Produces one 'person' at a pseudo-random distance derived from a numeric field in the window.
    - Produces an occlusion hint when many objects or a synthetic condition is met.
    Replace with real model inference later.
    """
    # Simple deterministic heuristic: use a numeric field from the first sample if present
    distance_m = float("inf")
    confidence = 0.0
    try:
        first = record.sensor_window[0]
        # try to use a numeric field if available (e.g., 'force' or 'distance_raw')
        for _k, v in first.model_dump().items():
            if isinstance(v, (int, float)):
                distance_m = max(0.05, float(v) / 10.0)  # scale to meters for demo
                confidence = 0.8
                break
    except Exception:
        distance_m = 1.0
        confidence = 0.5

    vision_objects = [
        {
            "label": "person",
            "distance_m": distance_m,
            "confidence": confidence,
            "bbox_id": "demo-bbox-1",
            "camera_id": "demo-cam",
        }
    ]

    # Simple occlusion heuristic: if window size is small or embedding is None, no occlusion
    occlusion = None
    if len(record.sensor_window) > 5 and confidence < 0.6:
        occlusion = {"area_pct": 70.0, "duration_s": 4.0}

    return vision_objects, occlusion


def _to_scene_object(v: dict[str, Any], ema_state: dict[str, float]) -> SceneObject:
    """Convert a vision detection dict into a SceneObject."""
    obj_id = v.get("bbox_id") or v.get("id") or uuid.uuid4().hex
    props: dict[str, Any] = {}

    if (dm := v.get("distance_m")) is not None:
        try:
            props["distance_m"] = float(dm)
        except Exception:
            props["distance_m"] = None

    props["confidence"] = float(v.get("confidence", 0.0))

    ema_key = f"{v.get('label')}_distance"
    if ema_key in ema_state:
        props["distance_ema"] = ema_state[ema_key]

    return SceneObject(
        id=str(obj_id),
        label=str(v.get("label", "unknown")),
        location_2d=v.get("bbox"),
        pose_3d=None,
        properties=props,
    )


def _is_near(a: SceneObject, b: SceneObject, threshold_m: float = 0.5) -> bool:
    """Proximity heuristic using distance_m or bbox overlap (normalized coords)."""
    da = a.properties.get("distance_m")
    db = b.properties.get("distance_m")
    if isinstance(da, (int, float)) and isinstance(db, (int, float)):
        return abs(da - db) <= threshold_m or da <= threshold_m or db <= threshold_m

    bbox_a = a.location_2d
    bbox_b = b.location_2d
    if bbox_a and bbox_b and len(bbox_a) == 4 and len(bbox_b) == 4:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        if area_a > 0 and (inter_area / area_a) > 0.1:
            return True

    return False


def _build_scene_graph_from_vision(
    timestamp: datetime,
    vision_objects: Iterable[dict[str, Any]],
    ema_state: dict[str, float],
) -> SceneGraph:
    """Build a SceneGraph from vision_objects and EMA hints."""
    objs: list[SceneObject] = []
    rels: list[SceneRelationship] = []

    for v in vision_objects:
        try:
            objs.append(_to_scene_object(v, ema_state))
        except Exception:
            logger.debug("Skipping malformed vision object: %r", v)

    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i == j:
                continue
            if _is_near(a, b):
                rels.append(SceneRelationship(source_id=a.id, relationship="near", target_id=b.id))

            if (
                a.properties.get("confidence", 0.0) < 0.5
                and b.properties.get("confidence", 0.0) > 0.7
            ):
                if a.location_2d and b.location_2d:
                    rels.append(
                        SceneRelationship(source_id=a.id, relationship="occluding", target_id=b.id)
                    )

    return SceneGraph(timestamp=timestamp, objects=objs, relationships=rels)


class EdgeFusion:
    """
    Online sensor fusion using Exponential Moving Average (EMA).

    Processes windowed sensor records and maintains smoothed state via EMA.
    Each sample in a window updates the EMA sequentially, giving recency bias
    within the window itself.

    EMA Formula: new_ema = α * observation + (1-α) * old_ema
    - α near 0: Heavy smoothing (slow response)
    - α near 1: Light smoothing (fast response)

    Production Considerations (not implemented):
    - Adaptive alpha based on signal characteristics
    - Outlier rejection before EMA update
    - Multi-rate fusion (different α per sensor)
    - State persistence for warm restarts
    - Kalman filtering for better noise handling

    Design Choice: EMA over Kalman
    - EMA: Simple, low compute, no model required
    - Kalman: More accurate but needs process/measurement models
    - For edge deployment, EMA sufficient given stable sensors
    """

    def __init__(
        self,
        blackboard: Blackboard,
        state_estimator: OnlineLearnerStateEstimator | None = None,
        alpha: float = DEFAULT_ALPHA,
        custom_logger: logging.Logger | None = None,
        embedder: VisionEmbedder | None = None,
    ):
        """
        Initialize sensor fusion engine.

        Args:
            blackboard: Shared state for intent and snapshot updates
            alpha: EMA smoothing factor (0 < alpha <= 1)
            custom_logger: Optional logger instance

        Raises:
            ValueError: If alpha not in valid range
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"Alpha must be in (0, 1], got {alpha}")

        self._blackboard = blackboard
        self._alpha = alpha
        self._logger = custom_logger or logger
        self.embedder = embedder
        self._state_estimator: OnlineLearnerStateEstimator | None = state_estimator

        # EMA state: sensor_key -> smoothed_value
        self._ema_state: dict[str, float] = {}
        self._last_snapshot: FusionSnapshot | None = None

        # Metrics (for observability)
        self._samples_processed = 0
        self._records_processed = 0

    async def process_record(self, record: WindowedFusedRecord) -> FusionSnapshot:
        """
        Process a windowed sensor record and update fusion state.

        Strategy: Applies EMA to each sample in window sequentially.
        This gives temporal ordering within the window - later samples
        influence the EMA more than earlier ones.

        Args:
            record: Windowed fused sensor data from simulator

        Returns:
            FusionSnapshot with smoothed sensor values and derived features
        """

        # Process each sample in the window
        for sample in record.sensor_window:
            self._update_ema_state(sample)
            self._samples_processed += 1

        self._records_processed += 1

        # Compute image embedding (one per window)
        image_embedding = None
        if self.embedder is not None and hasattr(record, "rgb_path") and record.rgb_path:
            try:
                img = Image.open(record.rgb_path).convert("RGB")
                image_embedding = self.embedder.embed(img)
            except Exception as e:
                self._logger.warning(f"Failed to embed image {record.rgb_path}: {e}")

        serialized_embedding: Any
        if image_embedding is None:
            serialized_embedding = None
        elif isinstance(image_embedding, torch.Tensor):
            # move to CPU and convert to plain Python list for safe serialization/storage
            serialized_embedding = image_embedding.cpu().numpy().tolist()
        else:
            # fallback: keep as-is (could be already a list/None/other)
            serialized_embedding = image_embedding

        vision_objects, vision_occlusion = _mock_vision_inference(record, image_embedding)
        # build timezone-aware scene graph and publish to blackboard if available
        sg_timestamp = datetime.fromtimestamp(record.timestamp_ns / 1e9, tz=ZoneInfo("UTC"))
        scene_graph = _build_scene_graph_from_vision(
            sg_timestamp, vision_objects or [], self._ema_state
        )

        try:
            await self._blackboard.set_scene_graph(scene_graph)  # type: ignore[attr-defined]
        except AttributeError:
            logger.debug("Blackboard.set_scene_graph not available; skipping scene graph publish")

        # Create snapshot from current EMA state
        snapshot = FusionSnapshot(
            id=uuid.uuid4().hex,
            timestamp=datetime.fromtimestamp(record.timestamp_ns / 1e9),
            sensors={
                "raw": record.sensor_window,
                "ema_smoothed": self._ema_state.copy(),
                "window_stats": self._compute_window_stats(record.sensor_window),
                "image_embedding": serialized_embedding,
                "vision_objects": vision_objects,
                "vision_occlusion": vision_occlusion,
                "scene_graph_summary": [
                    {"id": o.id, "label": o.label} for o in scene_graph.objects
                ],
            },
            derived=self._ema_state.copy(),  # Smoothed values as derived features
        )

        self._last_snapshot = snapshot
        await self._blackboard.update_fusion_snapshot(snapshot)

        self._logger.debug(f"Updated snapshot with (window_size={len(record.sensor_window)})")

        if self._state_estimator:
            try:
                estimate = await self._state_estimator.update(snapshot)
                await self._blackboard.update_state_estimate(estimate)
            except Exception as e:
                self._logger.exception(f"StateEstimator failed: {e}")

        return snapshot

    def _update_ema_state(self, sample: Any) -> None:
        """
        Update EMA state with a single sensor sample.

        Skips:
        - Timestamp fields
        - None values
        - Non-numeric values

        Args:
            sample: Single sensor reading (Pydantic model)
        """
        for key, value in sample.model_dump().items():
            # Skip metadata and null values
            if key == "timestamp_ns" or value is None:
                continue

            # Only process numeric values (int, float)
            if not isinstance(value, (int, float)):
                self._logger.debug(f"Skipping non-numeric field: {key}={type(value).__name__}")
                continue

            # Initialize or update EMA
            if key not in self._ema_state:
                # First observation: initialize EMA
                self._ema_state[key] = float(value)
            else:
                # Subsequent: apply EMA formula
                # new = α * observation + (1-α) * old
                old_ema = self._ema_state[key]
                self._ema_state[key] = self._alpha * float(value) + (1 - self._alpha) * old_ema

    def reset_ema_state(self) -> None:
        """
        Reset EMA state to initial conditions.

        Useful for:
        - Testing/debugging
        - Switching between different operating modes
        - Recovery from anomalous sensor behavior
        """
        self._ema_state.clear()
        self._last_snapshot = None
        self._samples_processed = 0
        self._records_processed = 0
        self._logger.info("EMA state reset")

    def get_ema_state(self) -> dict[str, float]:
        """Get current EMA state (for debugging/monitoring)."""
        return self._ema_state.copy()

    def get_metrics(self) -> dict[str, int]:
        """Get processing metrics."""
        return {
            "records_processed": self._records_processed,
            "samples_processed": self._samples_processed,
            "ema_state_size": len(self._ema_state),
        }

    def _compute_window_stats(self, window: list[Any]) -> dict[str, dict[str, float]]:
        """
        Compute statistical features over the sensor window.

        Production Use: These features could feed into anomaly detection
        or be used as input to ML models.

        Returns:
            Dict mapping sensor_key -> {mean, std, min, max, range}
        """
        stats: dict[str, list[float]] = defaultdict(list)

        for sample in window:
            for key, value in sample.model_dump().items():
                if key != "timestamp_ns" and isinstance(value, (int, float)):
                    stats[key].append(float(value))

        result: dict[str, dict[str, float]] = {}
        for key, values in stats.items():
            if values:
                result[key] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                }

        return result
