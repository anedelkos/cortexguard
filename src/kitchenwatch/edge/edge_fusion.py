from __future__ import annotations

import logging
import statistics
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import torch
from PIL import Image
from torchvision import models, transforms

from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.reasoning_trace_entry import TraceSeverity
from kitchenwatch.edge.models.scene_graph import SceneGraph, SceneObject, SceneRelationship
from kitchenwatch.edge.online_learner_state_estimator import OnlineLearnerStateEstimator
from kitchenwatch.edge.utils.tracing import BaseTraceSink, TraceSink
from kitchenwatch.simulation.models.windowed_fused_record import WindowedFusedRecord

# Default smoothing factor for EMA (0 < alpha <= 1)
# Lower alpha = more smoothing, higher alpha = more responsive
DEFAULT_ALPHA = 0.1

logger = logging.getLogger(__name__)

# Configurable heuristics
_IOU_OCCLUSION_THRESHOLD = 0.1
_NEAR_THRESHOLD_M = 0.5
_NEAR_CAMERA_THRESHOLD_M = 0.5  # optional: treat objects near camera as near


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
    Minimal deterministic vision stub.

    Behavior:
    - All anomalies must be injected explicitly via ChaosEngine or tests.
    """
    return [], None


def _to_scene_object(v: dict[str, Any], ema_state: dict[str, float]) -> SceneObject:
    obj_id = v.get("bbox_id") or v.get("id") or uuid.uuid4().hex
    props: dict[str, Any] = {}

    dm = v.get("distance_m")
    if dm is not None:
        try:
            props["distance_m"] = float(dm)
        except Exception as exc:
            logger.debug("Invalid distance_m for vision object %r: %s", v, exc)

    # confidence always present as float
    try:
        props["confidence"] = float(v.get("confidence", 0.0))
    except Exception:
        props["confidence"] = 0.0

    # EMA key: prefer label_distance but tolerate common variants
    label = (v.get("label") or "unknown").lower()
    ema_key = f"{label}_distance"
    if ema_key in ema_state:
        props["distance_ema"] = ema_state[ema_key]

    return SceneObject(
        id=str(obj_id),
        label=str(label),
        location_2d=v.get("bbox"),
        pose_3d=None,
        properties=props,
    )


def _iou(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    return (inter_area / union) if union > 0 else 0.0


def _is_near(a: SceneObject, b: SceneObject, threshold_m: float = _NEAR_THRESHOLD_M) -> bool:
    da = a.properties.get("distance_m")
    db = b.properties.get("distance_m")
    if isinstance(da, (int, float)) and isinstance(db, (int, float)):
        # symmetric proximity: both distances within threshold of each other
        if abs(da - db) <= threshold_m:
            return True
        # also consider either object very close to camera as near
        if min(da, db) <= _NEAR_CAMERA_THRESHOLD_M:
            return True
        return False

    bbox_a = a.location_2d
    bbox_b = b.location_2d
    if bbox_a and bbox_b and len(bbox_a) == 4 and len(bbox_b) == 4:
        return _iou(bbox_a, bbox_b) > _IOU_OCCLUSION_THRESHOLD

    return False


def _build_scene_graph_from_vision(
    timestamp: datetime,
    vision_objects: Iterable[dict[str, Any]],
    ema_state: dict[str, float],
) -> SceneGraph:
    objs: list[SceneObject] = []
    rels_set: set[tuple[str, str, str]] = set()  # (source, relationship, target)

    for v in vision_objects:
        try:
            objs.append(_to_scene_object(v, ema_state))
        except Exception as exc:
            logger.debug("Skipping malformed vision object: %r (%s)", v, exc)

    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i == j:
                continue
            if _is_near(a, b):
                rels_set.add((a.id, "near", b.id))

            # occlusion: lower-confidence object occluding a higher-confidence one
            conf_a = float(a.properties.get("confidence", 0.0))
            conf_b = float(b.properties.get("confidence", 0.0))
            if conf_a < 0.5 and conf_b > 0.7 and a.location_2d and b.location_2d:
                # require IoU to be non-zero to consider occlusion
                if _iou(a.location_2d, b.location_2d) > 0.0:
                    rels_set.add((a.id, "occluding", b.id))

    rels = [SceneRelationship(source_id=s, relationship=r, target_id=t) for (s, r, t) in rels_set]
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

    SMOKE_PPM_THRESHOLD = 50.0  # threshold in ppm
    VISUAL_OPACITY_THRESHOLD = 0.5  # 0.0-1.0 opacity threshold for vision smoke
    SMOKE_SET_CONSECUTIVE = 2  # require 2 consecutive windows to set
    SMOKE_CLEAR_CONSECUTIVE = 2  # require 2 consecutive windows to clear
    DRIFT_FAIL_MM = 10.0
    FORCE_MIN_PCT = 20.0
    FORCE_DROP_PCT = 30.0
    EXPECTED_PERIOD_MS = 50
    SOFT_DEGRADE_MS = 200
    MAX_GAP_MS = 500

    def __init__(
        self,
        blackboard: Blackboard,
        trace_sink: BaseTraceSink | None = None,
        state_estimator: OnlineLearnerStateEstimator | None = None,
        alpha: float = DEFAULT_ALPHA,
        custom_logger: logging.Logger | None = None,
        embedder: VisionEmbedder | None = None,
        vision_inference: Any | None = None,
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
        self._trace_sink: BaseTraceSink = (
            trace_sink if trace_sink is not None else TraceSink(blackboard=self._blackboard)
        )
        self._alpha = alpha
        self._logger = custom_logger or logger
        self.embedder = embedder
        self._state_estimator: OnlineLearnerStateEstimator | None = state_estimator
        self.vision_inference = vision_inference or _mock_vision_inference

        # EMA state: sensor_key -> smoothed_value
        self._ema_state: dict[str, float] = {}
        self._last_snapshot: FusionSnapshot | None = None

        # smoke fusion state (hysteresis)
        self._smoke_state: bool = False
        self._smoke_consec_set: int = 0
        self._smoke_consec_clear: int = 0

        # Metrics (for observability)
        self._samples_processed = 0
        self._records_processed = 0
        self._last_arrival_ns: int | None = None

    async def _detect_smoke(
        self, record: WindowedFusedRecord, vision_objects: list[dict[str, Any]]
    ) -> bool:
        """Detect smoke from smoke and camera sensors"""
        last_smoke_ppm = next(
            (r.smoke_ppm for r in reversed(record.sensor_window) if r.smoke_ppm is not None), None
        )
        visual_smoke = False
        if vision_objects:
            for v in vision_objects:
                label = (v.get("label") or "").lower()
                if label == "smoke":
                    visual_smoke = True
                    break
                # tolerate opacity in properties (0.0-1.0)
                props = v.get("properties") or {}
                try:
                    opacity = float(props.get("opacity", 0.0))
                except Exception:
                    opacity = 0.0
                if opacity >= self.VISUAL_OPACITY_THRESHOLD:
                    visual_smoke = True
                    break

        # 3) raw flag: either numeric sensor above threshold OR visual evidence
        raw_smoke_flag = (
            last_smoke_ppm is not None and last_smoke_ppm >= self.SMOKE_PPM_THRESHOLD
        ) or visual_smoke

        # 4) hysteresis: update consecutive counters and decide smoke_detected
        if raw_smoke_flag:
            self._smoke_consec_set += 1
            self._smoke_consec_clear = 0
        else:
            self._smoke_consec_clear += 1
            self._smoke_consec_set = 0

        if self._smoke_consec_set >= self.SMOKE_SET_CONSECUTIVE:
            smoke_detected = True
        elif self._smoke_consec_clear >= self.SMOKE_CLEAR_CONSECUTIVE:
            smoke_detected = False
        else:
            smoke_detected = self._smoke_state

        return smoke_detected

    def _compute_misgrasp_derived(
        self, record: WindowedFusedRecord, vision_objects: list[dict[str, Any]] | None
    ) -> dict[str, float | bool]:
        forces = []
        for r in record.sensor_window:
            v = getattr(r, "force_x", None)
            if v is not None:
                forces.append(float(v))

        max_force: float | None = max(forces) if forces else None

        # last_force
        last_force = None
        for r in reversed(record.sensor_window):
            v = getattr(r, "force_x", None)
            if v is not None:
                last_force = float(v)
                break

        force_drop_pct: float | None = None
        if max_force is not None and last_force is not None and max_force > 0:
            force_drop_pct = 100.0 * (max_force - last_force) / max_force

        object_drift_mm: float | None = next(
            (
                cast(dict[str, Any], v.get("properties", {})).get("drift_mm")
                for v in (vision_objects or [])
                if isinstance(v.get("id", ""), str) and v.get("id", "").startswith("chaos_slip_")
            ),
            None,
        )

        misgrasp_candidate = False
        if object_drift_mm is not None and object_drift_mm >= self.DRIFT_FAIL_MM:
            misgrasp_candidate = True
        elif max_force is not None:
            if max_force < self.FORCE_MIN_PCT or (
                force_drop_pct is not None and force_drop_pct >= self.FORCE_DROP_PCT
            ):
                misgrasp_candidate = True

        # Normalize grasp_success to a bool (no None in return type)
        if misgrasp_candidate:
            grasp_success = False
        elif max_force is not None:
            grasp_success = True
        else:
            grasp_success = False

        return {
            "max_force_in_window": float(max_force) if max_force is not None else 0.0,
            "last_force": float(last_force) if last_force is not None else 0.0,
            "force_drop_pct": float(force_drop_pct) if force_drop_pct is not None else 0.0,
            "object_drift_mm": float(object_drift_mm) if object_drift_mm is not None else 0.0,
            "misgrasp_candidate": bool(misgrasp_candidate),
            "grasp_success": bool(grasp_success),
        }

    def _compute_comm_lag_and_tag(self, record: WindowedFusedRecord) -> tuple[int, bool]:
        """
        Compute comm lag from record.arrival_time_ns.
        Returns (lag_ms, timing_degraded).
        """
        arrival_ns = getattr(record, "arrival_time_ns", None)
        if arrival_ns is None:
            return 0, False

        if self._last_arrival_ns is None:
            self._last_arrival_ns = arrival_ns
            return 0, False

        arrival_gap_ns = arrival_ns - self._last_arrival_ns
        if arrival_gap_ns < 0:
            arrival_gap_ns = 0

        lag_ms = arrival_gap_ns // 1_000_000
        timing_degraded = lag_ms > self.SOFT_DEGRADE_MS

        self._last_arrival_ns = arrival_ns
        return lag_ms, timing_degraded

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

        vision_objects, vision_occlusion = self.vision_inference(record, image_embedding)
        # build timezone-aware scene graph and publish to blackboard if available
        sg_timestamp = datetime.fromtimestamp(record.timestamp_ns / 1e9, tz=ZoneInfo("UTC"))
        scene_graph = _build_scene_graph_from_vision(
            sg_timestamp, vision_objects or [], self._ema_state
        )

        try:
            await self._blackboard.set_scene_graph(scene_graph)  # type: ignore[attr-defined]
        except AttributeError:
            logger.debug("Blackboard.set_scene_graph not available; skipping scene graph publish")

        last_temp = next(
            (
                r.temp_c
                for r in reversed(record.sensor_window)
                if getattr(r, "temp_c", None) is not None
            ),
            None,
        )
        smoke_detected = await self._detect_smoke(record, vision_objects)
        self._smoke_state = smoke_detected

        misgrasped_derived = self._compute_misgrasp_derived(record, vision_objects)
        derived = self._ema_state.copy()
        derived.update(misgrasped_derived)

        lag_ms, timing_degraded = self._compute_comm_lag_and_tag(record)
        derived["comm_lag_ms"] = lag_ms
        derived["timing_degraded"] = timing_degraded
        if timing_degraded:
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="COMM_TIMING_DEGRADATION",
                reasoning_text=f"Degraded sensor data arrival by {lag_ms}, expected period: {self.EXPECTED_PERIOD_MS}",
                severity=TraceSeverity.HIGH,
            )

        # Create snapshot from current EMA state
        snapshot = FusionSnapshot(
            id=uuid.uuid4().hex,
            timestamp=datetime.fromtimestamp(record.timestamp_ns / 1e9),
            sensors={
                "temp_celsius": last_temp,
                "smoke_detected": smoke_detected,
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
            derived=derived,  # Smoothed values as derived features
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
                new_ema = self._alpha * float(value) + (1 - self._alpha) * old_ema
                self._ema_state[key] = new_ema

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
