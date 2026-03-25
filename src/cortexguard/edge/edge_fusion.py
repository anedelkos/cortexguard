from __future__ import annotations

import asyncio
import inspect
import logging
import statistics
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

try:
    import torch
    from PIL import Image
    from torchvision import models, transforms

    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

from opentelemetry import trace

from cortexguard.common.constants import DEFAULT_ALPHA
from cortexguard.edge.constants import (
    _IOU_OCCLUSION_THRESHOLD,
    _NEAR_CAMERA_THRESHOLD_M,
    _NEAR_THRESHOLD_M,
)
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot
from cortexguard.edge.models.reasoning_trace_entry import TraceSeverity
from cortexguard.edge.models.scene_graph import SceneGraph, SceneObject, SceneRelationship
from cortexguard.edge.online_learner_state_estimator import OnlineLearnerStateEstimator
from cortexguard.edge.utils.metrics import (
    component_duration_ms,
)
from cortexguard.edge.utils.tracing import BaseTraceSink, TraceSink
from cortexguard.simulation.models.windowed_fused_record import WindowedFusedRecord

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("cortexguard.edge_fusion")


class VisionEmbedder:
    """Lightweight ResNet-based embedder for images."""

    def __init__(self, device: str = "cpu"):
        if not _VISION_AVAILABLE:
            raise RuntimeError(
                "torch/torchvision/Pillow are not installed. "
                "Install the full ML dependencies with: uv sync --extra ml"
            )
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

    @(torch.no_grad() if _VISION_AVAILABLE else lambda f: f)  # type: ignore[misc]
    def embed(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0).to(self._device)
        emb = cast(torch.Tensor, self._model(x).squeeze())
        return emb.cpu()


def _mock_vision_inference(
    record: WindowedFusedRecord, image_embedding: Any
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """
    Minimal deterministic vision stub.

    Returns any vision_objects pre-populated on the record (e.g. injected by
    ChaosEngine and serialised over HTTP).  Falls back to empty list.
    """
    objects: list[dict[str, Any]] = [dict(v) for v in getattr(record, "vision_objects", []) or []]
    occlusion: dict[str, Any] | None = getattr(record, "vision_occlusion", None)
    return objects, occlusion


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

    Design Choice: EMA over Kalman
    - EMA: Simple, low compute, no model required
    - Kalman: More accurate but needs process/measurement models
    - For edge deployment, EMA sufficient given stable sensors
    """

    _SMOKE_PPM_THRESHOLD = 50.0  # threshold in ppm
    _VISUAL_OPACITY_THRESHOLD = 0.5  # 0.0-1.0 opacity threshold for vision smoke
    _SMOKE_SET_CONSECUTIVE = 2  # require 2 consecutive windows to set
    _SMOKE_CLEAR_CONSECUTIVE = 2  # require 2 consecutive windows to clear
    _SMOKE_MAX_SCORE = 5  # Tune based on window frequency
    _DRIFT_FAIL_MM = 10.0
    _FORCE_MIN_PCT = 0.0
    _FORCE_DROP_PCT = 100.1
    _EXPECTED_PERIOD_MS = 50
    _SOFT_DEGRADE_MS = 200
    _MAX_GAP_MS = 500

    def __init__(
        self,
        blackboard: Blackboard,
        trace_sink: BaseTraceSink | None = None,
        state_estimator: OnlineLearnerStateEstimator | None = None,
        alpha: float = DEFAULT_ALPHA,
        custom_logger: logging.Logger | None = None,
        embedder: VisionEmbedder | None = None,
        vision_inference: Any | None = None,
        force_min_n: float | None = None,
        force_drop_pct: float | None = None,
        drift_fail_mm: float | None = None,
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

        if force_min_n is not None:
            self._FORCE_MIN_PCT = force_min_n
        if force_drop_pct is not None:
            self._FORCE_DROP_PCT = force_drop_pct
        if drift_fail_mm is not None:
            self._DRIFT_FAIL_MM = drift_fail_mm

        # EMA state: sensor_key -> smoothed_value
        self._ema_state: dict[str, float] = {}
        self._last_snapshot: FusionSnapshot | None = None
        self._last_processed_timestamp_ns: int = 0

        # smoke fusion state (hysteresis)
        self._smoke_state: bool = False
        self._smoke_consec_set: int = 0
        self._smoke_consec_clear: int = 0

        # Metrics (for observability)
        self._samples_processed = 0
        self._records_processed = 0
        self._last_arrival_ns: int | None = None

        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        self._smoke_score = 0

    def close(self) -> None:
        """
        Shutdown the executor and cleanup resources.

        Should be called when EdgeFusion is no longer needed to prevent
        resource leaks. Can be called multiple times safely.

        Raises:
            None - suppresses exceptions to ensure cleanup
        """
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True)
            except Exception as e:
                self._logger.warning(f"Error shutting down executor: {e}")

    async def __aenter__(self) -> EdgeFusion:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Async context manager exit — shuts down the executor."""
        self.close()

    async def _detect_smoke(
        self, record: WindowedFusedRecord, vision_objects: list[Any] | None
    ) -> bool:
        """Detect smoke using a bounded counter for robust hysteresis."""
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
                opacity = float((v.get("properties") or {}).get("opacity", 0.0))
                if opacity >= self._VISUAL_OPACITY_THRESHOLD:
                    visual_smoke = True
                    break

        raw_smoke_flag = (
            last_smoke_ppm is not None and last_smoke_ppm >= self._SMOKE_PPM_THRESHOLD
        ) or visual_smoke

        # Integration-based hysteresis (Leaky Bucket)
        async with self._lock:
            if raw_smoke_flag:
                self._smoke_score = min(self._SMOKE_MAX_SCORE, self._smoke_score + 1)
            else:
                self._smoke_score = max(0, self._smoke_score - 1)

            # High threshold to trigger, low threshold to clear
            if self._smoke_score >= self._SMOKE_SET_CONSECUTIVE:
                self._smoke_state = True
            elif self._smoke_score == 0:
                self._smoke_state = False

            return self._smoke_state

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
        if max_force is not None and last_force is not None and max_force > 0 and last_force >= 0:
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
        if object_drift_mm is not None and object_drift_mm >= self._DRIFT_FAIL_MM:
            misgrasp_candidate = True
        elif max_force is not None:
            if max_force < self._FORCE_MIN_PCT or (
                force_drop_pct is not None and force_drop_pct >= self._FORCE_DROP_PCT
            ):
                misgrasp_candidate = True

        # grasp_success is None when there is no force data — indeterminate, not a failure.
        # LogicalRuleDetector skips None values so no consecutive-failure count is incremented.
        grasp_success: bool | None = None
        if misgrasp_candidate:
            grasp_success = False
        elif max_force is not None:
            grasp_success = True

        derived: dict[str, Any] = {
            "max_force_in_window": float(max_force) if max_force is not None else 0.0,
            "last_force": float(last_force) if last_force is not None else 0.0,
            "object_drift_mm": float(object_drift_mm) if object_drift_mm is not None else 0.0,
            "misgrasp_candidate": bool(misgrasp_candidate),
        }
        if grasp_success is not None:
            derived["grasp_success"] = grasp_success
        return derived

    async def _compute_comm_lag_and_tag(self, record: WindowedFusedRecord) -> tuple[int, bool]:
        """Monotonic communication lag calculation."""
        arrival_ns = getattr(record, "arrival_time_ns", None)
        async with self._lock:
            if arrival_ns is None or self._last_arrival_ns is None:
                self._last_arrival_ns = arrival_ns
                return 0, False

            # Ensure we handle potential system clock jumps
            arrival_gap_ns = max(0, arrival_ns - self._last_arrival_ns)

            lag_ms = arrival_gap_ns // 1_000_000
            timing_degraded = lag_ms > self._SOFT_DEGRADE_MS

            self._last_arrival_ns = arrival_ns
            return int(lag_ms), bool(timing_degraded)

    async def _update_ema(self, record: WindowedFusedRecord) -> dict[str, float] | None:
        """Returns a copy of the updated EMA state."""
        async with self._lock:
            if self._last_processed_timestamp_ns > record.timestamp_ns:
                self._logger.warning(
                    f"Dropping out-of-order record: {record.timestamp_ns} "
                    f"(last processed: {self._last_processed_timestamp_ns})"
                )
                return None

            self._last_processed_timestamp_ns = record.timestamp_ns

            with tracer.start_as_current_span("fusion.ema_update"):
                for sample in record.sensor_window:
                    self._update_ema_state(sample)
                    self._samples_processed += 1

            self._records_processed += 1
            return self._ema_state.copy()

    async def _compute_image_embedding(self, record: WindowedFusedRecord) -> tuple[Any, Any]:
        if not self.embedder or not getattr(record, "rgb_path", None):
            return None, None

        embedder = self.embedder

        with tracer.start_as_current_span("fusion.image_embedding"):
            try:
                loop = asyncio.get_running_loop()

                def load_and_embed() -> Any:
                    img = Image.open(record.rgb_path).convert("RGB")  # type: ignore[union-attr]
                    return embedder.embed(img)

                image_embedding = await loop.run_in_executor(self._executor, load_and_embed)
            except Exception as e:
                self._logger.warning(f"Failed to embed image {record.rgb_path}: {e}")
                return None, None

        if _VISION_AVAILABLE and torch.is_tensor(image_embedding):  # type: ignore[union-attr]
            return image_embedding, image_embedding.detach().cpu().numpy().tolist()
        return image_embedding, image_embedding

    async def _run_vision_inference(
        self, record: WindowedFusedRecord, embedding: Any
    ) -> tuple[list[Any], Any]:
        with tracer.start_as_current_span("fusion.vision_inference"):
            if inspect.iscoroutinefunction(self.vision_inference):
                result = await self.vision_inference(record, embedding)
                return cast(tuple[list[Any], Any], result)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: self.vision_inference(record, embedding),
                )
                return cast(tuple[list[Any], Any], result)

    async def _build_and_publish_scene_graph(
        self,
        record: WindowedFusedRecord,
        vision_objects: list[Any] | None,
        ema_snapshot: dict[str, float],
    ) -> Any:
        with tracer.start_as_current_span("fusion.scene_graph"):
            ts = datetime.fromtimestamp(record.timestamp_ns / 1e9, tz=ZoneInfo("UTC"))
            scene_graph = _build_scene_graph_from_vision(ts, vision_objects or [], ema_snapshot)
            try:
                await self._blackboard.set_scene_graph(scene_graph)
            except AttributeError:
                logger.debug("Blackboard.set_scene_graph not available; skipping publish")
            return scene_graph

    async def _compute_derived_features(
        self,
        record: WindowedFusedRecord,
        vision_objects: list[Any] | None,
        ema_snapshot: dict[str, float],
    ) -> tuple[float | None, bool, dict[str, Any]]:
        with tracer.start_as_current_span("fusion.derived_features"):
            last_temp = next(
                (
                    r.temp_c
                    for r in reversed(record.sensor_window)
                    if getattr(r, "temp_c", None) is not None
                ),
                None,
            )

            smoke_detected = await self._detect_smoke(record, vision_objects)

            derived = ema_snapshot.copy()
            derived.update(self._compute_misgrasp_derived(record, vision_objects))

            lag_ms, timing_degraded = await self._compute_comm_lag_and_tag(record)
            derived["comm_lag_ms"] = lag_ms
            derived["timing_degraded"] = timing_degraded

            if timing_degraded:
                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="COMM_TIMING_DEGRADATION",
                    reasoning_text=f"Degraded sensor data arrival by {lag_ms}, expected period: {self._EXPECTED_PERIOD_MS}",
                    severity=TraceSeverity.HIGH,
                )

            return last_temp, smoke_detected, derived

    async def _update_state_estimator(self, snapshot: FusionSnapshot) -> None:
        if not self._state_estimator:
            return
        with tracer.start_as_current_span("fusion.state_estimator"):
            try:
                estimate = await self._state_estimator.update(snapshot)
                await self._blackboard.update_state_estimate(estimate)
            except Exception as e:
                self._logger.exception(f"StateEstimator failed: {e}")

    async def process_record(self, record: WindowedFusedRecord) -> FusionSnapshot | None:
        """
        Process a windowed sensor record and update fusion state.

        Strategy: Applies EMA to each sample in window sequentially.
        This gives temporal ordering within the window - later samples
        influence the EMA more than earlier ones.

        Args:
            record: Windowed fused sensor data

        Returns:
            FusionSnapshot with smoothed sensor values and derived features
        """
        start = time.perf_counter()

        with tracer.start_as_current_span("fusion.process_record") as span:
            span.set_attribute("window.size", len(record.sensor_window))
            span.set_attribute("has_rgb", bool(getattr(record, "rgb_path", None)))

            # 1. EMA update
            ema_snapshot = await self._update_ema(record)
            if ema_snapshot is None:
                # Out-of-order record, skip processing
                return None

            # 2. Image embedding
            image_embedding, serialized_embedding = await self._compute_image_embedding(record)

            # 3. Vision inference
            vision_objects, vision_occlusion = await self._run_vision_inference(
                record, image_embedding
            )

            # 4. Scene graph
            scene_graph = await self._build_and_publish_scene_graph(
                record, vision_objects, ema_snapshot
            )

            # 5. Derived features
            last_temp, smoke_detected, derived = await self._compute_derived_features(
                record, vision_objects, ema_snapshot
            )

            # 6. Build snapshot
            snapshot = FusionSnapshot(
                id=uuid.uuid4().hex,
                timestamp=datetime.fromtimestamp(record.timestamp_ns / 1e9),
                sensors={
                    "temp_celsius": last_temp,
                    "smoke_detected": smoke_detected,
                    "raw": record.sensor_window,
                    "ema_smoothed": ema_snapshot,
                    "window_stats": self._compute_window_stats(record.sensor_window),
                    "image_embedding": serialized_embedding,
                    "vision_objects": vision_objects,
                    "vision_occlusion": vision_occlusion,
                    "scene_graph_summary": [
                        {"id": o.id, "label": o.label} for o in scene_graph.objects
                    ],
                },
                derived=derived,
            )

            async with self._lock:
                self._last_snapshot = snapshot

            await self._blackboard.update_fusion_snapshot(snapshot)

            # 7. State estimator
            await self._update_state_estimator(snapshot)

            duration_ms = (time.perf_counter() - start) * 1000.0
            component_duration_ms.labels(component="fusion_process_record").observe(duration_ms)

            return snapshot

    def _update_ema_state(self, sample: Any) -> None:
        """Update EMA state with a single sensor sample."""
        for key, value in sample.model_dump().items():
            if key == "timestamp_ns" or value is None:
                continue

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self._logger.debug(f"Skipping non-numeric field: {key}={type(value).__name__}")
                continue

            if key not in self._ema_state:
                self._ema_state[key] = float(value)
            else:
                old_ema = self._ema_state[key]
                new_ema = self._alpha * float(value) + (1 - self._alpha) * old_ema
                self._ema_state[key] = new_ema

    async def reset_ema_state(self) -> None:
        """Reset EMA state to initial conditions."""
        async with self._lock:
            self._ema_state.clear()
            self._last_snapshot = None
            self._samples_processed = 0
            self._records_processed = 0
        self._logger.info("EMA state reset")

    async def get_ema_state(self) -> dict[str, float]:
        """Get current EMA state (for debugging/monitoring)."""
        async with self._lock:
            return self._ema_state.copy()

    async def get_metrics(self) -> dict[str, int]:
        """Get processing metrics."""
        async with self._lock:
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
                if (
                    key != "timestamp_ns"
                    and not isinstance(value, bool)
                    and isinstance(value, (int, float))
                ):
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
