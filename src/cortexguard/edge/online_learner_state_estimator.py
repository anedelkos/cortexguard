import datetime
import logging
import time
from collections import deque
from statistics import mean, stdev

from opentelemetry import trace

from cortexguard.core.interfaces.base_online_learner import BaseOnlineLearner
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot
from cortexguard.edge.models.state_estimate import StateEstimate
from cortexguard.edge.utils.metrics import (
    component_duration_ms,
    estimator_confidence,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("cortexguard.online_learner_state_estimator")

# Anomaly detection thresholds
_SIGMA_THRESHOLD_NOMINAL = 3.0  # Standard deviations for anomaly detection
_SIGMA_THRESHOLD_IMPULSE = 6.0  # High threshold for impulse events (2x nominal)
_MIN_HISTORY_SAMPLES = 10  # Minimum samples needed for statistical significance
_EPSILON = 1e-6  # Small value to avoid division by zero
_MIN_SIGMA = (
    1e-3  # Practical sigma floor to avoid tiny-variance amplification (tune to sensor noise)
)


class OnlineLearnerStateEstimator:
    """
    State estimator that translates statistical deviations into system states.

    Architecture Philosophy:
    This component is intentionally unit-agnostic and domain-agnostic.
    It operates purely on statistical signal properties (Z-scores) rather than
    domain-specific thresholds. This makes it robust to:
    - Unknown datasets (RobotFeeding vs others)
    - Mixed units (forces, positions, temperatures)
    - Sensor drift and calibration issues

    Design Pattern: Statistical Process Control (SPC)
    - Uses sliding window for online statistics
    - Z-score normalization for scale-invariant anomaly detection
    - Confidence decreases with deviation from expected behavior

    Production Considerations (not implemented):
    - Multi-scale anomaly detection (multiple window sizes)
    - Adaptive thresholds based on signal characteristics
    - Contextual anomaly detection (different thresholds per intent)
    - Time-series forecasting for predictive anomalies
    """

    def __init__(
        self,
        learner: BaseOnlineLearner,
        blackboard: Blackboard,
        window_size: int = 50,
        sigma_threshold: float = _SIGMA_THRESHOLD_NOMINAL,
        min_history: int = _MIN_HISTORY_SAMPLES,
    ) -> None:
        """
        Initialize state estimator.

        Args:
            learner: Online learning model for predictions
            window_size: Number of recent residuals to retain per feature
            sigma_threshold: Z-score threshold for anomaly detection
            min_history: Minimum samples needed before computing statistics
        """
        self._learner = learner
        self._blackboard = blackboard
        self._residuals: dict[str, deque[float]] = {}
        self._window_size = window_size

        # Sensitivity Configuration
        self._sigma_threshold = sigma_threshold
        self._min_history = min_history

        # Metrics for observability
        self._update_count = 0
        self._anomaly_count = 0

        # Per-feature observed counts (counts only numeric, non-None observations)
        self._seen_per_feature: dict[str, int] = {}

        # Persistence counters (consecutive windows above threshold)
        self._consec_high: dict[str, int] = {}

    def _z_to_symbol(self, z: float) -> str:
        """
        Classifies a Z-score into a human-readable symbolic state (Layer 2).

        This aligns with the proposed Google-style architecture, providing the LLM
        with categorical context.
        """
        if z < -3.0:
            return "critical_low"
        if z < -1.5:
            return "low"
        if z < 1.5:
            return "nominal"
        if z < 3.0:
            return "high"
        return "critical_high"

    async def _fetch_current_intent(self) -> str:
        """Fetches the current intent from the Blackboard."""
        try:
            current_step = await self._blackboard.get_current_step()

            return current_step.description if current_step else "unknown"
        except Exception as e:
            logger.exception(f"Failed to retrieve intent from Blackboard: {e}")
            return "exception"

    def _set_timestamp_attribute(
        self, span: trace.Span, ts: datetime.datetime | float | int
    ) -> None:
        if isinstance(ts, datetime.datetime):
            span.set_attribute("timestamp", ts.isoformat())
        else:
            span.set_attribute("timestamp", float(ts))

    async def _augment_with_scene_graph(
        self, features: dict[str, float]
    ) -> datetime.datetime | None:
        try:
            sg = await self._blackboard.get_scene_graph()
        except Exception as exc:
            logger.debug("Failed to get scene graph from blackboard: %s", exc)
            return None

        if sg is None:
            logger.debug("No SceneGraph available; skipping vision-derived features")
            return None

        objs = getattr(sg, "objects", None)
        rels = getattr(sg, "relationships", None)

        has_objects = isinstance(objs, (list, tuple)) and len(objs) > 0
        has_relationships = isinstance(rels, (list, tuple)) and len(rels) > 0

        if not (has_objects or has_relationships):
            logger.debug("No populated SceneGraph available; skipping vision-derived features")
            return None

        scene_graph_frame: datetime.datetime | None = None
        sg_ts = getattr(sg, "timestamp", None)
        if isinstance(sg_ts, datetime.datetime):
            scene_graph_frame = sg_ts

        # Extract nearest human/hand distance (if any)
        nearest_human: float | None = None
        for o in objs or []:
            label = (o.label or "").lower()
            if label in ("person", "human", "hand"):
                d = o.properties.get("distance_m")
                if isinstance(d, (int, float)):
                    nearest_human = d if nearest_human is None else min(nearest_human, float(d))
        if nearest_human is not None:
            features["vision_nearest_human_m"] = float(nearest_human)

        # Count occluding relationships as a simple vision-derived feature
        occlusion_count = sum(
            1 for r in rels or [] if getattr(r, "relationship", None) == "occluding"
        )
        features["vision_occlusion_count"] = float(occlusion_count)

        return scene_graph_frame

    def _compute_residuals_and_stats(self, features: dict[str, float]) -> tuple[
        dict[str, float],  # residuals
        dict[str, float],  # uncertainty
        dict[str, float],  # z_scores
        float,  # max_z_score
        str | None,  # max_z_feature
        dict[str, float],  # features_for_update
    ]:
        residuals = {}
        uncertainty = {}
        z_scores = {}
        max_z_score = 0.0
        max_z_feature = None
        features_for_update = {}

        with tracer.start_as_current_span("estimator.predict_and_residuals"):
            expected = self._learner.predict(features)

            for key, observed in features.items():
                if observed is None:
                    continue

                self._seen_per_feature[key] = self._seen_per_feature.get(key, 0) + 1
                features_for_update[key] = observed

                exp = expected.get(key, observed)
                residual = observed - exp
                residuals[key] = residual

                # history
                if key not in self._residuals:
                    self._residuals[key] = deque(maxlen=self._window_size)
                self._residuals[key].append(residual)

                # not enough history
                if self._seen_per_feature[key] < self._min_history:
                    uncertainty[key] = 0.0
                    z_scores[key] = 0.0
                    continue

                hist = self._residuals[key]
                sigma_raw = stdev(hist) if len(hist) > 1 else 0.0
                mu = mean(hist) if hist else 0.0
                uncertainty[key] = sigma_raw

                sigma_used = max(sigma_raw, _MIN_SIGMA)
                z = abs(residual - mu) / sigma_used
                z_scores[key] = z

                # persistence
                if z > self._sigma_threshold:
                    self._consec_high[key] = self._consec_high.get(key, 0) + 1
                else:
                    self._consec_high[key] = 0

                if self._consec_high[key] < 2:
                    continue

                if z > max_z_score:
                    max_z_score = z
                    max_z_feature = key

        return (
            residuals,
            uncertainty,
            z_scores,
            max_z_score,
            max_z_feature,
            features_for_update,
        )

    def _safe_update_learner(self, features_for_update: dict[str, float]) -> None:
        try:
            if features_for_update:
                self._learner.update(features_for_update)
        except Exception as e:
            logger.exception("Learner update failed: %s", e)
        self._update_count += 1

    def _compute_symbolic_state(self, z_scores: dict[str, float]) -> dict[str, str]:
        symbolic = {}
        for key, z in z_scores.items():
            if z != 0.0:
                symbolic[f"{key}_state"] = self._z_to_symbol(z)
        return symbolic

    def _attach_scene_graph_frame(
        self, flags: dict[str, float], frame: datetime.datetime | None
    ) -> None:
        if frame is None:
            return
        try:
            flags["scene_graph_frame"] = float(frame.timestamp())
        except Exception:
            logger.debug("Could not convert scene_graph_frame to timestamp: %r", frame)

    def _update_metrics(self, label: str) -> None:
        if label != "nominal":
            self._anomaly_count += 1

    def _normalize_timestamp(self, ts: datetime.datetime | float | int) -> datetime.datetime:
        if isinstance(ts, datetime.datetime):
            return ts
        # assume float or int epoch seconds
        return datetime.datetime.fromtimestamp(float(ts))

    def _build_state_estimate(
        self,
        timestamp: datetime.datetime | float | int,
        label: str,
        confidence: float,
        residuals: dict[str, float],
        uncertainty: dict[str, float],
        z_scores: dict[str, float],
        flags: dict[str, float],
        intent: str | None,
        symbolic_state: dict[str, str],
    ) -> StateEstimate:

        ts = self._normalize_timestamp(timestamp)

        return StateEstimate(
            timestamp=ts,
            label=label,
            confidence=confidence,
            residuals=residuals,
            uncertainty=uncertainty,
            z_scores=z_scores,
            ttd=None,
            ttf=None,
            flags=flags,
            source_intent=intent,
            symbolic_system_state=symbolic_state,
        )

    async def update(self, snapshot: FusionSnapshot) -> StateEstimate:
        """
        Process a fusion snapshot and estimate system state.

        Pipeline:
        1. Extract features from snapshot
        2. Predict expected values using online learner
        3. Calculate residuals (observed - expected)
        4. Compute Z-scores for scale-invariant anomaly detection
        5. Update learner with new observations
        6. Generate state estimate with confidence

        Args:
            snapshot: Fused sensor snapshot with derived features

        Returns:
            StateEstimate with label, confidence, and diagnostics
        """
        start = time.perf_counter()

        with tracer.start_as_current_span("estimator.update") as span:
            self._set_timestamp_attribute(span, snapshot.timestamp)
            span.set_attribute("feature.count", len(snapshot.derived or {}))

            now = snapshot.timestamp
            current_intent = await self._fetch_current_intent()

            # 1. Extract base features
            features = snapshot.derived.copy()

            # 2. Augment with scene graph
            scene_graph_frame = await self._augment_with_scene_graph(features)

            if not features:
                logger.warning("Empty features in snapshot, returning nominal state")
                return self._create_nominal_state(now, current_intent)

            # 3. Predict + residuals + stats
            (
                residuals,
                uncertainty,
                z_scores,
                max_z_score,
                max_z_feature,
                features_for_update,
            ) = self._compute_residuals_and_stats(features)

            # 4. Update learner
            self._safe_update_learner(features_for_update)

            # 5. Classify state
            label, flags = self._classify_state(max_z_score, max_z_feature, z_scores)
            span.set_attribute("max_z_score", max_z_score)
            span.set_attribute("label", label)

            # 6. Symbolic state
            symbolic_state = self._compute_symbolic_state(z_scores)

            # 7. Confidence
            confidence = self._calculate_confidence(max_z_score)
            estimator_confidence.set(confidence)

            # 8. Attach scene graph frame
            self._attach_scene_graph_frame(flags, scene_graph_frame)

            # 9. Update metrics
            self._update_metrics(label)

            duration_ms = (time.perf_counter() - start) * 1000.0
            component_duration_ms.labels(component="estimator_update").observe(duration_ms)

            # 10. Build final state estimate
            return self._build_state_estimate(
                now,
                label,
                confidence,
                residuals,
                uncertainty,
                z_scores,
                flags,
                current_intent,
                symbolic_state,
            )

    def _classify_state(
        self,
        max_z_score: float,
        max_z_feature: str | None,
        z_scores: dict[str, float],
    ) -> tuple[str, dict[str, float]]:
        """
        Classify system state based on Z-score magnitude.

        Classification Scheme:
        - nominal: All signals within 3 sigma (99.7% confidence)
        - transient_disturbance: Peak signal 3-6 sigma (rare but possible)
        - impulse_event: Peak signal >6 sigma (extremely rare, likely real event)

        Args:
            max_z_score: Maximum Z-score across all features
            max_z_feature: Feature name with maximum Z-score
            z_scores: Dictionary of Z-scores per feature

        Returns:
            Tuple of (state_label, diagnostic_flags)
        """
        flags: dict[str, float] = {
            "max_z": max_z_score,
        }

        if max_z_feature:
            flags["max_z_feature"] = max_z_feature  # type: ignore[assignment]

        # Statistical Process Control thresholds
        if max_z_score > _SIGMA_THRESHOLD_IMPULSE:
            # >6 sigma: Extremely rare (0.0001% probability)
            # Likely a real physical event, not noise
            return "impulse_event", flags
        elif max_z_score > self._sigma_threshold:
            # 3-6 sigma: Rare (0.3-0.0001% probability)
            # Could be noise or minor disturbance
            return "transient_disturbance", flags
        else:
            # <3 sigma: Normal process variation (99.7% probability)
            return "nominal", flags

    def _calculate_confidence(self, max_z_score: float) -> float:
        """
        Calculate confidence as inverse of normalized deviation.

        Confidence represents how well the observations match predictions.
        - confidence = 1.0: Perfect match (z=0)
        - confidence = 0.5: Halfway to anomaly threshold
        - confidence = 0.0: At or beyond anomaly threshold

        Args:
            max_z_score: Maximum Z-score across all features

        Returns:
            Confidence value in range [0.0, 1.0]
        """
        # Clamp to [0, 1] range
        normalized_z = max_z_score / self._sigma_threshold
        confidence = max(0.0, min(1.0, 1.0 - normalized_z))
        return confidence

    def _create_nominal_state(
        self, timestamp: datetime.datetime, intent: str | None
    ) -> StateEstimate:
        """Create a nominal state estimate when no features available."""
        return StateEstimate(
            timestamp=timestamp,
            label="nominal",
            confidence=1.0,
            residuals={},
            uncertainty={},
            ttd=None,
            ttf=None,
            flags={"max_z": 0.0, "empty_features": True},
            source_intent=intent,
            symbolic_system_state={},
        )

    def get_metrics(self) -> dict[str, int | float]:
        """
        Get estimator metrics for monitoring.

        Returns:
            Dictionary of diagnostic metrics
        """
        return {
            "update_count": self._update_count,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": (
                self._anomaly_count / self._update_count if self._update_count > 0 else 0.0
            ),
            "features_tracked": len(self._residuals),
        }
