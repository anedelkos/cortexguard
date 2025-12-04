import datetime
import logging
from collections import deque
from statistics import stdev

from kitchenwatch.core.interfaces.base_online_learner import BaseOnlineLearner
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.state_estimate import StateEstimate

logger = logging.getLogger(__name__)

# Anomaly detection thresholds
SIGMA_THRESHOLD_NOMINAL = 3.0  # Standard deviations for anomaly detection
SIGMA_THRESHOLD_IMPULSE = 6.0  # High threshold for impulse events (2x nominal)
MIN_HISTORY_SAMPLES = 10  # Minimum samples needed for statistical significance
EPSILON = 1e-6  # Small value to avoid division by zero


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
        sigma_threshold: float = SIGMA_THRESHOLD_NOMINAL,
        min_history: int = MIN_HISTORY_SAMPLES,
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
        now = snapshot.timestamp
        current_intent = await self._fetch_current_intent()

        # Use EMA-smoothed features to reduce noise
        features = snapshot.derived.copy()

        scene_graph_frame: datetime.datetime | None = None
        try:
            sg = await self._blackboard.get_scene_graph()
        except Exception as exc:
            logger.debug("Failed to get scene graph from blackboard: %s", exc)
            sg = None

        objs = getattr(sg, "objects", None)
        rels = getattr(sg, "relationships", None)

        has_objects = isinstance(objs, (list, tuple)) and len(objs) > 0
        has_relationships = isinstance(rels, (list, tuple)) and len(rels) > 0

        if sg is not None and (has_objects or has_relationships):
            # record a small reference to the scene graph timestamp/frame if it's a datetime
            sg_ts = getattr(sg, "timestamp", None)
            if isinstance(sg_ts, datetime.datetime):
                scene_graph_frame = sg_ts

            # Extract nearest human/hand distance (if any)
            nearest_human: float | None = None
            for o in sg.objects:
                label = (o.label or "").lower()
                if label in ("person", "human", "hand"):
                    d = o.properties.get("distance_m")
                    if isinstance(d, (int, float)):
                        nearest_human = d if nearest_human is None else min(nearest_human, float(d))
            if nearest_human is not None:
                features["vision_nearest_human_m"] = float(nearest_human)

            # Count occluding relationships as a simple vision-derived feature
            occlusion_count = sum(
                1 for r in sg.relationships if getattr(r, "relationship", None) == "occluding"
            )
            features["vision_occlusion_count"] = float(occlusion_count)
        else:
            # No usable scene graph found; do not add vision-derived features.
            logger.debug("No populated SceneGraph available; skipping vision-derived features")

        if not features:
            logger.warning("Empty features in snapshot, returning nominal state")
            return self._create_nominal_state(now, current_intent)

        residuals: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        z_scores: dict[str, float] = {}

        # 1. Predict expected values
        expected = self._learner.predict(features)

        # 2. Calculate residuals and update history
        max_z_score = 0.0
        max_z_feature: str | None = None

        for key, observed_value in features.items():
            expected_value = expected.get(key, observed_value)
            residual = observed_value - expected_value
            residuals[key] = residual

            # Initialize residual history for new features
            if key not in self._residuals:
                self._residuals[key] = deque(maxlen=self._window_size)
            self._residuals[key].append(residual)

            # 3. Calculate statistical metrics
            residual_history = self._residuals[key]

            if len(residual_history) >= self._min_history:
                # Calculate standard deviation of residuals
                # Low sigma = high confidence, high sigma = high uncertainty
                sigma = stdev(residual_history) if len(residual_history) > 1 else 0.0
                uncertainty[key] = sigma

                # Calculate Z-score (normalized deviation)
                if sigma > EPSILON:
                    z = abs(residual) / sigma
                    z_scores[key] = z

                    if z > max_z_score:
                        max_z_score = z
                        max_z_feature = key
                else:
                    # Zero variance - model is perfect or not enough data
                    z_scores[key] = 0.0
            else:
                # Not enough history for reliable statistics
                uncertainty[key] = 0.0
                z_scores[key] = 0.0

        # 4. Update learner with new observations
        self._learner.update(features)
        self._update_count += 1

        # 5. Classify state based on statistical significance
        label, flags = self._classify_state(
            max_z_score=max_z_score,
            max_z_feature=max_z_feature,
            z_scores=z_scores,
        )

        # 6. Calculate Symbolic State
        symbolic_system_state: dict[str, str] = {}
        for key, z_score in z_scores.items():
            # Only track features with enough history
            if z_score != 0.0:
                symbolic_system_state[f"{key}_state"] = self._z_to_symbol(z_score)

        # Track anomalies
        if label != "nominal":
            self._anomaly_count += 1

        # 7. Calculate confidence (inverse of normalized Z-score)
        # z=0 → confidence=1.0 (perfect match)
        # z=threshold → confidence=0.0 (anomaly boundary)
        confidence = self._calculate_confidence(max_z_score)

        if scene_graph_frame is not None:
            # store as epoch seconds (float) to preserve flags: dict[str, float]
            try:
                flags["scene_graph_frame"] = float(scene_graph_frame.timestamp())
            except Exception:
                logger.debug(
                    "Could not convert scene_graph_frame to timestamp: %r", scene_graph_frame
                )

        return StateEstimate(
            timestamp=now,
            label=label,
            confidence=confidence,
            residuals=residuals,
            uncertainty=uncertainty,
            ttd=None,  # Time to degradation (not implemented)
            ttf=None,  # Time to failure (not implemented)
            flags=flags,
            source_intent=current_intent,
            symbolic_system_state=symbolic_system_state,
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
        if max_z_score > SIGMA_THRESHOLD_IMPULSE:
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
