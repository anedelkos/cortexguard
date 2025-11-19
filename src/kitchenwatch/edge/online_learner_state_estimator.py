import datetime
import logging
from collections import deque
from statistics import stdev

from kitchenwatch.core.interfaces.base_online_learner import BaseOnlineLearner
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
        window_size: int = 50,
        sigma_threshold: float = SIGMA_THRESHOLD_NOMINAL,
        min_history: int = MIN_HISTORY_SAMPLES,
        custom_logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize state estimator.

        Args:
            learner: Online learning model for predictions
            window_size: Number of recent residuals to retain per feature
            sigma_threshold: Z-score threshold for anomaly detection
            min_history: Minimum samples needed before computing statistics
            custom_logger: Optional logger instance
        """
        self._learner = learner
        self._residuals: dict[str, deque[float]] = {}
        self._window_size = window_size
        self._logger = custom_logger or logger

        # Sensitivity Configuration
        self._sigma_threshold = sigma_threshold
        self._min_history = min_history

        # Metrics for observability
        self._update_count = 0
        self._anomaly_count = 0

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

        # Use EMA-smoothed features to reduce noise
        features = snapshot.derived.copy()

        if not features:
            self._logger.warning("Empty features in snapshot, returning nominal state")
            return self._create_nominal_state(now, snapshot.intent)

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

        # Track anomalies
        if label != "nominal":
            self._anomaly_count += 1

        # 6. Calculate confidence (inverse of normalized Z-score)
        # z=0 → confidence=1.0 (perfect match)
        # z=threshold → confidence=0.0 (anomaly boundary)
        confidence = self._calculate_confidence(max_z_score)

        return StateEstimate(
            timestamp=now,
            label=label,
            confidence=confidence,
            residuals=residuals,
            uncertainty=uncertainty,
            ttd=None,  # Time to degradation (not implemented)
            ttf=None,  # Time to failure (not implemented)
            flags=flags,
            source_intent=snapshot.intent,
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
