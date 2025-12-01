import logging
from typing import Any

from kitchenwatch.core.interfaces.base_detector import BaseDetector
from kitchenwatch.edge.models.anomaly_event import AnomalySeverity
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.online_learner_state_estimator import OnlineLearnerStateEstimator

logger = logging.getLogger(__name__)


class StatisticalImpulseDetector(BaseDetector):
    """
    Detects sudden, high-magnitude deviations (impulses/impulse)
    by checking the Z-scores calculated by the OnlineLearnerStateEstimator.

    This component is designed for Tier 0 (Safety Critical) scenarios like S0.3
    (Impact During Motion), requiring high sensitivity and low latency.
    """

    # Default threshold for flagging an impulse event (5 standard deviations)
    DEFAULT_Z_SCORE_THRESHOLD: float = 5.0

    # Minimum allowed standard deviation (sigma) to prevent division by zero.
    # Used to ensure the state estimator has learned *something* before detection.
    _FEATURE_MIN_UNCERTAINTY: float = 1e-6

    # ----------------------------------------------------------------------

    def __init__(
        self,
        state_estimator: OnlineLearnerStateEstimator,
        z_score_threshold: float = DEFAULT_Z_SCORE_THRESHOLD,
    ) -> None:
        """
        Initializes the detector.

        Args:
            state_estimator: The component producing the statistical state.
            z_score_threshold: The Z-score (standard deviation) above which
                               an impulse event is flagged as 'high' severity.
                               Defaults to DEFAULT_Z_SCORE_THRESHOLD.
        """
        self._estimator = state_estimator
        self._threshold = z_score_threshold

    async def detect(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        """
        Detects anomalies based on the Z-score of the current snapshot's residuals.

        Returns:
            Anomaly dictionary if high deviation is found, otherwise an empty dict.
        """
        # Ensure the estimator has run and updated the snapshot with its state
        state_estimate = await self._estimator.update(snapshot)

        residuals = state_estimate.residuals
        uncertainty = state_estimate.uncertainty

        if not uncertainty or not residuals:
            logger.debug("Skipping tick: residuals or uncertainty data is missing.")
            return {}  # Cannot detect without required metrics

        max_z_score = 0.0
        feature_name = ""

        # Find the feature with the highest Z-score
        for key, res in residuals.items():
            sigma = uncertainty.get(key, 0.0)

            # Avoid division by zero, using the class constant
            if sigma > self._FEATURE_MIN_UNCERTAINTY:
                z = abs(res) / sigma
                if z > max_z_score:
                    max_z_score = z
                    feature_name = key

        # Decision Logic (Tier 0 - Safety Critical)
        if max_z_score >= self._threshold:
            # Check defensively before accessing dictionary keys
            if not feature_name:
                logger.error(
                    f"Max Z-score {max_z_score:.2f} reached, but feature_name is empty. Data inconsistency."
                )
                return {}

            # Map Z-score magnitude to anomaly_score (0.0 - 1.0)
            # Normalize the score relative to the threshold for smooth scaling
            anomaly_score = min(1.0, (max_z_score - self._threshold) / self._threshold)

            # Log the full details for production monitoring/debugging
            logger.warning(
                f"🚨 Impulse detected! Z={max_z_score:.2f} ({feature_name}). "
                f"Score={anomaly_score:.2f}. Residual={residuals[feature_name]:.3f}, "
                f"Sigma={uncertainty[feature_name]:.3f}"
            )

            # Defensive block to prevent crashing on final dictionary assembly
            try:
                return {
                    "key": "ft_impact_impulse",
                    "anomaly_score": anomaly_score,
                    "severity": AnomalySeverity.HIGH,
                    "metadata": {
                        "max_z_score": max_z_score,
                        "trigger_feature": feature_name,
                        "residual": residuals[feature_name],
                        "uncertainty": uncertainty[feature_name],
                    },
                }
            except KeyError:
                logger.error(
                    f"Failed to assemble final anomaly dict due to missing key for '{feature_name}'.",
                    exc_info=True,
                )
                return {}

        return {}
