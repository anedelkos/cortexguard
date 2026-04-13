from __future__ import annotations

import logging
from typing import Any

from cortexguard.core.interfaces.base_detector import BaseDetector
from cortexguard.edge.models.anomaly_event import AnomalySeverity
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot

logger = logging.getLogger(__name__)


class StatisticalImpulseDetector(BaseDetector):
    """
    Detects sudden, high-magnitude deviations (impulses/impulse)
    by checking the Z-scores calculated by the OnlineLearnerStateEstimator.

    This component is designed for Tier 0 (Safety Critical) scenarios like S0.3
    (Impact During Motion), requiring high sensitivity and low latency.

    The detector reads the StateEstimate from the Blackboard rather than calling
    update() on the estimator directly. EdgeFusion is the sole caller of
    estimator.update(); calling it a second time here would corrupt the residual
    window and learner state (C3).
    """

    # Default threshold for flagging an impulse event (5 standard deviations)
    _DEFAULT_Z_SCORE_THRESHOLD: float = 5.0

    # Minimum allowed standard deviation (sigma) to prevent division by zero.
    # Used to ensure the state estimator has learned *something* before detection.
    _FEATURE_MIN_UNCERTAINTY: float = 1e-6

    # ----------------------------------------------------------------------

    def __init__(
        self,
        blackboard: Blackboard,
        z_score_threshold: float = _DEFAULT_Z_SCORE_THRESHOLD,
        min_uncertainty_for_detection: float | None = None,
    ) -> None:
        """
        Initializes the detector.

        Args:
            blackboard: Shared state bus; the detector reads the latest
                        StateEstimate from it (published by EdgeFusion).
            z_score_threshold: The Z-score (standard deviation) above which
                               an impulse event is flagged as 'high' severity.
                               Defaults to DEFAULT_Z_SCORE_THRESHOLD.
            min_uncertainty_for_detection: Treat features with uncertainty
                               <= this value as "not ready" and skip detection.
                               Defaults to _FEATURE_MIN_UNCERTAINTY.
        """
        self._blackboard = blackboard
        self._threshold = z_score_threshold
        # allow overriding the minimum uncertainty used to decide "ready for detection"
        self._min_uncertainty_for_detection = (
            min_uncertainty_for_detection
            if min_uncertainty_for_detection is not None
            else self._FEATURE_MIN_UNCERTAINTY
        )

    async def detect(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        """
        Detects anomalies based on the Z-score of the current snapshot's residuals.

        Reads the StateEstimate published to the Blackboard by EdgeFusion.
        Returns an empty dict if no estimate is available yet (cold start).

        Returns:
            Anomaly dictionary if high deviation is found, otherwise an empty dict.
        """
        state_estimate = await self._blackboard.get_latest_state_estimate()
        if state_estimate is None:
            logger.debug("No StateEstimate available yet; skipping impulse detection tick.")
            return {}

        residuals = state_estimate.residuals or {}
        uncertainty = state_estimate.uncertainty or {}

        if not residuals and not getattr(state_estimate, "z_scores", None):
            logger.debug("Skipping tick: no residuals or z_scores available.")
            return {}

        # Prefer z_scores produced by the estimator (centered, floored, persisted)
        z_scores = getattr(state_estimate, "z_scores", None)

        max_z_score = 0.0
        feature_name = ""

        if z_scores:
            # Use estimator-provided z scores directly
            for k, z in z_scores.items():
                if z is None:
                    continue
                if z > max_z_score:
                    max_z_score = z
                    feature_name = k
        else:
            # Fallback: recompute z using estimator uncertainty but skip features
            # with effectively zero uncertainty (insufficient data).
            MIN_SIGMA = 1e-3  # used only for numerical stability when sigma > 0
            for key, res in residuals.items():
                sigma = uncertainty.get(key, 0.0)

                # If sigma is effectively zero or below the detection floor, skip this feature
                if sigma <= self._min_uncertainty_for_detection:
                    logger.debug(
                        "Skipping detection for %s due to insufficient uncertainty (sigma=%s).",
                        key,
                        sigma,
                    )
                    continue

                # Use a practical floor only for numerical stability, not to force detection from zero sigma
                sigma_used = max(sigma, MIN_SIGMA)
                z = abs(res) / sigma_used
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
                f"Score={anomaly_score:.2f}. Residual={residuals.get(feature_name, 0.0):.3f}, "
                f"Sigma={uncertainty.get(feature_name, 0.0):.3f}"
            )

            return {
                "key": "ft_impact_impulse",
                "anomaly_score": anomaly_score,
                "severity": AnomalySeverity.HIGH,
                "metadata": {
                    "max_z_score": max_z_score,
                    "trigger_feature": feature_name,
                    "residual": residuals.get(feature_name, 0.0),
                    "uncertainty": uncertainty.get(feature_name, 0.0),
                },
            }

        return {}
