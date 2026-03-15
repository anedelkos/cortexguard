from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from cortexguard.edge.detectors.numeric.statistical_impulse_detector import (
    StatisticalImpulseDetector,
)
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot
from cortexguard.edge.models.state_estimate import StateEstimate
from cortexguard.edge.online_learner_state_estimator import OnlineLearnerStateEstimator


@pytest.fixture
def mock_state_estimator() -> OnlineLearnerStateEstimator:
    """Fixture for a fully mocked OnlineLearnerStateEstimator."""
    # We only need the 'update' method to be an AsyncMock
    mock_estimator = MagicMock(spec=OnlineLearnerStateEstimator)
    mock_estimator.update = AsyncMock()
    return mock_estimator


@pytest.fixture
def mock_snapshot() -> FusionSnapshot:
    """Fixture for a simple mock FusionSnapshot."""
    return FusionSnapshot(id="111", timestamp=datetime.now(UTC), derived={}, sensors={})


# Helper function to configure the mock estimator's return value
def set_estimator_return(
    mock_estimator: OnlineLearnerStateEstimator,
    residuals: dict[str, float],
    uncertainty: dict[str, float],
) -> None:
    """Sets the return value for the mock estimator's update method."""
    mock_state = StateEstimate(
        timestamp=datetime.now(UTC),
        label="mock",
        confidence=1.0,
        residuals=residuals,
        uncertainty=uncertainty,
    )

    cast(AsyncMock, mock_estimator.update).return_value = mock_state


@pytest.mark.asyncio
class TestStatisticalImpulseDetector:
    # --- Test 1: High Impulse Detected (Trigger) ---

    async def test_high_z_score_triggers_impulse_event(
        self, mock_state_estimator: OnlineLearnerStateEstimator, mock_snapshot: FusionSnapshot
    ) -> None:
        """
        Tests detection when the max Z-score exceeds the default threshold (5.0).
        Z-score calculation: |Residual| / Uncertainty = |12.0| / 2.0 = 6.0
        """
        # Arrange: Set up estimator to return a high Z-score
        residuals = {"force_z": 12.0, "torque_x": 1.0}
        uncertainty = {"force_z": 2.0, "torque_x": 0.5}
        set_estimator_return(mock_state_estimator, residuals, uncertainty)

        detector = StatisticalImpulseDetector(mock_state_estimator, z_score_threshold=5.0)

        result = await detector.detect(mock_snapshot)

        assert result["key"] == "ft_impact_impulse"
        assert result["severity"] == "high"

        # Anomaly score calculation: (Z_max - Threshold) / Threshold = (6.0 - 5.0) / 5.0 = 0.2
        assert result["anomaly_score"] == pytest.approx(0.2)

        metadata = result["metadata"]
        assert metadata["max_z_score"] == pytest.approx(6.0)
        assert metadata["trigger_feature"] == "force_z"
        assert metadata["residual"] == 12.0
        assert metadata["uncertainty"] == 2.0

    # --- Test 2: Sub-Threshold (No Trigger) ---

    async def test_sub_threshold_z_score_returns_empty(
        self, mock_state_estimator: OnlineLearnerStateEstimator, mock_snapshot: FusionSnapshot
    ) -> None:
        """
        Tests detection when max Z-score is below the threshold (e.g., 4.0 < 5.0).
        Z-score calculation: |Residual| / Uncertainty = |2.0| / 0.5 = 4.0
        """
        # Arrange: Set up estimator to return a sub-threshold Z-score
        residuals = {"temp": 5.0, "torque_x": 2.0}
        uncertainty = {"temp": 2.0, "torque_x": 0.5}
        set_estimator_return(mock_state_estimator, residuals, uncertainty)

        detector = StatisticalImpulseDetector(mock_state_estimator, z_score_threshold=5.0)

        result = await detector.detect(mock_snapshot)

        assert result == {}

    # --- Test 3: Multiple Features (Picks the Max Z-Score) ---

    async def test_correctly_identifies_max_z_score_feature(
        self, mock_state_estimator: OnlineLearnerStateEstimator, mock_snapshot: FusionSnapshot
    ) -> None:
        """
        Tests that the detector selects the highest Z-score out of multiple features.
        - Z1 ('t_x'): |1.5| / 0.2 = 7.5 (MAX)
        - Z2 ('f_y'): |10.0| / 2.0 = 5.0
        """
        residuals = {"torque_x": 1.5, "force_y": -10.0}
        uncertainty = {"torque_x": 0.2, "force_y": 2.0}
        set_estimator_return(mock_state_estimator, residuals, uncertainty)

        detector = StatisticalImpulseDetector(mock_state_estimator, z_score_threshold=6.0)

        result = await detector.detect(mock_snapshot)

        assert result["key"] == "ft_impact_impulse"
        assert result["metadata"]["max_z_score"] == pytest.approx(7.5)
        assert result["metadata"]["trigger_feature"] == "torque_x"

    # --- Test 4: Handling Zero Uncertainty (Initial Startup) ---

    async def test_zero_uncertainty_is_ignored(
        self, mock_state_estimator: OnlineLearnerStateEstimator, mock_snapshot: FusionSnapshot
    ) -> None:
        """
        Tests that zero uncertainty (sigma) prevents division by zero and detection.
        This often happens early in the lifecycle when history is insufficient.
        """
        # Arrange: Uncertainty is zero (or very close)
        residuals = {"force_z": 100.0}
        uncertainty = {"force_z": 0.0}
        set_estimator_return(mock_state_estimator, residuals, uncertainty)

        detector = StatisticalImpulseDetector(mock_state_estimator, z_score_threshold=5.0)

        result = await detector.detect(mock_snapshot)

        # Assert: No detection should occur as Z-score cannot be reliably calculated
        assert result == {}

    # --- Test 5: Anomaly Score Calculation Edges ---

    async def test_anomaly_score_caps_at_one(
        self, mock_state_estimator: OnlineLearnerStateEstimator, mock_snapshot: FusionSnapshot
    ) -> None:
        """
        Tests that the anomaly score is capped at 1.0 even with massive Z-scores.
        Z-score: 50.0 / 1.0 = 50.0. Threshold: 5.0
        Score calculation: (50.0 - 5.0) / 5.0 = 9.0 (should be capped at 1.0)
        """
        residuals = {"force_z": 50.0}
        uncertainty = {"force_z": 1.0}
        set_estimator_return(mock_state_estimator, residuals, uncertainty)

        detector = StatisticalImpulseDetector(mock_state_estimator, z_score_threshold=5.0)

        result = await detector.detect(mock_snapshot)

        assert result["anomaly_score"] == pytest.approx(1.0)
        assert result["metadata"]["max_z_score"] == pytest.approx(50.0)
