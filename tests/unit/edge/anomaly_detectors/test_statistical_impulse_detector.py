from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from cortexguard.core.interfaces.base_online_learner import BaseOnlineLearner
from cortexguard.edge.detectors.numeric.statistical_impulse_detector import (
    StatisticalImpulseDetector,
)
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot
from cortexguard.edge.models.state_estimate import StateEstimate
from cortexguard.edge.online_learner_state_estimator import OnlineLearnerStateEstimator


class _DummyLearner(BaseOnlineLearner):
    def predict(self, features: dict[str, float]) -> dict[str, float]:
        return {k: v for k, v in features.items()}

    def update(self, features: dict[str, float]) -> None:
        pass

    def anomaly_score(self, features: dict[str, float]) -> float:
        return 0.0


@pytest.fixture
def mock_blackboard() -> Blackboard:
    """Blackboard stub: get_latest_state_estimate returns None by default."""
    bb = MagicMock(spec=Blackboard)
    bb.get_latest_state_estimate = AsyncMock(return_value=None)
    return bb


@pytest.fixture
def mock_snapshot() -> FusionSnapshot:
    """Fixture for a simple mock FusionSnapshot."""
    return FusionSnapshot(id="111", timestamp=datetime.now(UTC), derived={}, sensors={})


def set_blackboard_state(
    bb: Blackboard,
    residuals: dict[str, float],
    uncertainty: dict[str, float],
) -> None:
    """Configure the blackboard stub to return a StateEstimate with given residuals/uncertainty."""
    state = StateEstimate(
        timestamp=datetime.now(UTC),
        label="mock",
        confidence=1.0,
        residuals=residuals,
        uncertainty=uncertainty,
    )
    bb.get_latest_state_estimate = AsyncMock(return_value=state)  # type: ignore[method-assign]


@pytest.mark.asyncio
class TestStatisticalImpulseDetector:
    # --- Test 1: High Impulse Detected (Trigger) ---

    async def test_high_z_score_triggers_impulse_event(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        """
        Tests detection when the max Z-score exceeds the default threshold (5.0).
        Z-score calculation: |Residual| / Uncertainty = |12.0| / 2.0 = 6.0
        """
        residuals = {"force_z": 12.0, "torque_x": 1.0}
        uncertainty = {"force_z": 2.0, "torque_x": 0.5}
        set_blackboard_state(mock_blackboard, residuals, uncertainty)

        detector = StatisticalImpulseDetector(mock_blackboard, z_score_threshold=5.0)

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
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        """
        Tests detection when max Z-score is below the threshold (e.g., 4.0 < 5.0).
        Z-score calculation: |Residual| / Uncertainty = |2.0| / 0.5 = 4.0
        """
        residuals = {"temp": 5.0, "torque_x": 2.0}
        uncertainty = {"temp": 2.0, "torque_x": 0.5}
        set_blackboard_state(mock_blackboard, residuals, uncertainty)

        detector = StatisticalImpulseDetector(mock_blackboard, z_score_threshold=5.0)

        result = await detector.detect(mock_snapshot)

        assert result == {}

    # --- Test 3: Multiple Features (Picks the Max Z-Score) ---

    async def test_correctly_identifies_max_z_score_feature(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        """
        Tests that the detector selects the highest Z-score out of multiple features.
        - Z1 ('t_x'): |1.5| / 0.2 = 7.5 (MAX)
        - Z2 ('f_y'): |10.0| / 2.0 = 5.0
        """
        residuals = {"torque_x": 1.5, "force_y": -10.0}
        uncertainty = {"torque_x": 0.2, "force_y": 2.0}
        set_blackboard_state(mock_blackboard, residuals, uncertainty)

        detector = StatisticalImpulseDetector(mock_blackboard, z_score_threshold=6.0)

        result = await detector.detect(mock_snapshot)

        assert result["key"] == "ft_impact_impulse"
        assert result["metadata"]["max_z_score"] == pytest.approx(7.5)
        assert result["metadata"]["trigger_feature"] == "torque_x"

    # --- Test 4: Handling Zero Uncertainty (Initial Startup) ---

    async def test_zero_uncertainty_is_ignored(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        """
        Tests that zero uncertainty (sigma) prevents division by zero and detection.
        This often happens early in the lifecycle when history is insufficient.
        """
        residuals = {"force_z": 100.0}
        uncertainty = {"force_z": 0.0}
        set_blackboard_state(mock_blackboard, residuals, uncertainty)

        detector = StatisticalImpulseDetector(mock_blackboard, z_score_threshold=5.0)

        result = await detector.detect(mock_snapshot)

        assert result == {}

    # --- Test 5: Anomaly Score Calculation Edges ---

    async def test_anomaly_score_caps_at_one(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        """
        Tests that the anomaly score is capped at 1.0 even with massive Z-scores.
        Z-score: 50.0 / 1.0 = 50.0. Threshold: 5.0
        Score calculation: (50.0 - 5.0) / 5.0 = 9.0 (should be capped at 1.0)
        """
        residuals = {"force_z": 50.0}
        uncertainty = {"force_z": 1.0}
        set_blackboard_state(mock_blackboard, residuals, uncertainty)

        detector = StatisticalImpulseDetector(mock_blackboard, z_score_threshold=5.0)

        result = await detector.detect(mock_snapshot)

        assert result["anomaly_score"] == pytest.approx(1.0)
        assert result["metadata"]["max_z_score"] == pytest.approx(50.0)

    # --- Test 6: Cold start (no estimate yet) ---

    async def test_returns_empty_when_no_state_estimate(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        """blackboard has no estimate yet → return {} without crashing."""
        # mock_blackboard fixture returns None by default
        detector = StatisticalImpulseDetector(mock_blackboard, z_score_threshold=5.0)
        result = await detector.detect(mock_snapshot)
        assert result == {}


# ---------------------------------------------------------------------------
# Regression: double estimator update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_does_not_call_estimator_update() -> None:
    """detect() must not call estimator.update() — EdgeFusion already does that once per snapshot."""
    # Real blackboard — the detector reads from it; the estimator writes to it.
    bb = Blackboard()
    bb_for_estimator = MagicMock(spec=Blackboard)
    bb_for_estimator.get_current_step = AsyncMock(return_value=None)
    bb_for_estimator.get_scene_graph = AsyncMock(return_value=None)

    estimator = OnlineLearnerStateEstimator(
        learner=_DummyLearner(),
        blackboard=bb_for_estimator,
        window_size=50,
        min_history=1,
    )

    snapshot = FusionSnapshot(
        id="snap1",
        timestamp=datetime.now(UTC),
        derived={"force_z": 1.0},
        sensors={},
    )

    # Simulate EdgeFusion: call update once, publish result to the shared blackboard.
    state = await estimator.update(snapshot)
    assert estimator._update_count == 1
    await bb.update_state_estimate(state)

    # Detector reads from the blackboard — must NOT call estimator.update() again.
    detector = StatisticalImpulseDetector(bb, z_score_threshold=5.0)
    await detector.detect(snapshot)

    assert estimator._update_count == 1, (
        f"estimator.update() called {estimator._update_count} times; expected 1. "
        "StatisticalImpulseDetector.detect must read from the blackboard, "
        "not call estimator.update() a second time."
    )
