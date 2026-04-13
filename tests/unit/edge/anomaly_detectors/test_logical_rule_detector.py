import logging
from collections.abc import Callable
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from cortexguard.edge.detectors.rule_based.logical_rule_detector import LogicalRuleDetector
from cortexguard.edge.models.anomaly_event import AnomalySeverity
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot


@pytest.fixture
def mock_logger() -> MagicMock:
    """Fixture for a mock logger instance."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def mock_snapshot() -> Callable[[bool | None], FusionSnapshot]:
    temp_value = 25.0

    def _create_snapshot(success: bool | None) -> FusionSnapshot:
        nonlocal temp_value
        temp_value += 0.1  # ensure non-zero variance

        derived_data = {}
        if success is not None:
            derived_data[LogicalRuleDetector.KEY_SYSTEM_SUCCESS_STATUS] = success

        sensors = {
            "window_stats": {
                "force_n": {
                    "mean": 1.0,
                    "std": 0.1,
                    "min": 0.9,
                    "max": 1.1,
                    "range": 0.2,
                }
            },
            "raw": [],
            "temp_celsius": temp_value,
        }

        return FusionSnapshot(
            id="111",
            timestamp=datetime.now(UTC),
            sensors=sensors,
            derived=derived_data,
        )

    return _create_snapshot


@pytest.fixture
def detector(mock_logger: MagicMock) -> LogicalRuleDetector:
    """Fixture for a LogicalRuleDetector instance with default settings."""
    return LogicalRuleDetector()


# --- Test S1.1: Consecutive Failure Logic ---


@pytest.mark.asyncio
async def test_initial_state_no_anomaly(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """Should return no anomaly on successful tick."""
    snapshot = mock_snapshot(True)
    result = await detector.detect(snapshot)
    assert result == {}
    assert detector._consecutive_failure_count == 0


@pytest.mark.asyncio
async def test_single_failure_no_anomaly(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """Should increment counter but not trigger anomaly on single failure."""
    snapshot = mock_snapshot(False)
    result = await detector.detect(snapshot)
    assert result == {}
    assert detector._consecutive_failure_count == 1


@pytest.mark.asyncio
async def test_failure_reset_on_success(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """Failure count should reset to zero after a success."""
    # Fail 1
    await detector.detect(mock_snapshot(False))
    # Success
    await detector.detect(mock_snapshot(True))
    assert detector._consecutive_failure_count == 0


@pytest.mark.asyncio
async def test_anomaly_triggers_on_max_consecutive_failures(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """Should trigger MEDIUM anomaly when failure limit is reached (3 failures)."""
    # Fail 1
    await detector.detect(mock_snapshot(False))
    # Fail 2
    await detector.detect(mock_snapshot(False))
    assert detector._consecutive_failure_count == 2

    # Fail 3 (Triggers anomaly)
    snapshot = mock_snapshot(False)
    result = await detector.detect(snapshot)

    assert result["key"] == "repeated_system_failure"
    assert result["severity"] == AnomalySeverity.MEDIUM.value
    # Score calculation: 3 / 3 = 1.0
    assert result["anomaly_score"] == 1.0
    assert result["metadata"]["failure_count"] == 3
    assert detector._consecutive_failure_count == 3


@pytest.mark.asyncio
async def test_failure_continues_after_anomaly(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """Failure count should continue to increment after the anomaly is triggered."""
    # Fail 1, 2, 3 (Anomaly triggered)
    for _ in range(3):
        await detector.detect(mock_snapshot(False))

    # Fail 4 (Anomaly should still be returned, score might be > 1.0 but clamped)
    result = await detector.detect(mock_snapshot(False))
    assert result["key"] == "repeated_system_failure"
    assert detector._consecutive_failure_count == 4
    # Score calculation: min(1.0, 4/3) = 1.0
    assert result["anomaly_score"] == 1.0


@pytest.mark.asyncio
async def test_no_anomaly_if_key_is_missing(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """If the key is missing in the snapshot, state should be held (no change)."""
    # Fail 1
    await detector.detect(mock_snapshot(False))
    assert detector._consecutive_failure_count == 1

    # Missing key (success=None)
    result = await detector.detect(mock_snapshot(None))
    assert result == {}
    assert detector._consecutive_failure_count == 1  # Should not change


# ---------------------------------------------------------------------------
# Regression: freeze deque must not fill on repeated ticks of the same snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_false_freeze_on_repeated_same_snapshot() -> None:
    """Calling detect() N times on the same snapshot must never trigger a false freeze.

    The detector polls at 100 ms while records arrive at 1 s, so the same
    FusionSnapshot is processed ~10× before a new one arrives.  Prior to the
    fix the internal deque accumulated identical values on each tick, causing
    range=0 → false temp_celsius_value_freeze.
    """
    detector = LogicalRuleDetector()
    snapshot = FusionSnapshot(
        id="same-snapshot-id",
        timestamp=datetime.now(UTC),
        sensors={"temp_celsius": 25.0, "window_stats": {}, "raw": []},
        derived={},
    )

    for _ in range(10):
        result = await detector.detect(snapshot)
        assert result == {}, f"False freeze on repeated tick: {result}"


@pytest.mark.asyncio
async def test_freeze_detected_across_constant_snapshots() -> None:
    """A genuine sensor freeze (constant value across distinct snapshots) must be detected."""
    detector = LogicalRuleDetector()
    const_temp = 25.0

    for i in range(detector._SNAPSHOT_HISTORY_K):
        snapshot = FusionSnapshot(
            id=f"snap-{i}",
            timestamp=datetime.now(UTC),
            sensors={"temp_celsius": const_temp, "window_stats": {}, "raw": []},
            derived={},
        )
        result = await detector.detect(snapshot)

    assert (
        result.get("key") == "temp_celsius_value_freeze"
    ), f"Expected freeze detection after {detector._SNAPSHOT_HISTORY_K} distinct constant snapshots, got: {result}"


# ---------------------------------------------------------------------------
# Regression: noise-band values must not trigger freeze (threshold sensitivity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_false_freeze_on_noisy_baseline() -> None:
    """Values varying within sensor noise band must not trigger a false freeze.

    With epsilon=0.01 and baseline noise of ±0.02°C, three consecutive
    snapshots occasionally clustered within 0.01°C of each other, causing
    false temp_celsius_value_freeze detections on clean S0.0 data.
    The epsilon was raised to 0.05 to require a genuinely stuck sensor.
    """
    detector = LogicalRuleDetector()
    # Values vary by ~0.4°C — within realistic sensor noise (±0.5°C), well above epsilon=0.01
    noisy_temps = [0.3, -0.1, 0.2]

    for i, temp in enumerate(noisy_temps):
        snapshot = FusionSnapshot(
            id=f"noisy-{i}",
            timestamp=datetime.now(UTC),
            sensors={"temp_celsius": temp, "window_stats": {}, "raw": []},
            derived={},
        )
        result = await detector.detect(snapshot)

    assert (
        result == {}
    ), f"False freeze on noisy baseline (range={max(noisy_temps)-min(noisy_temps):.3f}°C): {result}"


# ---------------------------------------------------------------------------
# Regression: bare dict["key"] accesses in _check_value_freeze
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_value_freeze_does_not_raise_on_missing_sensor_keys(
    detector: LogicalRuleDetector,
) -> None:
    """Snapshot with missing sensor keys must return {} gracefully, not raise KeyError."""
    snapshot = FusionSnapshot(
        id="bare-key-test",
        timestamp=datetime.now(UTC),
        derived={},
        sensors={},  # all expected keys absent
    )

    # Should return {} gracefully, not raise KeyError
    result = await detector.detect(snapshot)

    assert result == {}
