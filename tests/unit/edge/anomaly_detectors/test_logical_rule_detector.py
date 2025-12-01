import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from freezegun import freeze_time

from kitchenwatch.edge.detectors.rule_based.logical_rule_detector import LogicalRuleDetector
from kitchenwatch.edge.models.anomaly_event import AnomalySeverity
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot


@pytest.fixture
def mock_logger() -> MagicMock:
    """Fixture for a mock logger instance."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def mock_snapshot() -> Callable[[bool | None], FusionSnapshot]:
    """
    Fixture for a factory function to create mock FusionSnapshot objects.
    This helps simulate different states easily.
    """

    def _create_snapshot(success: bool | None) -> FusionSnapshot:
        derived_data: dict[str, Any] = {}
        if success is not None:
            derived_data[LogicalRuleDetector.KEY_SYSTEM_SUCCESS_STATUS] = success

        return FusionSnapshot(
            id="111",
            timestamp=datetime.now(),
            sensors={},
            derived=derived_data,
        )

    return _create_snapshot


@pytest.fixture
def detector(mock_logger: MagicMock) -> LogicalRuleDetector:
    """Fixture for a LogicalRuleDetector instance with default settings."""
    return LogicalRuleDetector(
        max_failures=3,
        freeze_limit_s=2.0,
    )


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


# --- Test S2.3: Data Freeze Logic (Requires time mocking) ---


@pytest.mark.asyncio
@freeze_time("2024-01-01 10:00:00")
async def test_freeze_no_anomaly_on_first_tick(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """First tick should initialize the time state but not trigger a freeze."""
    result = await detector.detect(mock_snapshot(True))
    assert result == {}


@pytest.mark.asyncio
@freeze_time("2024-01-01 10:00:00")
async def test_freeze_no_anomaly_on_fast_ticks(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """Should not trigger anomaly if ticks are within the limit (2.0s)."""
    # Tick 1: Init time to 10:00:00
    await detector.detect(mock_snapshot(True))

    # Tick 2: 1.5 seconds passed (within limit)
    with freeze_time("2024-01-01 10:00:01.5"):
        result = await detector.detect(mock_snapshot(True))
        assert result == {}


@pytest.mark.asyncio
@freeze_time("2024-01-01 10:00:00")
async def test_freeze_anomaly_triggers_on_time_limit_breach(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """Should trigger HIGH anomaly if time limit (2.0s) is breached."""
    # Tick 1: Init time to 10:00:00
    await detector.detect(mock_snapshot(True))

    # Tick 2: 2.1 seconds passed (breach)
    with freeze_time("2024-01-01 10:00:02.1"):
        result = await detector.detect(mock_snapshot(True))

        assert result["key"] == "blackboard_data_freeze"
        assert result["severity"] == AnomalySeverity.HIGH.value
        assert result["metadata"]["limit_s"] == 2.0
        # The reported time since last update should be close to 2.1 seconds
        assert 2.0 < result["metadata"]["time_since_last_update_s"] < 2.2


@pytest.mark.asyncio
@freeze_time("2024-01-01 10:00:00")
async def test_freeze_anomaly_updates_tick_time_and_is_not_sticky(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """
    A freeze anomaly should only last for one tick, as the detector's internal
    tick time is updated upon entering detect().
    """
    # Tick 1: Init time to 10:00:00
    await detector.detect(mock_snapshot(True))

    # Tick 2: 5.0 seconds passed (Freeze triggered)
    with freeze_time("2024-01-01 10:00:05.0"):
        freeze_result = await detector.detect(mock_snapshot(True))
        assert freeze_result["key"] == "blackboard_data_freeze"

    # Tick 3: 1.0 second later (Should now pass the freeze check)
    with freeze_time("2024-01-01 10:00:06.0"):
        # The difference is now 6.0 - 5.0 = 1.0s, which is < 2.0s limit
        clean_result = await detector.detect(mock_snapshot(True))
        assert clean_result == {}


# --- Test Precedence and Interaction ---


@pytest.mark.asyncio
@freeze_time("2024-01-01 10:00:00")
async def test_freeze_anomaly_takes_precedence(
    detector: LogicalRuleDetector, mock_snapshot: Callable[[bool | None], FusionSnapshot]
) -> None:
    """
    If both a freeze (HIGH) and a failure (MEDIUM) are triggered simultaneously,
    the freeze anomaly (HIGH) should be reported due to precedence. The consecutive
    failure count should not increment because the HIGH severity anomaly returns early.
    """
    # 1. Init time state (T0) and ensure the detector state reflects 2 failures.

    # Tick 1 (Init time): Sets _previous_tick_time to 10:00:00
    await detector.detect(mock_snapshot(True))

    # Fail 1 (10:00:00): Time since last update is 0, so no freeze. Count = 1.
    await detector.detect(mock_snapshot(False))

    # Fail 2 (10:00:00): Time since last update is 0, no freeze. Count = 2.
    await detector.detect(mock_snapshot(False))

    # We confirm the counter is at 2 before the final tick
    assert detector._consecutive_failure_count == 2

    # 2. Final Tick (T0 + 3.0s): Time jump and Failure 3

    with freeze_time("2024-01-01 10:00:03.0"):
        # Time elapsed since last tick (10:00:00) is 3.0s (Freeze HIGH)
        # Snapshot reports failure (Failure 3, MEDIUM)
        snapshot = mock_snapshot(False)

        result = await detector.detect(snapshot)

        # S2.3 (Freeze) is checked first and should be returned
        assert result["key"] == "blackboard_data_freeze"
        assert result["severity"] == AnomalySeverity.HIGH.value

        # The failure state update is skipped because the HIGH severity anomaly causes an immediate return.
        assert detector._consecutive_failure_count == 2
