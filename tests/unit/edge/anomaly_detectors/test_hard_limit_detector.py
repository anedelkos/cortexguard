import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from cortexguard.edge.detectors.rule_based import HardLimitDetector
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot


@pytest.fixture
def mock_logger() -> MagicMock:
    """Fixture for a mock logger."""
    # We return MagicMock so tests can access .assert_called, etc.
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def snapshot_factory() -> Callable[[dict[str, Any]], FusionSnapshot]:
    """Helper to create snapshots with specific sensor data."""

    def _create(sensors: dict[str, Any]) -> FusionSnapshot:
        return FusionSnapshot(
            id="111",
            timestamp=datetime.now(UTC),
            sensors=sensors,
            derived={},  # Not used by HardLimitDetector
        )

    return _create


@pytest.mark.asyncio
class TestHardLimitDetector:
    # --- Test 1: Initialization ---

    async def test_initialization_defaults(self, mock_logger: MagicMock) -> None:
        # Fix: Cast MagicMock to Logger to satisfy detector's type hint
        detector = HardLimitDetector(logger=cast(logging.Logger, mock_logger))

        assert detector._temp_threshold == 70.0
        # This is now valid because mock_logger is typed as MagicMock
        mock_logger.debug.assert_called()

    async def test_initialization_override(self, mock_logger: MagicMock) -> None:
        detector = HardLimitDetector(logger=cast(logging.Logger, mock_logger), temp_threshold=55.5)
        assert detector._temp_threshold == 55.5

    # --- Test 2: Normal Operation (No Anomaly) ---

    async def test_detect_returns_empty_when_normal(
        self, mock_logger: MagicMock, snapshot_factory: Callable[[dict[str, Any]], FusionSnapshot]
    ) -> None:
        # Arrange: Temp below threshold (60 < 70), No smoke
        sensors = {"temp_celsius": 60.0, "smoke_detected": False}
        snapshot = snapshot_factory(sensors)
        detector = HardLimitDetector(logger=cast(logging.Logger, mock_logger))

        result = await detector.detect(snapshot)

        assert result == {}
        mock_logger.warning.assert_not_called()
        mock_logger.critical.assert_not_called()

    # --- Test 3: Scenario S0.2 - Fire (Overheat + Smoke) ---

    async def test_detect_fire_combo_triggers_high_severity(
        self, mock_logger: MagicMock, snapshot_factory: Callable[[dict[str, Any]], FusionSnapshot]
    ) -> None:
        # Arrange: Temp > 70 AND Smoke = True
        sensors = {"temp_celsius": 80.0, "smoke_detected": True}
        snapshot = snapshot_factory(sensors)
        detector = HardLimitDetector(logger=cast(logging.Logger, mock_logger))

        result = await detector.detect(snapshot)

        assert result["key"] == "overheat_smoke_combo"
        assert result["severity"] == "high"
        assert result["anomaly_score"] == 1.0
        assert result["metadata"]["smoke"] is True

        mock_logger.critical.assert_called_once()

    # --- Test 4: Overheat Warning (No Smoke) ---

    async def test_detect_overheat_only_triggers_medium_severity(
        self, mock_logger: MagicMock, snapshot_factory: Callable[[dict[str, Any]], FusionSnapshot]
    ) -> None:
        # Arrange: Temp > 70, but Smoke = False
        sensors = {"temp_celsius": 75.0, "smoke_detected": False}
        snapshot = snapshot_factory(sensors)
        detector = HardLimitDetector(logger=cast(logging.Logger, mock_logger))

        result = await detector.detect(snapshot)

        assert result["key"] == "temp_limit_breach"
        assert result["severity"] == "medium"
        # Score logic: min(1.0, (75 - 70) / 10) = 0.5
        assert result["anomaly_score"] == pytest.approx(0.5)

        mock_logger.warning.assert_called_once()

    # --- Test 5: Boundary Conditions ---

    async def test_boundary_exact_threshold_is_safe(
        self, mock_logger: MagicMock, snapshot_factory: Callable[[dict[str, Any]], FusionSnapshot]
    ) -> None:
        # Arrange: Temp is EXACTLY 70.0
        sensors = {"temp_celsius": 70.0, "smoke_detected": True}
        snapshot = snapshot_factory(sensors)
        detector = HardLimitDetector(logger=cast(logging.Logger, mock_logger))

        result = await detector.detect(snapshot)

        # Assert: Should be safe (logic is > not >=)
        assert result == {}

    async def test_boundary_just_over_threshold(
        self, mock_logger: MagicMock, snapshot_factory: Callable[[dict[str, Any]], FusionSnapshot]
    ) -> None:
        # Arrange: Temp is 70.01
        sensors = {"temp_celsius": 70.01, "smoke_detected": False}
        snapshot = snapshot_factory(sensors)
        detector = HardLimitDetector(logger=cast(logging.Logger, mock_logger))

        result = await detector.detect(snapshot)

        # Assert: Should trigger
        assert result["key"] == "temp_limit_breach"

    # --- Test 6: Robustness & Error Handling ---

    async def test_missing_sensor_key_returns_empty(
        self, mock_logger: MagicMock, snapshot_factory: Callable[[dict[str, Any]], FusionSnapshot]
    ) -> None:
        # Arrange: Empty sensors dict
        snapshot = snapshot_factory({})
        detector = HardLimitDetector(logger=cast(logging.Logger, mock_logger))

        result = await detector.detect(snapshot)

        assert result == {}
        # Should not log errors, just return empty silently (as per design)
        mock_logger.error.assert_not_called()

    async def test_invalid_data_type_logs_error(
        self, mock_logger: MagicMock, snapshot_factory: Callable[[dict[str, Any]], FusionSnapshot]
    ) -> None:
        # Arrange: Temp is a string "hot" instead of float
        sensors = {"temp_celsius": "hot", "smoke_detected": False}
        snapshot = snapshot_factory(sensors)
        detector = HardLimitDetector(logger=cast(logging.Logger, mock_logger))

        result = await detector.detect(snapshot)

        assert result == {}
        # Should detect the type mismatch and log an error
        mock_logger.error.assert_called_once()
        assert "Invalid type" in mock_logger.error.call_args[0][0]
