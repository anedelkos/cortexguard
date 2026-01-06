import asyncio
import logging
from datetime import datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kitchenwatch.core.interfaces.base_detector import BaseDetector
from kitchenwatch.edge.detectors.anomaly_detector import AnomalyDetector
from kitchenwatch.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot

# --- Path for Patching ---
ANOMALY_DETECTOR_LOGGER_PATH = "kitchenwatch.edge.detectors.anomaly_detector.logger"


# Define the base mock detector
class MockDetectorBase(BaseDetector):
    """A mock implementation of the BaseDetector Protocol for testing."""

    def __init__(self, key: str, score: float, severity: str, sleep_time: float = 0.0) -> None:
        self.key = key
        self.score = score
        self.severity = severity
        self.sleep_time = sleep_time

    async def detect(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        """Simulates detection logic."""
        await asyncio.sleep(self.sleep_time)
        return {
            "key": self.key,
            "score": self.score,
            "severity": self.severity,
            "metadata": {"processed_at": snapshot.timestamp},
        }


# Detector A (used as the primary mock detector)
class MockDetectorA(MockDetectorBase):
    pass


# Detector B (used specifically to test unique contributions from a second instance)
class MockDetectorB(MockDetectorBase):
    pass


# Define a detector that intentionally fails (to test exception handling)
class FailingDetector(BaseDetector):
    async def detect(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        raise RuntimeError("Detector failed intentionally")


@pytest.fixture
def mock_blackboard() -> Blackboard:
    """
    Fixture for a mocked Blackboard, updated to reflect AnomalyEvent methods.
    """
    # Create a mock object that implements the Blackboard interface
    mock_bb = MagicMock(spec=Blackboard)

    # Assign AsyncMocks to the coroutine methods on the mock object
    mock_bb.get_fusion_snapshot = AsyncMock()
    mock_bb.set_anomaly = AsyncMock()
    mock_bb.clear_anomaly = AsyncMock()
    mock_bb.get_current_step = AsyncMock(return_value=None)

    # We must explicitly set the return type to Blackboard, though it's a mock.
    return cast(Blackboard, mock_bb)


@pytest.fixture
def mock_snapshot() -> FusionSnapshot:
    """Fixture for a mock FusionSnapshot."""
    return FusionSnapshot(id="test_snap_123", timestamp=datetime.now(), derived={}, sensors={})


@pytest.fixture
def test_logger() -> logging.Logger:
    """Fixture for a test logger that captures output."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    # Prevent propagation to avoid confusing test output
    logger.propagate = False
    return logger


@pytest.mark.asyncio
class TestAnomalyDetector:
    # --- Test 1: Initialization and Registration ---

    async def test_initialization_and_registration(self, mock_blackboard: Blackboard) -> None:
        detector = AnomalyDetector(mock_blackboard)

        assert detector._blackboard is mock_blackboard
        assert detector._tick_interval == 0.1
        assert len(detector._sub_detectors) == 0

        mock_sub_detector = MockDetectorA("test", 0.5, "low")
        detector.register_detector(mock_sub_detector)

        assert len(detector._sub_detectors) == 1
        assert detector._sub_detectors[0] is mock_sub_detector

    # --- Test 2: Concurrent Execution and Latency ---

    async def test_concurrent_execution_is_faster_than_sequential(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange: Setup 3 detectors that each take 0.1s
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot

        detector = AnomalyDetector(mock_blackboard)
        detector.register_detector(MockDetectorA("d1", 0.9, "high", sleep_time=0.1))
        detector.register_detector(MockDetectorA("d2", 0.7, "medium", sleep_time=0.1))
        detector.register_detector(MockDetectorA("d3", 0.5, "low", sleep_time=0.1))

        start_time = asyncio.get_event_loop().time()

        # Act: Run one tick
        await detector._run_tick()

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        # Assert: Duration should be close to 0.1s (concurrent), not 0.3s (sequential)
        assert duration == pytest.approx(0.1, abs=0.05)
        # Check that the clear method was not called unnecessarily
        cast(AsyncMock, mock_blackboard.clear_anomaly).assert_not_called()

    # --- Test 3: Result Aggregation Logic ---

    async def test_aggregation_strategy_medium_and_high_trigger_event(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange: Setup detectors with different severities
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot

        detector = AnomalyDetector(mock_blackboard)
        # d1: low severity -> Ignored by aggregation (not active)
        detector.register_detector(MockDetectorA("low_risk", 0.2, "low"))
        # d2: medium severity -> Active event
        detector.register_detector(MockDetectorA("med_risk", 0.6, "medium"))
        # d3: high severity -> Active event
        detector.register_detector(MockDetectorA("high_risk", 0.9, "high"))

        # Act: Run one tick
        await detector._run_tick()

        # Assert 1: set_anomaly called two times (only for medium and high severity results)
        calls = cast(AsyncMock, mock_blackboard.set_anomaly).call_args_list
        assert len(calls) == 2

        # Assert 2: Verify the exact content of the AnomalyEvent objects
        events = [c[0][0] for c in calls]

        # Check the "med_risk" event
        med_event = next(e for e in events if e.key == "med_risk")
        assert isinstance(med_event, AnomalyEvent)
        assert med_event.severity == AnomalySeverity.MEDIUM
        assert med_event.score == 0.6
        assert med_event.contributing_detectors == ["MockDetectorA"]

        # Check the "high_risk" event
        high_event = next(e for e in events if e.key == "high_risk")
        assert isinstance(high_event, AnomalyEvent)
        assert high_event.severity == AnomalySeverity.HIGH
        assert high_event.score == 0.9
        assert high_event.contributing_detectors == ["MockDetectorA"]

        cast(AsyncMock, mock_blackboard.clear_anomaly).assert_not_called()

    # --- Test 4: OR Logic in Aggregation ---

    async def test_aggregation_or_logic_for_duplicate_keys(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange: Setup two detectors using the SAME key but DIFFERENT detector classes
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot

        detector = AnomalyDetector(mock_blackboard)
        # d1: high severity (True, Score 0.9) - Class A
        detector.register_detector(MockDetectorA("collision", 0.9, "high"))
        # d2: high severity (True, Score 0.8) - Class B
        detector.register_detector(MockDetectorB("collision", 0.8, "high"))

        # Act: Run one tick
        await detector._run_tick()

        # Assert: set_anomaly called only ONCE, with the HIGHEST severity and MAX score
        cast(AsyncMock, mock_blackboard.set_anomaly).assert_called_once()

        event = cast(AsyncMock, mock_blackboard.set_anomaly).call_args[0][0]

        assert isinstance(event, AnomalyEvent)
        assert event.key == "collision"
        # Highest severity wins (they are equal: HIGH)
        assert event.severity == AnomalySeverity.HIGH
        # Max score wins (0.9 from A)
        assert event.score == 0.9
        # Both detectors contribute - now they have unique class names!
        assert sorted(event.contributing_detectors) == ["MockDetectorA", "MockDetectorB"]

        cast(AsyncMock, mock_blackboard.clear_anomaly).assert_not_called()

    # --- Test 5: Exception Handling ---

    @pytest.mark.asyncio
    @patch(ANOMALY_DETECTOR_LOGGER_PATH, spec=logging.Logger)
    async def test_failing_detector_does_not_halt_ensemble(
        self,
        mock_logger: MagicMock,
        mock_blackboard: Blackboard,
        mock_snapshot: FusionSnapshot,
    ) -> None:
        # Arrange: Patching is handled by the decorator.
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot

        detector = AnomalyDetector(mock_blackboard)

        # d1: High risk detector (successful)
        detector.register_detector(MockDetectorA("critical", 0.9, "high"))
        # d2: Detector that raises an exception
        detector.register_detector(FailingDetector())

        # Act: Run one tick
        await detector._run_tick()

        # Assert 1: The successful detector result was posted (critical: HIGH)
        cast(AsyncMock, mock_blackboard.set_anomaly).assert_called_once()

        event = cast(AsyncMock, mock_blackboard.set_anomaly).call_args[0][0]
        assert isinstance(event, AnomalyEvent)
        assert event.key == "critical"
        assert event.severity == AnomalySeverity.HIGH
        assert event.score == 0.9
        assert event.contributing_detectors == ["MockDetectorA"]

        # Assert 2: An exception was logged, but the tick completed successfully
        mock_logger.error.assert_called_once()
        # Check the error message
        assert "Sub-detector FailingDetector failed" in mock_logger.error.call_args[0][0]

        cast(AsyncMock, mock_blackboard.clear_anomaly).assert_not_called()

    # --- Test 6: Lifecycle Management ---

    async def test_start_stop_lifecycle(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange: Setup the detector with a low tick interval for quick test run
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot
        detector = AnomalyDetector(mock_blackboard, tick_interval=0.01)
        detector.register_detector(MockDetectorA("a", 0.9, "high"))

        # Act 1: Start the loop
        await detector.start()
        await asyncio.sleep(0.05)  # Allow a few ticks to run

        # Assert 1: Loop is running and processing ticks
        assert detector._loop_running is True
        assert detector._task is not None
        assert detector._ticks_processed > 0

        # Act 2: Stop the loop
        await detector.stop()

        # Assert 2: Loop is stopped and task is done/cancelled
        assert detector._loop_running is False
        assert detector._task is not None
        assert detector._task.done() is True
        assert detector._task.cancelled() is True

        # Assert 3: Subsequent stop call is harmless
        await detector.stop()

    # --- Test 7: Polling and Metrics ---

    async def test_polling_and_metrics_update(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange: Simulate one successful snapshot, followed by None
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).side_effect = [mock_snapshot, None]

        detector = AnomalyDetector(mock_blackboard)
        # Register a detector that always flags TRUE for metric testing
        detector.register_detector(MockDetectorA("always_true", 0.9, "high"))

        # Act 1: Run tick with snapshot available
        await detector._run_tick()

        # Assert 1: Metrics updated after successful tick
        metrics = detector.get_metrics()
        assert metrics["ticks_processed"] == 1
        assert metrics["anomalies_detected"] == 1
        assert metrics["registered_detectors"] == 1

        # Act 2: Run tick with no snapshot available (should not increment ticks_processed)
        await detector._run_tick()

        # Assert 2: ticks_processed remains 1 (as no snapshot means the core logic didn't run)
        metrics = detector.get_metrics()
        assert metrics["ticks_processed"] == 1
        assert metrics["anomalies_detected"] == 1

        # Assert 3: set_anomaly called only once (during the first tick)
        cast(AsyncMock, mock_blackboard.set_anomaly).assert_called_once()

        event = cast(AsyncMock, mock_blackboard.set_anomaly).call_args[0][0]
        assert event.key == "always_true"
        assert event.severity == AnomalySeverity.HIGH
        assert event.score == 0.9

        cast(AsyncMock, mock_blackboard.clear_anomaly).assert_not_called()

    # --- Test 8: Clearing Logic ---

    async def test_clearing_logic_removes_inactive_flags(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange 1: First tick reports 'active_a' and 'active_b'.
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot

        detector = AnomalyDetector(mock_blackboard)
        # Register two detectors for the first tick
        detector.register_detector(MockDetectorA("active_a", 0.9, "high"))
        detector.register_detector(MockDetectorB("active_b", 0.7, "medium"))

        # Act 1: Run first tick (sets 'active_a', 'active_b')
        await detector._run_tick()

        # Check setup state and reset mock
        calls_t1 = cast(AsyncMock, mock_blackboard.set_anomaly).call_args_list
        assert len(calls_t1) == 2
        assert len(detector._active_anomaly_keys) == 2
        cast(AsyncMock, mock_blackboard.set_anomaly).reset_mock()
        cast(AsyncMock, mock_blackboard.clear_anomaly).reset_mock()

        # Arrange 2: Remove the detector for 'active_b' for the second tick.
        # Now only 'active_a' will be reported.
        # Note: We must update the _sub_detectors list on the detector instance
        detector._sub_detectors = [
            d for d in detector._sub_detectors if d.__class__.__name__ == "MockDetectorA"
        ]

        # Act 2: Run second tick (should set 'active_a' and clear 'active_b')
        await detector._run_tick()

        # Assert 1: Only one set_anomaly call (for 'active_a')
        cast(AsyncMock, mock_blackboard.set_anomaly).assert_called_once()
        event = cast(AsyncMock, mock_blackboard.set_anomaly).call_args[0][0]
        assert event.key == "active_a"

        # Assert 2: One clear_anomaly call (for 'active_b')
        cast(AsyncMock, mock_blackboard.clear_anomaly).assert_called_once_with("active_b")

        # Assert 3: State is updated
        assert len(detector._active_anomaly_keys) == 1
