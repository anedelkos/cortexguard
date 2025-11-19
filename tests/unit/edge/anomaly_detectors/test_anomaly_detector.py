import asyncio
import logging
from datetime import datetime  # ADDED: Required for FusionSnapshot timestamp
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

# Assuming these imports are correct based on your project structure
from kitchenwatch.core.interfaces.base_detector import BaseDetector
from kitchenwatch.edge.detectors.anomaly_detector import AnomalyDetector
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot


# Define a mock detector to use in the tests
class MockDetector(BaseDetector):
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
            "anomaly_score": self.score,
            "severity": self.severity,
            "metadata": {"processed_at": snapshot.timestamp},
        }


# Define a detector that intentionally fails (to test exception handling)
class FailingDetector(BaseDetector):
    async def detect(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        raise RuntimeError("Detector failed intentionally")


@pytest.fixture
def mock_blackboard() -> Blackboard:
    """
    Fixture for a mocked Blackboard.
    """
    # Create a mock object that implements the Blackboard interface
    mock_bb = MagicMock(spec=Blackboard)

    # Assign AsyncMocks to the coroutine methods on the mock object
    # Mypy now accepts this assignment because mock_bb is a mock, not a live instance.
    mock_bb.get_fusion_snapshot = AsyncMock()
    mock_bb.set_anomaly_flag = AsyncMock()

    # We must explicitly set the return type to Blackboard, though it's a mock.
    return cast(Blackboard, mock_bb)


@pytest.fixture
def mock_snapshot() -> FusionSnapshot:
    """Fixture for a mock FusionSnapshot."""
    # FIX: Use datetime.now() instead of float to satisfy FusionSnapshot type hint
    return FusionSnapshot(timestamp=datetime.now(), derived={}, sensors={}, intent=None)


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

        mock_sub_detector = MockDetector("test", 0.5, "low")
        detector.register_detector(mock_sub_detector)

        assert len(detector._sub_detectors) == 1
        assert detector._sub_detectors[0] is mock_sub_detector

    # --- Test 2: Concurrent Execution and Latency ---

    async def test_concurrent_execution_is_faster_than_sequential(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange: Setup 3 detectors that each take 0.1s
        # FIX: Cast before accessing return_value
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot

        detector = AnomalyDetector(mock_blackboard)
        detector.register_detector(MockDetector("d1", 0.9, "high", sleep_time=0.1))
        detector.register_detector(MockDetector("d2", 0.7, "medium", sleep_time=0.1))
        detector.register_detector(MockDetector("d3", 0.5, "low", sleep_time=0.1))

        start_time = asyncio.get_event_loop().time()

        # Act: Run one tick
        await detector._run_tick()

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        # Assert: Duration should be close to 0.1s (concurrent), not 0.3s (sequential)
        # Allow a small buffer (e.g., 50ms) for execution overhead
        assert duration == pytest.approx(0.1, abs=0.05)

        # --- Test 3: Result Aggregation Logic ---

    async def test_aggregation_strategy_medium_and_high_trigger_flag(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange: Setup detectors with different severities
        # FIX: Cast before accessing return_value
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot

        detector = AnomalyDetector(mock_blackboard)
        # d1: low severity -> False
        detector.register_detector(MockDetector("low_risk", 0.2, "low"))
        # d2: medium severity -> True
        detector.register_detector(MockDetector("med_risk", 0.6, "medium"))
        # d3: high severity -> True
        detector.register_detector(MockDetector("high_risk", 0.9, "high"))

        # Act: Run one tick
        await detector._run_tick()

        # Assert 1: set_anomaly_flag called three times for unique keys
        # FIX: Cast before accessing call_args_list
        calls = cast(AsyncMock, mock_blackboard.set_anomaly_flag).call_args_list
        assert len(calls) == 3

        # Assert 2: Verify boolean flags based on severity logic
        flag_map = {args[0][0]: args[0][1] for args in calls}
        assert flag_map["low_risk"] is False
        assert flag_map["med_risk"] is True
        assert flag_map["high_risk"] is True

    # --- Test 4: OR Logic in Aggregation ---

    async def test_aggregation_or_logic_for_duplicate_keys(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange: Setup two detectors using the SAME key
        # FIX: Cast before accessing return_value
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot

        detector = AnomalyDetector(mock_blackboard)
        # d1: high severity (True)
        detector.register_detector(MockDetector("collision", 0.9, "high"))
        # d2: low severity (False)
        detector.register_detector(MockDetector("collision", 0.3, "low"))

        # Act: Run one tick
        await detector._run_tick()

        # Assert: set_anomaly_flag called only ONCE for the key, and the result is True (High OR Low = True)
        # FIX: Cast before accessing assert_called_once_with
        cast(AsyncMock, mock_blackboard.set_anomaly_flag).assert_called_once_with("collision", True)

    # --- Test 5: Exception Handling ---

    async def test_failing_detector_does_not_halt_ensemble(
        self,
        mock_blackboard: Blackboard,
        mock_snapshot: FusionSnapshot,
        test_logger: logging.Logger,
    ) -> None:
        # Arrange: Use MagicMock to capture logging calls
        mock_logger = MagicMock(spec=logging.Logger)
        # FIX: Cast before accessing return_value
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot

        detector = AnomalyDetector(mock_blackboard, custom_logger=mock_logger)
        # d1: High risk detector (successful)
        detector.register_detector(MockDetector("critical", 0.9, "high"))
        # d2: Detector that raises an exception
        detector.register_detector(FailingDetector())

        # Act: Run one tick
        await detector._run_tick()

        # Assert 1: The successful detector result was posted (critical: True)
        # FIX: Cast before accessing assert_called_once_with
        cast(AsyncMock, mock_blackboard.set_anomaly_flag).assert_called_once_with("critical", True)

        # Assert 2: An exception was logged, but the tick completed successfully
        mock_logger.error.assert_called_once()
        assert "Sub-detector FailingDetector failed" in mock_logger.error.call_args[0][0]

    # --- Test 6: Lifecycle Management ---

    async def test_start_stop_lifecycle(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange: Setup the detector with a low tick interval for quick test run
        # FIX: Cast before accessing return_value
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).return_value = mock_snapshot
        detector = AnomalyDetector(mock_blackboard, tick_interval=0.01)
        detector.register_detector(MockDetector("a", 0.9, "high"))

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
        assert detector._task is not None  # Added check to satisfy mypy's possible None
        assert detector._task.done() is True
        assert detector._task.cancelled() is True

        # Assert 3: Subsequent stop call is harmless
        await detector.stop()

    # --- Test 7: Polling and Metrics ---

    async def test_polling_and_metrics_update(
        self, mock_blackboard: Blackboard, mock_snapshot: FusionSnapshot
    ) -> None:
        # Arrange: Simulate one successful snapshot, followed by None
        # FIX: Cast before accessing side_effect
        cast(AsyncMock, mock_blackboard.get_fusion_snapshot).side_effect = [mock_snapshot, None]

        detector = AnomalyDetector(mock_blackboard)
        # Register a detector that always flags TRUE for metric testing
        detector.register_detector(MockDetector("always_true", 0.9, "high"))

        # Act 1: Run tick with snapshot available
        await detector._run_tick()

        # Assert 1: Metrics updated after successful tick
        metrics = detector.get_metrics()
        assert metrics["ticks_processed"] == 1
        assert metrics["anomalies_detected"] == 1
        assert metrics["registered_detectors"] == 1

        # Act 2: Run tick with no snapshot available
        await detector._run_tick()

        # Assert 2: ticks_processed remains 1 because the second tick found NO snapshot (Logical Fix)
        metrics = detector.get_metrics()
        assert metrics["ticks_processed"] == 1  # LOGICAL FIX APPLIED
        assert metrics["anomalies_detected"] == 1

        # Assert 3: set_anomaly_flag called only once (during the first tick)
        cast(AsyncMock, mock_blackboard.set_anomaly_flag).assert_called_once()
