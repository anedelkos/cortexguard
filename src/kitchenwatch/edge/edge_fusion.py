import logging
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any

from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.plan import IntentContext
from kitchenwatch.simulation.models.windowed_fused_record import WindowedFusedRecord

# Default smoothing factor for EMA (0 < alpha <= 1)
# Lower alpha = more smoothing, higher alpha = more responsive
DEFAULT_ALPHA = 0.1

logger = logging.getLogger(__name__)


class EdgeFusion:
    """
    Online sensor fusion using Exponential Moving Average (EMA).

    Processes windowed sensor records and maintains smoothed state via EMA.
    Each sample in a window updates the EMA sequentially, giving recency bias
    within the window itself.

    EMA Formula: new_ema = α * observation + (1-α) * old_ema
    - α near 0: Heavy smoothing (slow response)
    - α near 1: Light smoothing (fast response)

    Production Considerations (not implemented):
    - Adaptive alpha based on signal characteristics
    - Outlier rejection before EMA update
    - Multi-rate fusion (different α per sensor)
    - State persistence for warm restarts
    - Kalman filtering for better noise handling

    Design Choice: EMA over Kalman
    - EMA: Simple, low compute, no model required
    - Kalman: More accurate but needs process/measurement models
    - For edge deployment, EMA sufficient given stable sensors
    """

    def __init__(
        self,
        blackboard: Blackboard,
        alpha: float = DEFAULT_ALPHA,
        custom_logger: logging.Logger | None = None,
    ):
        """
        Initialize sensor fusion engine.

        Args:
            blackboard: Shared state for intent and snapshot updates
            alpha: EMA smoothing factor (0 < alpha <= 1)
            custom_logger: Optional logger instance

        Raises:
            ValueError: If alpha not in valid range
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"Alpha must be in (0, 1], got {alpha}")

        self._blackboard = blackboard
        self._alpha = alpha
        self._logger = custom_logger or logger

        # EMA state: sensor_key -> smoothed_value
        self._ema_state: dict[str, float] = {}
        self._last_snapshot: FusionSnapshot | None = None

        # Metrics (for observability)
        self._samples_processed = 0
        self._records_processed = 0

    async def process_record(self, record: WindowedFusedRecord) -> FusionSnapshot:
        """
        Process a windowed sensor record and update fusion state.

        Strategy: Applies EMA to each sample in window sequentially.
        This gives temporal ordering within the window - later samples
        influence the EMA more than earlier ones.

        Args:
            record: Windowed fused sensor data from simulator

        Returns:
            FusionSnapshot with smoothed sensor values and derived features
        """
        # Fetch current intent context
        current_intent: IntentContext | None = await self._blackboard.get_intent()

        # Process each sample in the window
        for sample in record.sensor_window:
            self._update_ema_state(sample)
            self._samples_processed += 1

        self._records_processed += 1

        # Create snapshot from current EMA state
        snapshot = FusionSnapshot(
            timestamp=datetime.fromtimestamp(record.timestamp_ns / 1e9),
            sensors={
                "raw": record.sensor_window,
                "ema_smoothed": self._ema_state.copy(),
                "window_stats": self._compute_window_stats(record.sensor_window),
            },
            derived=self._ema_state.copy(),  # Smoothed values as derived features
            intent=current_intent.action if current_intent else None,
        )

        self._last_snapshot = snapshot
        await self._blackboard.update_fusion_snapshot(snapshot)

        self._logger.debug(
            f"Updated snapshot for intent='{snapshot.intent}' "
            f"(window_size={len(record.sensor_window)})"
        )

        return snapshot

    def _update_ema_state(self, sample: Any) -> None:
        """
        Update EMA state with a single sensor sample.

        Skips:
        - Timestamp fields
        - None values
        - Non-numeric values

        Args:
            sample: Single sensor reading (Pydantic model)
        """
        for key, value in sample.model_dump().items():
            # Skip metadata and null values
            if key == "timestamp_ns" or value is None:
                continue

            # Only process numeric values (int, float)
            if not isinstance(value, (int, float)):
                self._logger.debug(f"Skipping non-numeric field: {key}={type(value).__name__}")
                continue

            # Initialize or update EMA
            if key not in self._ema_state:
                # First observation: initialize EMA
                self._ema_state[key] = float(value)
            else:
                # Subsequent: apply EMA formula
                # new = α * observation + (1-α) * old
                old_ema = self._ema_state[key]
                self._ema_state[key] = self._alpha * float(value) + (1 - self._alpha) * old_ema

    def reset_ema_state(self) -> None:
        """
        Reset EMA state to initial conditions.

        Useful for:
        - Testing/debugging
        - Switching between different operating modes
        - Recovery from anomalous sensor behavior
        """
        self._ema_state.clear()
        self._last_snapshot = None
        self._samples_processed = 0
        self._records_processed = 0
        self._logger.info("EMA state reset")

    def get_ema_state(self) -> dict[str, float]:
        """Get current EMA state (for debugging/monitoring)."""
        return self._ema_state.copy()

    def get_metrics(self) -> dict[str, int]:
        """Get processing metrics."""
        return {
            "records_processed": self._records_processed,
            "samples_processed": self._samples_processed,
            "ema_state_size": len(self._ema_state),
        }

    def _compute_window_stats(self, window: list[Any]) -> dict[str, dict[str, float]]:
        """
        Compute statistical features over the sensor window.

        Production Use: These features could feed into anomaly detection
        or be used as input to ML models.

        Returns:
            Dict mapping sensor_key -> {mean, std, min, max, range}
        """
        stats: dict[str, list[float]] = defaultdict(list)

        for sample in window:
            for key, value in sample.model_dump().items():
                if key != "timestamp_ns" and isinstance(value, (int, float)):
                    stats[key].append(float(value))

        result: dict[str, dict[str, float]] = {}
        for key, values in stats.items():
            if values:
                result[key] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                }

        return result
