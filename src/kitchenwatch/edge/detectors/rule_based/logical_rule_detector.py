import logging
from datetime import datetime
from typing import Any

from kitchenwatch.core.interfaces.base_detector import BaseDetector
from kitchenwatch.edge.models.anomaly_severity import AnomalySeverity
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot

# Fallback logger if none is injected
default_logger = logging.getLogger(__name__)


class LogicalRuleDetector(BaseDetector):
    """
    Detects anomalies based on temporal sequence and logical state rules.

    This detector is responsible for detecting complex, stateful safety scenarios
    that require historical context, such as:
    - S1.1: Repeated system failures (e.g., Misgrasp Count)
    - S2.3: Sensor/Blackboard data freeze (Lack of updates)
    """

    # --- CLASS-LEVEL CONSTANTS (Configuration) ---

    # S1.1: Number of consecutive failures before a MEDIUM anomaly is raised.
    MAX_CONSECUTIVE_FAILURES: int = 3

    # S2.3: Time in seconds without a new snapshot timestamp before a HIGH anomaly is raised.
    FREEZE_TIME_LIMIT_S: float = 2.0

    # Key expected in the FusionSnapshot.derived dictionary to track a critical metric status.
    # In this case, we track the success/failure of the physical grasp action.
    KEY_SYSTEM_SUCCESS_STATUS: str = "grasp_success"
    # The value that signifies success (Failure is anything else, usually False)
    VALUE_SYSTEM_SUCCESS: bool = True

    # ------------------------------------------------------------

    def __init__(
        self,
        logger: logging.Logger,
        max_failures: int | None = None,
        freeze_limit_s: float | None = None,
    ) -> None:
        """
        Initialize the LogicalRuleDetector with state.
        """
        self._logger = logger
        self._max_consecutive_failures = max_failures or self.MAX_CONSECUTIVE_FAILURES
        self._freeze_limit_s = freeze_limit_s or self.FREEZE_TIME_LIMIT_S

        # --- Internal State ---
        self._consecutive_failure_count: int = 0
        # Tracks when this detector last received and processed a tick.
        self._last_detector_tick: datetime = datetime.min

        self._logger.debug(
            f"LogicalRuleDetector initialized with consecutive failure limit={self._max_consecutive_failures}, "
            f"freeze limit={self._freeze_limit_s}s"
        )

    async def detect(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        """
        Checks for temporal and logical rule violations.
        """
        current_time = datetime.now()

        # Update the detector's internal tick time immediately.
        # This prevents the data freeze check from self-reporting a freeze
        # if other rules cause the function to return early.
        previous_tick_time = self._last_detector_tick
        self._last_detector_tick = current_time

        # 1. Check for data freeze (S2.3) - High Priority
        freeze_anomaly = self._check_data_freeze(previous_tick_time, current_time)
        if freeze_anomaly:
            return freeze_anomaly

        # 2. Check for repeated failures (S1.1)
        consecutive_failure_anomaly = self._check_consecutive_failure(snapshot)
        if consecutive_failure_anomaly:
            return consecutive_failure_anomaly

        return {}

    def _check_data_freeze(
        self, previous_tick_time: datetime, current_time: datetime
    ) -> dict[str, Any]:
        """
        Checks if the time elapsed since the last successful detector tick
        exceeds the freeze limit (S2.3).
        """
        # On the first ever run, previous_tick_time will be datetime.min, so we skip the check.
        if previous_tick_time == datetime.min:
            return {}

        time_since_last_update = current_time - previous_tick_time

        if time_since_last_update.total_seconds() > self._freeze_limit_s:
            self._logger.critical(
                f"🚨 DETECTOR FREEZE DETECTED: No update in "
                f"{time_since_last_update.total_seconds():.2f}s "
                f"(Limit: {self._freeze_limit_s}s). This implies the AnomalyDetector "
                f"tick loop is not running consistently."
            )

            # NOTE: We do NOT update the internal state here, as the state was updated
            # at the start of the detect method, ensuring the next run starts clean.

            return self._build_anomaly_event(
                key="blackboard_data_freeze",
                severity=AnomalySeverity.HIGH,
                score=1.0,
                metadata={
                    "time_since_last_update_s": time_since_last_update.total_seconds(),
                    "limit_s": self._freeze_limit_s,
                },
            )

        return {}

    def _check_consecutive_failure(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        """
        Tracks consecutive failures of a critical system metric (S1.1).
        """
        success_status = snapshot.derived.get(self.KEY_SYSTEM_SUCCESS_STATUS)

        # If the key is missing, the perception model hasn't run yet, so we hold state.
        if success_status is None:
            return {}

        if success_status == self.VALUE_SYSTEM_SUCCESS:
            # Success: Reset the counter
            if self._consecutive_failure_count > 0:
                self._logger.info(
                    f"System success reported. Resetting failure count from {self._consecutive_failure_count}."
                )
            self._consecutive_failure_count = 0
            return {}

        # Failure: Increment counter
        self._consecutive_failure_count += 1
        self._logger.warning(
            f"System failure detected via derived metric '{self.KEY_SYSTEM_SUCCESS_STATUS}'. "
            f"Count: {self._consecutive_failure_count} / {self._max_consecutive_failures}"
        )

        if self._consecutive_failure_count >= self._max_consecutive_failures:
            self._logger.error(
                f"🛑 CONSECUTIVE FAILURE (S1.1): Count reached {self._max_consecutive_failures}. "
                f"Requesting system pause."
            )

            # Severity is MEDIUM: requires a system pause/retry, not a full E-STOP.
            return self._build_anomaly_event(
                key="repeated_system_failure",
                severity=AnomalySeverity.MEDIUM,
                score=min(1.0, self._consecutive_failure_count / self._max_consecutive_failures),
                metadata={
                    "failure_key": self.KEY_SYSTEM_SUCCESS_STATUS,
                    "failure_count": self._consecutive_failure_count,
                    "limit": self._max_consecutive_failures,
                },
            )

        return {}

    def _build_anomaly_event(
        self, key: str, severity: AnomalySeverity, score: float, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Helper to construct the dictionary safely."""
        try:
            return {
                "key": key,
                "anomaly_score": score,
                "severity": severity.value,  # Use the string value for the output contract
                "metadata": metadata,
            }
        except Exception as e:
            self._logger.error(f"Failed to construct anomaly event: {e}", exc_info=True)
            return {}
