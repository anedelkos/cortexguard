import collections
import logging
from typing import Any

from kitchenwatch.core.interfaces.base_detector import BaseDetector
from kitchenwatch.edge.models.anomaly_event import AnomalySeverity
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot

logger = logging.getLogger(__name__)


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
    _MAX_CONSECUTIVE_FAILURES: int = 3

    # S2.3
    _WINDOW_SIZE: int = 5
    _PERSISTENCE_S: float = 1.0
    _VARIANCE_EPSILON: dict[str, float] = {"temp_celsius": 0.01, "force_n": 0.05}
    _SNAPSHOT_HISTORY_K: int = 3

    # Key expected in the FusionSnapshot.derived dictionary to track a critical metric status.
    # In this case, we track the success/failure of the physical grasp action.
    KEY_SYSTEM_SUCCESS_STATUS: str = "grasp_success"
    # The value that signifies success (Failure is anything else, usually False)
    VALUE_SYSTEM_SUCCESS: bool = True

    # ------------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        # --- Internal State ---
        self._consecutive_failure_count: int = 0
        self._recent_snapshot_values: dict[str, collections.deque[float]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self._SNAPSHOT_HISTORY_K)
        )

        logger.debug(
            f"LogicalRuleDetector initialized with consecutive failure limit={self._MAX_CONSECUTIVE_FAILURES}"
        )

    async def detect(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        """
        Checks for temporal and logical rule violations.
        """

        # 1. Check for data freeze (S2.3) - High Priority
        value_freeze_anomaly = self._check_value_freeze(snapshot)
        if value_freeze_anomaly:
            return value_freeze_anomaly

        # 2. Check for repeated failures (S1.1)
        consecutive_failure_anomaly = self._check_consecutive_failure(snapshot)
        if consecutive_failure_anomaly:
            return consecutive_failure_anomaly

        return {}

    def _check_value_freeze(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        sensors = snapshot.sensors
        window_stats = sensors["window_stats"]

        for key, stats in window_stats.items():
            rng = stats["range"]
            std = stats["std"]
            eps = self._VARIANCE_EPSILON.get(key, 1e-6)
            if rng <= eps or std <= eps:
                return self._build_anomaly_event(
                    key=f"{key}_value_freeze",
                    severity=AnomalySeverity.MEDIUM,
                    score=0.8,
                    metadata={"sensor_id": key, "window_range": rng, "window_std": std},
                )

        raw_window = sensors["raw"]
        per_key_values: dict[str, list[float]] = {}

        for sample in raw_window:
            for k, v in sample.model_dump().items():
                if isinstance(v, (int, float)):
                    per_key_values.setdefault(k, []).append(float(v))

        for key, values in per_key_values.items():
            if len(values) < max(1, self._WINDOW_SIZE // 2):
                continue
            eps = self._VARIANCE_EPSILON.get(key, 1e-6)
            rng = max(values) - min(values)
            if rng <= eps:
                return self._build_anomaly_event(
                    key=f"{key}_value_freeze",
                    severity=AnomalySeverity.MEDIUM,
                    score=0.8,
                    metadata={"sensor_id": key, "window_range": rng, "window_samples": len(values)},
                )

        for key in ("temp_celsius",):
            val = sensors[key]
            dq = self._recent_snapshot_values[key]
            if isinstance(val, bool):
                dq.append(1 if val else 0)
            elif isinstance(val, (int, float)):
                dq.append(float(val))
            else:
                continue

            if len(dq) == dq.maxlen:
                rng = max(dq) - min(dq)
                eps = self._VARIANCE_EPSILON.get(key, 1e-6)
                if rng <= eps:
                    return self._build_anomaly_event(
                        key=f"{key}_value_freeze",
                        severity=AnomalySeverity.MEDIUM,
                        score=0.8,
                        metadata={
                            "sensor_id": key,
                            "persistence_range": rng,
                            "persistence_samples": len(dq),
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
                logger.info(
                    f"System success reported. Resetting failure count from {self._consecutive_failure_count}."
                )
            self._consecutive_failure_count = 0
            return {}

        # Failure: Increment counter
        self._consecutive_failure_count += 1
        logger.warning(
            f"System failure detected via derived metric '{self.KEY_SYSTEM_SUCCESS_STATUS}'. "
            f"Count: {self._consecutive_failure_count} / {self._MAX_CONSECUTIVE_FAILURES}"
        )

        if self._consecutive_failure_count >= self._MAX_CONSECUTIVE_FAILURES:
            logger.error(
                f"🛑 CONSECUTIVE FAILURE (S1.1): Count reached {self._MAX_CONSECUTIVE_FAILURES}. "
                f"Requesting system pause."
            )

            # Severity is MEDIUM: requires a system pause/retry, not a full E-STOP.
            return self._build_anomaly_event(
                key="repeated_system_failure",
                severity=AnomalySeverity.MEDIUM,
                score=min(1.0, self._consecutive_failure_count / self._MAX_CONSECUTIVE_FAILURES),
                metadata={
                    "failure_key": self.KEY_SYSTEM_SUCCESS_STATUS,
                    "failure_count": self._consecutive_failure_count,
                    "limit": self._MAX_CONSECUTIVE_FAILURES,
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
            logger.error(f"Failed to construct anomaly event: {e}", exc_info=True)
            return {}
