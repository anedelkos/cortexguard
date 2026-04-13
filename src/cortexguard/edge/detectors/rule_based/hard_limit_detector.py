from __future__ import annotations

import logging
from typing import Any

from cortexguard.core.interfaces.base_detector import BaseDetector
from cortexguard.edge.models.anomaly_event import AnomalySeverity
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot

default_logger = logging.getLogger(__name__)


class HardLimitDetector(BaseDetector):
    """
    Detects critical safety violations based on rigid, deterministic thresholds.

    Primarily addresses Tier 0 Safety scenarios where ML models are too slow
    or probabilistic, such as Scenario S0.2 (Overheat + Smoke).

    Logic:
    - Monitoring raw sensor values from the FusionSnapshot.
    - Triggers 'high' severity anomalies if values exceed hard-coded safety limits.
    """

    # --- CLASS-LEVEL CONSTANTS (Configuration) ---

    # Critical temperature threshold in Celsius (Scenario S0.2)
    DEFAULT_TEMP_THRESHOLD_C: float = 70.0

    # Keys expected in the FusionSnapshot.sensors dictionary
    # In production, these might map to specific hardware IDs
    KEY_TEMP_SENSOR: str = "temp_celsius"
    KEY_SMOKE_SENSOR: str = "smoke_detected"

    # ------------------------------------------------------------

    def __init__(
        self,
        logger: logging.Logger = default_logger,
        temp_threshold: float | None = None,
    ) -> None:
        """
        Initialize the HardLimitDetector.

        Args:
            logger: Dependency-injected logger for observability.
            temp_threshold: Optional override for temperature limit.
                            Defaults to DEFAULT_TEMP_THRESHOLD_C.
        """
        self._logger = logger
        self._temp_threshold = (
            temp_threshold if temp_threshold is not None else self.DEFAULT_TEMP_THRESHOLD_C
        )

        self._logger.debug(f"HardLimitDetector initialized with threshold={self._temp_threshold}°C")

    async def detect(self, snapshot: FusionSnapshot) -> dict[str, Any]:
        """
        Checks raw sensor data against safety thresholds.

        Returns:
            Anomaly dictionary if a limit is breached, otherwise empty dict.
        """

        # Use raw sensor values to avoid EMA phase delay.
        sensors = snapshot.sensors

        current_temp = sensors.get(self.KEY_TEMP_SENSOR)
        smoke_flag = sensors.get(self.KEY_SMOKE_SENSOR)

        if current_temp is None:
            # Only log periodically or once in production to avoid log spam
            return {}

        # Ensure types are correct before comparison (defensive)
        if not isinstance(current_temp, (int, float)):
            self._logger.error(
                f"Invalid type for temp sensor: {type(current_temp)}. Expected float."
            )
            return {}

        # Critical Logic: Overheat + Smoke (S0.2)
        # We prioritize the compound scenario first as it is Tier 0.
        is_overheating = current_temp > self._temp_threshold
        is_smoking = bool(smoke_flag)

        if is_overheating and is_smoking:
            self._logger.critical(
                f"🔥 FIRE DETECTED! Temp: {current_temp}°C > {self._temp_threshold}°C "
                f"AND Smoke Detected."
            )

            return self._build_anomaly_event(
                key="overheat_smoke_combo",
                severity=AnomalySeverity.HIGH,
                score=1.0,  # Certainty is 100% for hard limits
                metadata={"temp": current_temp, "threshold": self._temp_threshold, "smoke": True},
            )

        # Smoke alone — no overheat, but smoke is an explicit stop condition
        if is_smoking:
            self._logger.critical(
                f"🚨 Smoke detected at {current_temp}°C (below overheat threshold)."
            )
            return self._build_anomaly_event(
                key="smoke_only",
                severity=AnomalySeverity.HIGH,
                score=1.0,
                metadata={"temp": current_temp, "smoke": True},
            )

        # Fallback Logic: Just Overheat (Tier 1/2 Warning)
        if is_overheating:
            self._logger.warning(
                f"⚠️ Overheat Warning. Temp: {current_temp}°C > {self._temp_threshold}°C"
            )

            # Calculate a linear score based on how far over threshold we are
            # e.g., 77C vs 70C limit -> score ~0.1
            overshoot = current_temp - self._temp_threshold
            score = min(1.0, overshoot / 10.0)  # Cap at 1.0 for 10 degrees over

            return self._build_anomaly_event(
                key="temp_limit_breach",
                severity=AnomalySeverity.MEDIUM,  # Medium severity allows local cooling, not E-STOP
                score=score,
                metadata={"temp": current_temp, "threshold": self._temp_threshold, "smoke": False},
            )

        return {}

    def _build_anomaly_event(
        self, key: str, severity: str, score: float, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Helper to construct the dictionary safely."""
        try:
            return {
                "key": key,
                "anomaly_score": score,
                "severity": severity,
                "metadata": metadata,
            }
        except Exception as e:
            self._logger.error(f"Failed to construct anomaly event: {e}", exc_info=True)
            return {}
