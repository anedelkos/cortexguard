import asyncio
import logging
from typing import Any, cast

from kitchenwatch.core.interfaces.base_detector import BaseDetector
from kitchenwatch.edge.models.anomaly_severity import AnomalySeverity
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Ensemble coordinator for edge anomaly detection.

    Architecture:
    - Polls Blackboard for latest FusionSnapshots
    - Distributes snapshots to registered sub-detectors
    - Aggregates detection results using ensemble voting
    - Posts anomaly flags back to Blackboard

    Design Rationale:
    - Ensemble approach combines multiple detection strategies
    - Async execution allows concurrent detector processing
    - Pluggable sub-detectors enable easy experimentation
    - Fail-fast: Individual detector failures don't halt system

    Production Enhancements (not implemented):
    - Weighted voting based on detector confidence/accuracy
    - Temporal smoothing to reduce false positives (e.g., 3/5 consecutive)
    - Adaptive thresholds based on historical data
    - Circuit breaker for consistently failing detectors
    - Metrics export to Prometheus/CloudWatch
    - Detector performance tracking (latency, accuracy)
    """

    def __init__(
        self,
        blackboard: Blackboard,
        tick_interval: float = 0.1,
        custom_logger: logging.Logger = logger,
    ):
        """
        Initialize anomaly detector ensemble.

        Args:
            blackboard: Shared state for snapshot retrieval and flag updates
            tick_interval: Polling interval in seconds (default 100ms)
            custom_logger: Optional logger instance
        """
        self._blackboard = blackboard
        self._tick_interval = tick_interval
        self._logger = custom_logger

        self._sub_detectors: list[BaseDetector] = []
        self._loop_running = False
        self._task: asyncio.Task[Any] | None = None

        # State to manage which keys are currently active on the Blackboard.
        # This is essential for clearing flags that are no longer reported.
        self._active_anomaly_keys: set[str] = set()

        # Metrics for observability
        self._ticks_processed = 0
        self._anomalies_detected = 0
        self._detector_failures = 0

    def register_detector(self, detector: BaseDetector) -> None:
        """
        Register a sub-detector to the ensemble.

        Args:
            detector: SubDetector instance implementing the detection protocol
        """
        self._sub_detectors.append(detector)
        self._logger.info(f"Registered sub-detector: {detector.__class__.__name__}")

    async def _poll_blackboard(self) -> FusionSnapshot | None:
        """
        Fetch latest fused sensor snapshot from blackboard.

        Returns:
            Latest FusionSnapshot or None if unavailable
        """
        snapshot = await self._blackboard.get_fusion_snapshot()
        if snapshot is None:
            self._logger.debug("No fusion snapshot available yet")
        return snapshot

    async def _run_tick(self) -> None:
        """
        Execute one detection cycle.

        Process:
        1. Poll blackboard for latest snapshot
        2. Run all sub-detectors concurrently
        3. Aggregate results using ensemble logic
        4. Update blackboard anomaly flags
        """
        snapshot = await self._poll_blackboard()

        # 1. Handle no snapshot available.
        if snapshot is None:
            # We don't increment the counter here, as no work was done.
            return

        # 2. Process the available snapshot.
        try:
            # Run detectors concurrently for lower latency
            tasks = [detector.detect(snapshot) for detector in self._sub_detectors]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and log errors
            valid_results: list[dict[str, Any]] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    detector = self._sub_detectors[i]

                    # NOTE: Keeping .error(..., exc_info=) is functionally correct
                    # but requires the test suite update below.
                    self._logger.error(
                        f"Sub-detector {detector.__class__.__name__} failed",
                        exc_info=result,
                    )
                    self._detector_failures += 1
                else:
                    valid_results.append(cast(dict[str, Any], result))

            # Aggregate and post results
            if valid_results:
                # Returns only currently active anomalies: {key: (severity, True)}
                active_aggregated_flags = self._aggregate_results(valid_results)

                newly_active_keys = set(active_aggregated_flags.keys())

                # --- Anomaly Clearing Logic ---
                # Step 3: Clear flags that were active last tick but are NOT active now.
                keys_to_clear = self._active_anomaly_keys - newly_active_keys
                for key in keys_to_clear:
                    # Explicitly set is_active=False to clear the flag on the Blackboard
                    # Severity is irrelevant for clearing, but a placeholder is needed for the signature.
                    await self._blackboard.set_anomaly_flag(
                        key, AnomalySeverity.LOW, is_active=False
                    )

                # Step 4: Set/Update currently active flags
                for key, (severity, is_active) in active_aggregated_flags.items():
                    # is_active will always be True here, and we pass the required severity
                    await self._blackboard.set_anomaly_flag(key, severity, is_active)

                # Step 5: Update the state tracker for the next tick
                self._active_anomaly_keys = newly_active_keys

                # Track metrics
                if newly_active_keys:
                    self._anomalies_detected += 1
                    self._logger.info(f"Anomalies detected: {list(newly_active_keys)}")

        except Exception as e:
            # Catch any unexpected errors in aggregation or posting
            self._logger.exception(f"Unexpected error during tick processing: {e}")

        finally:
            # Increment tick counter only after processing work (snapshot was available)
            self._ticks_processed += 1

    def _aggregate_results(
        self, results: list[dict[str, Any]]
    ) -> dict[str, tuple[AnomalySeverity, bool]]:
        """
        Aggregate sub-detector results into (Severity, is_active=True) flags for currently active anomalies.

        Strategy: Flag as anomalous if ANY detector reports medium/high severity.
        The severity recorded is the HIGHEST reported severity that triggered the flag.

        Args:
            results: List of detection results from sub-detectors

        Returns:
            Dictionary mapping anomaly keys to a tuple: (AnomalySeverity, bool is_active).
            Only returns keys that are currently active (is_active=True).
        """
        # Map: Anomaly Key -> (Highest Triggering Severity, is_active)
        aggregated: dict[str, tuple[AnomalySeverity, bool]] = {}

        # Severity ranking for comparison (HIGH > MEDIUM > LOW)
        SEVERITY_RANKING = {
            AnomalySeverity.LOW: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.HIGH: 3,
        }

        for result in results:
            key = result.get("key", "generic_anomaly")

            # Ensure severity is the AnomalySeverity enum type
            severity_str = result.get("severity", AnomalySeverity.LOW)
            # NOTE: Assuming AnomalySeverity can be constructed from a string
            # and that results['severity'] contains the appropriate string representation.
            try:
                severity = AnomalySeverity(severity_str)
            except ValueError:
                self._logger.warning(
                    f"Invalid severity string '{severity_str}' received for key '{key}'"
                )
                severity = AnomalySeverity.LOW  # Default to LOW on bad data

            # Flag as anomalous if medium or high severity (detectors only report if it IS anomalous)
            is_anomalous = severity in (AnomalySeverity.MEDIUM, AnomalySeverity.HIGH)

            # Since a detector only reports a result if an anomaly is present,
            # we only process if the result indicates an actual anomaly.
            if is_anomalous:
                # Get current aggregated state for this key, default to LOW/False if not seen yet
                current_severity, current_is_active = aggregated.get(
                    key, (AnomalySeverity.LOW, False)
                )

                # The flag is active: is_active=True
                new_is_active = True

                # Determine the highest severity seen for this key
                highest_severity = current_severity

                if SEVERITY_RANKING[severity] > SEVERITY_RANKING[current_severity]:
                    highest_severity = severity

                # Only store active flags
                aggregated[key] = (highest_severity, new_is_active)

        return aggregated

    async def _run_loop(self) -> None:
        """Main detection loop with graceful error handling."""
        self._loop_running = True
        self._logger.info(
            f"AnomalyDetector loop started "
            f"(detectors={len(self._sub_detectors)}, "
            f"tick_interval={self._tick_interval}s)"
        )

        try:
            while self._loop_running:
                await self._run_tick()
                await asyncio.sleep(self._tick_interval)

        except asyncio.CancelledError:
            self._logger.info("AnomalyDetector loop cancelled gracefully")
            raise
        except Exception as e:
            self._logger.exception(f"Fatal error in AnomalyDetector loop: {e}")
            raise
        finally:
            self._loop_running = False
            self._logger.info(
                f"AnomalyDetector stopped "
                f"(ticks={self._ticks_processed}, "
                f"anomalies={self._anomalies_detected}, "
                f"failures={self._detector_failures})"
            )

    async def start(self) -> None:
        """Start the anomaly detection loop."""
        if self._loop_running:
            self._logger.warning("AnomalyDetector already running")
            return

        if self._task and not self._task.done():
            self._logger.warning("Previous task still active")
            return

        if not self._sub_detectors:
            self._logger.warning("No sub-detectors registered, starting anyway")

        self._task = asyncio.create_task(self._run_loop())
        self._logger.info("AnomalyDetector started")

    async def stop(self) -> None:
        """Stop the anomaly detection loop gracefully."""
        self._logger.info("Stopping AnomalyDetector...")
        self._loop_running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                self._logger.debug("AnomalyDetector task cancelled")

        self._logger.info("AnomalyDetector stopped")

    def get_metrics(self) -> dict[str, int | float]:
        """
        Get detector metrics for monitoring.

        Returns:
            Dictionary with operational metrics
        """
        return {
            "ticks_processed": self._ticks_processed,
            "anomalies_detected": self._anomalies_detected,
            "detector_failures": self._detector_failures,
            "registered_detectors": len(self._sub_detectors),
            "failure_rate": (
                self._detector_failures / (self._ticks_processed * len(self._sub_detectors))
                if self._ticks_processed > 0 and self._sub_detectors
                else 0.0
            ),
        }
