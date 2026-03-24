from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from opentelemetry import trace

from cortexguard.core.interfaces.base_detector import BaseDetector
from cortexguard.edge.models.anomaly_event import SEVERITY_RANKING, AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot
from cortexguard.edge.models.reasoning_trace_entry import TraceSeverity
from cortexguard.edge.utils.metrics import (
    active_anomalies,
    anomalies_total,
    component_duration_ms,
    detector_failures_total,
)
from cortexguard.edge.utils.tracing import BaseTraceSink, TraceSink

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("cortexguard.anomaly_detector")


class AnomalyDetector:
    """Ensemble coordinator for edge anomaly detection."""

    def __init__(
        self,
        blackboard: Blackboard,
        trace_sink: BaseTraceSink | None = None,
        tick_interval: float = 0.1,
    ):
        """
        Initialize anomaly detector ensemble.

        Args:
            blackboard: Shared state for snapshot retrieval and flag updates
            tick_interval: Polling interval in seconds (default 100ms)
        """
        self._blackboard = blackboard
        self._trace_sink: BaseTraceSink = (
            trace_sink if trace_sink is not None else TraceSink(blackboard=self._blackboard)
        )
        self._tick_interval = tick_interval
        self._sub_detectors: list[BaseDetector] = []
        self._loop_running = False
        self._task: asyncio.Task[Any] | None = None

        # State to manage which keys are currently active on the Blackboard for clearing.
        self._active_anomaly_keys: set[str] = set()

        # Metrics for observability
        self._ticks_processed = 0
        self._anomalies_detected = 0
        self._detector_failures = 0

    def register_detector(self, detector: BaseDetector) -> None:
        """Register a sub-detector to the ensemble."""
        self._sub_detectors.append(detector)
        logger.info(f"Registered sub-detector: {detector.__class__.__name__}")

    async def _poll_blackboard(self) -> FusionSnapshot | None:
        """Fetch latest fused sensor snapshot from blackboard."""
        snapshot = await self._blackboard.get_fusion_snapshot()
        if snapshot is None:
            logger.debug("No fusion snapshot available yet")
        return snapshot

    async def _poll_blackboard_intent(self) -> str:
        """Fetch the current system intent (action string) from the blackboard."""
        current_step = await self._blackboard.get_current_step()
        if current_step:
            return current_step.description

        logger.debug("No active PlanStep or action found. Defaulting to 'Idle'")
        return "Idle"

    async def _run_detectors(self, snapshot: FusionSnapshot) -> list[tuple[str, Any]]:

        tasks = []

        for detector in self._sub_detectors:
            name = detector.__class__.__name__

            async def run_detector(detector: Any = detector, name: str = name) -> Any:
                with tracer.start_as_current_span("anomaly_detector.detector") as dspan:
                    dspan.set_attribute("detector.name", name)
                    return await detector.detect(snapshot)

            tasks.append(run_detector())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid = []

        for detector, result in zip(self._sub_detectors, results, strict=False):
            name = detector.__class__.__name__

            if isinstance(result, Exception):
                logger.error(f"Sub-detector {name} failed", exc_info=result)
                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="DETECTOR_FAILED",
                    reasoning_text=f"Sub-detector {name} raised an exception",
                    metadata={"detector": name, "error": str(result)},
                    severity=TraceSeverity.HIGH,
                )
                self._detector_failures += 1
                detector_failures_total.inc()
            elif result:
                valid.append((name, result))

        return valid

    async def _aggregate_and_emit(self, valid_results: list[tuple[str, Any]]) -> dict[str, Any]:
        active_flags = self._aggregate_results(valid_results)

        await self._trace_sink.post_trace_entry(
            source=self,
            event_type="AGGREGATION_SUMMARY",
            reasoning_text="Aggregation completed for tick",
            metadata={
                "keys_considered": len(valid_results),
                "anomalies_emitted": len(active_flags),
                "top_keys": list(active_flags.keys())[:5],
            },
        )

        return active_flags

    async def _clear_resolved_anomalies(self, new_keys: set[str]) -> None:
        keys_to_clear = self._active_anomaly_keys - new_keys

        for key in keys_to_clear:
            await self._blackboard.clear_anomaly(key)
            await self._trace_sink.post_trace_entry(
                source=self,
                event_type="ANOMALY_CLEARED",
                reasoning_text=f"Anomaly '{key}' resolved or no longer detected.",
                metadata={"anomaly_key": key},
                severity=TraceSeverity.INFO,
                refs={"anomaly_key": key},
            )

    async def _emit_and_store_active_anomalies(
        self,
        active_flags: dict[str, tuple[Any, float, list[str], dict[str, Any] | None]],
        snapshot: FusionSnapshot,
        current_intent: str | None,
    ) -> None:
        for key, (severity, score, detectors, flag_metadata) in active_flags.items():
            metadata = {
                "snapshot_id": snapshot.id,
                "current_intent": current_intent,
                "description": f"Automated detection of {key} (Severity: {severity.name}, Score: {score:.2f})",
                "contributing_detectors": detectors,
            }
            metadata = (flag_metadata or {}) | metadata

            event = AnomalyEvent(
                id=uuid.uuid4().hex,
                key=key,
                timestamp=snapshot.timestamp,
                severity=severity,
                score=score,
                contributing_detectors=detectors,
                metadata=metadata,
                window=None,
            )

            await self._blackboard.set_anomaly(event)

            if key not in self._active_anomaly_keys:
                anomalies_total.inc()

                trace_sev = (
                    TraceSeverity.CRITICAL
                    if severity == AnomalySeverity.HIGH
                    else (
                        TraceSeverity.HIGH
                        if severity == AnomalySeverity.MEDIUM
                        else TraceSeverity.WARN
                    )
                )

                await self._trace_sink.post_trace_entry(
                    source=self,
                    event_type="ANOMALY_TRIGGERED",
                    reasoning_text=f"New anomaly '{key}' detected. Severity: {severity.name}. Score: {score:.2f}.",
                    metadata={"anomaly_key": key, "severity": severity.name, "score": score},
                    severity=trace_sev,
                    refs={"anomaly_key": key},
                )

    async def _run_tick(self) -> None:
        """Execute one detection cycle: fetch snapshot, run detectors, aggregate, and update blackboard."""
        tick_start = time.perf_counter()

        with tracer.start_as_current_span("anomaly_detector.tick") as span:
            span.set_attribute("detector.count", len(self._sub_detectors))

            snapshot = await self._poll_blackboard()
            current_intent = await self._poll_blackboard_intent()

            if snapshot is None:
                return

            try:
                valid_results = await self._run_detectors(snapshot)

                if not valid_results:
                    return

                active_flags = await self._aggregate_and_emit(valid_results)

                newly_active_keys = set(active_flags.keys())

                await self._clear_resolved_anomalies(newly_active_keys)

                await self._emit_and_store_active_anomalies(active_flags, snapshot, current_intent)

                self._active_anomaly_keys = newly_active_keys
                active_anomalies.set(len(newly_active_keys))

                if newly_active_keys:
                    self._anomalies_detected += 1
                    logger.info(f"Anomalies detected: {list(newly_active_keys)}")

            except Exception as e:
                logger.exception(f"Unexpected error during tick processing: {e}")

            finally:
                self._ticks_processed += 1
                duration_ms = (time.perf_counter() - tick_start) * 1000.0
                component_duration_ms.labels(component="anomaly_detector_tick").observe(duration_ms)

    def _aggregate_results(
        self, results: list[tuple[str, dict[str, Any]]]
    ) -> dict[str, tuple[AnomalySeverity, float, list[str], dict[str, Any]] | None]:
        """
        Aggregate sub-detector results. A flag is set if ANY detector reports medium/high severity.
        Tracks the highest severity, maximum score, and list of contributing detectors for each key.

        Args:
            results: List of tuples (detector_name, detector_result_dict).

        Returns:
            Dictionary mapping anomaly keys to (highest severity, max score, contributing detector names, metadata).
        """
        # key -> (severity, score, detector_list)
        aggregated: dict[str, tuple[AnomalySeverity, float, list[str], dict[str, Any]] | None] = {}

        for detector_name, result in results:
            key = result.get("key", "generic_anomaly")

            # Validate Score: Default to 0.5 if missing, but cap it [0.0, 1.0]
            score_input = result.get("score") or result.get("anomaly_score")
            try:
                score = float(score_input) if score_input is not None else 0.5
                score = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                score = 0.5

            # Handle both string and Enum inputs for severity
            severity_input = result.get("severity", AnomalySeverity.LOW)
            try:
                if isinstance(severity_input, str):
                    severity = AnomalySeverity(severity_input.lower())
                else:
                    severity = severity_input
            except ValueError:
                logger.warning(
                    f"Invalid severity input '{severity_input}' received for key '{key}'"
                )
                severity = AnomalySeverity.LOW

            # Flag as anomalous if medium or high severity
            is_anomalous = severity in (AnomalySeverity.MEDIUM, AnomalySeverity.HIGH)

            if is_anomalous:
                # Get current aggregated state for this key
                current_severity, current_score, current_detectors, current_metadata = (
                    aggregated.get(key, (AnomalySeverity.LOW, 0.0, [], {}))
                )

                # 1. Determine the highest severity seen
                highest_severity = current_severity
                if SEVERITY_RANKING[severity] > SEVERITY_RANKING[current_severity]:
                    highest_severity = severity

                # 2. Determine the max score seen
                max_score = max(current_score, score)

                # 3. Collect unique list of contributing detectors
                updated_detectors = list(set(current_detectors + [detector_name]))

                # 4. Get optional metadata
                metadata: dict[str, Any] = result.get("metadata", {})

                # Update the aggregated state
                aggregated[key] = (highest_severity, max_score, updated_detectors, metadata)

        return aggregated

    async def _run_loop(self) -> None:
        """Main detection loop."""
        self._loop_running = True
        logger.info(
            f"AnomalyDetector loop started "
            f"(detectors={len(self._sub_detectors)}, "
            f"tick_interval={self._tick_interval}s)"
        )

        try:
            while self._loop_running:
                await self._run_tick()
                await asyncio.sleep(self._tick_interval)

        except asyncio.CancelledError:
            logger.info("AnomalyDetector loop cancelled gracefully")
            raise
        except Exception as e:
            logger.exception(f"Fatal error in AnomalyDetector loop: {e}")
            raise
        finally:
            self._loop_running = False
            logger.info(
                f"AnomalyDetector stopped "
                f"(ticks={self._ticks_processed}, "
                f"anomalies={self._anomalies_detected}, "
                f"failures={self._detector_failures})"
            )

    async def start(self) -> None:
        """Start the anomaly detection loop."""
        if self._loop_running:
            logger.warning("AnomalyDetector already running")
            return

        if self._task and not self._task.done():
            logger.warning("Previous task still active")
            return

        if not self._sub_detectors:
            logger.warning("No sub-detectors registered, starting anyway")

        self._task = asyncio.create_task(self._run_loop())
        logger.info("AnomalyDetector started")

    async def stop(self) -> None:
        """Stop the anomaly detection loop gracefully."""
        logger.info("Stopping AnomalyDetector...")
        self._loop_running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("AnomalyDetector task cancelled")

        logger.info("AnomalyDetector stopped")

    def get_metrics(self) -> dict[str, int | float]:
        """Get detector metrics for monitoring."""
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
