import copy
import logging
from collections.abc import Iterator
from typing import Any

from kitchenwatch.simulation.models.windowed_fused_record import SensorReading, WindowedFusedRecord
from tests.integration.scenario_loader import AnomalySpec, Scenario

logger = logging.getLogger(__name__)


class ChaosEngine:
    """
    Inject anomalies into a baseline fused record stream according to a Scenario.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self._vision_sidecar: dict[int, list[dict[str, Any]]] = {}
        self._vision_occlusion: dict[str, float] = {}
        self._repeat_state: dict[int, int | None] = {
            i: None for i in range(len(self.scenario.anomalies))
        }
        self._last_arrival_ns: int | None = None

    def inject(
        self, baseline_stream: Iterator[WindowedFusedRecord]
    ) -> Iterator[WindowedFusedRecord]:
        """
        Wrap a baseline stream and yield perturbed records based on scenario anomalies.
        """
        for fused_record in baseline_stream:
            perturbed = fused_record.model_copy(deep=True)

            for idx, anomaly in enumerate(self.scenario.anomalies):
                should_apply = False
                repeat_count = int(anomaly.repeat or 1)

                if repeat_count <= 1:
                    should_apply = True
                else:
                    remaining = self._repeat_state.get(idx)
                    if remaining is None:
                        self._repeat_state[idx] = repeat_count
                        should_apply = True
                    elif remaining > 0:
                        should_apply = True
                    else:
                        should_apply = False

                if should_apply:
                    perturbed = self._apply_anomaly(perturbed, anomaly)
                    if repeat_count > 1:
                        self._repeat_state[idx] = max(0, (self._repeat_state[idx] or 0) - 1)

            yield perturbed

    def _apply_anomaly(
        self, record: WindowedFusedRecord, anomaly: AnomalySpec
    ) -> WindowedFusedRecord:
        """
        Modify a record in-place based on anomaly spec.
        Works with WindowedFusedRecord fields so EdgeFusion can interpret anomalies.
        """

        match anomaly.type:
            case "human_intrusion":
                vision_obj: dict[str, Any] = {
                    "id": "chaos_person_1",
                    "label": "person",
                    "distance_m": float(anomaly.distance_m) if anomaly.distance_m else None,
                    "confidence": 0.99,
                    "bbox": [0.1, 0.1, 0.2, 0.3],
                }

                self._vision_sidecar.setdefault(record.timestamp_ns, []).append(vision_obj)

            case "temp_spike":
                last_temp = next(
                    (r.temp_c for r in reversed(record.sensor_window) if r.temp_c is not None), None
                )
                base = last_temp if last_temp is not None else 25.0
                injected_temp = base + (float(anomaly.delta_c) if anomaly.delta_c else 0.0)
                record.sensor_window.append(
                    SensorReading(timestamp_ns=record.timestamp_ns, temp_c=injected_temp)
                )

            case "visual_smoke":
                opacity = float(getattr(anomaly, "opacity", 1.0))

                # Map visual opacity to a numeric sensor reading (ppm) for the smoke sensor.
                # This mapping is arbitrary and should be tuned to your simulated sensor characteristics.
                # Example: treat opacity 1.0 -> 200 ppm, opacity 0.5 -> 100 ppm
                SMOKE_PPM_MAX = 200.0
                smoke_ppm = opacity * SMOKE_PPM_MAX

                # 1) Append a SensorReading with smoke_ppm so fusion can pick it up from sensor_window
                record.sensor_window.append(
                    SensorReading(timestamp_ns=record.timestamp_ns, smoke_ppm=smoke_ppm)
                )

                vision_obj = {
                    "id": f"chaos_smoke_{record.timestamp_ns}",
                    "label": "smoke",
                    "bbox": [0.1, 0.1, 0.9, 0.9],  # full-frame marker
                    "distance_m": 1.0,
                    "confidence": 0.95,
                    "properties": {"opacity": opacity},
                }

                # store in the vision sidecar keyed by timestamp so _mock_vision_inference can return it
                self._vision_sidecar.setdefault(record.timestamp_ns, []).append(vision_obj)

            case "slip":
                force_pct = float(getattr(anomaly, "force_pct", 40.0))
                drift_mm = float(getattr(anomaly, "drift_mm", 35.0))
                # 1) Append a SensorReading so fusion sees a force spike
                record.sensor_window.append(
                    SensorReading(timestamp_ns=record.timestamp_ns, force_x=force_pct)
                )

                # 2) Add deterministic vision evidence (so scene graph / perception sees drift)
                vision_obj = {
                    "id": f"chaos_slip_{record.timestamp_ns}",
                    "label": "patty",
                    "bbox": [0.2, 0.2, 0.6, 0.6],
                    "properties": {"drift_mm": drift_mm, "confidence": 0.9},
                    "confidence": 0.9,
                }
                self._vision_sidecar.setdefault(record.timestamp_ns, []).append(vision_obj)

            case "sensor_freeze":
                if record.sensor_window:
                    frozen = copy.deepcopy(record.sensor_window[-1])
                    frozen.timestamp_ns = record.timestamp_ns
                    record.sensor_window = [copy.deepcopy(frozen) for _ in range(record.n_samples)]

            case "vision_occlusion":
                ts = record.timestamp_ns

                vision_obj = {
                    "id": f"chaos_occlusion_{ts}",
                    "label": "occluding_object",
                    "bbox": [0.1, 0.1, 0.9, 0.9],
                    "properties": {"occluding": True, "opacity": anomaly.opacity},
                    "confidence": 0.4,
                    "source": "test_injector",
                }

                area_pct = float(anomaly.opacity) * 100.0 if anomaly.opacity is not None else 80.0
                self._vision_occlusion = {"area_pct": area_pct, "duration_s": 5.0}
                self._vision_sidecar.setdefault(record.timestamp_ns, []).append(vision_obj)

            case "comm_lag":
                comm_lag_duration_ns = int((getattr(anomaly, "duration_s", 1.0)) * 1e9)
                if self._last_arrival_ns:
                    record.arrival_time_ns = self._last_arrival_ns + comm_lag_duration_ns
                else:
                    record.arrival_time_ns = record.timestamp_ns + comm_lag_duration_ns

                self._last_arrival_ns = record.arrival_time_ns

        return record
