from __future__ import annotations

from pydantic import BaseModel, Field

from cortexguard.simulation.models.base_record import BaseFusedRecord


class SensorReading(BaseModel):
    timestamp_ns: int
    temp_c: float | None = None
    smoke_ppm: float | None = None

    force_x: float | None = None
    force_y: float | None = None
    force_z: float | None = None

    torque_x: float | None = None
    torque_y: float | None = None
    torque_z: float | None = None

    pos_x: float | None = None
    pos_y: float | None = None
    pos_z: float | None = None


class WindowedFusedRecord(BaseFusedRecord):
    """Fused record containing a short sequence of sensor samples."""

    timestamp_ns: int
    arrival_time_ns: int | None = None
    rgb_path: str
    depth_path: str | None = None

    window_size_s: float
    n_samples: int
    sensor_window: list[SensorReading] = Field(default_factory=list)
    vision_objects: list[dict[str, object]] = Field(default_factory=list)
    vision_occlusion: dict[str, float] | None = None
