from pydantic import BaseModel, Field

from kitchenwatch.simulation.models.base_record import BaseFusedRecord


class SensorReading(BaseModel):
    timestamp_ns: int

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
    rgb_path: str
    depth_path: str | None = None

    window_size_s: float
    n_samples: int
    sensor_window: list[SensorReading] = Field(default_factory=list)
