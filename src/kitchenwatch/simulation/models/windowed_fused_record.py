from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    timestamp_ns: int
    force_x: float
    force_y: float
    force_z: float
    torque_x: float
    torque_y: float
    torque_z: float
    pos_x: float
    pos_y: float
    pos_z: float


class WindowedFusedRecord(BaseModel):
    """Fused record containing a short sequence of sensor samples."""

    timestamp_ns: int
    rgb_path: str
    depth_path: str | None = None
    window_size_s: float
    n_samples: int
    sensor_window: list[SensorReading] = Field(default_factory=list)
