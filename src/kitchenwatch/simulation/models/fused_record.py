from pydantic import BaseModel, Field


class FusedRecord(BaseModel):
    timestamp_ns: int = Field(description="Nanosecond timestamp from image filename")
    rgb_path: str = Field(description="Path to RGB image")
    depth_path: str | None = Field(default=None, description="Path to depth image (optional)")

    force_x: float = Field(description="Force in X (N)")
    force_y: float = Field(description="Force in Y (N)")
    force_z: float = Field(description="Force in Z (N)")

    torque_x: float = Field(description="Torque in X (Nm)")
    torque_y: float = Field(description="Torque in Y (Nm)")
    torque_z: float = Field(description="Torque in Z (Nm)")

    pos_x: float = Field(description="Feeling position X (m)")
    pos_y: float = Field(description="Feeling position Y (m)")
    pos_z: float = Field(description="Feeling position Z (m)")
