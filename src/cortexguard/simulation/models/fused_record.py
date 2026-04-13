from __future__ import annotations

from pydantic import Field

from cortexguard.simulation.models.base_record import BaseFusedRecord


class FusedRecord(BaseFusedRecord):
    timestamp_ns: int = Field(description="Nanosecond timestamp from image filename")
    rgb_path: str = Field(description="Path to RGB image")
    depth_path: str | None = Field(default=None, description="Path to depth image (optional)")

    force_x: float | None = None
    force_y: float | None = None
    force_z: float | None = None

    torque_x: float | None = None
    torque_y: float | None = None
    torque_z: float | None = None

    pos_x: float | None = None
    pos_y: float | None = None
    pos_z: float | None = None
