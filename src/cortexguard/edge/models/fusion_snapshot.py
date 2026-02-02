from datetime import datetime
from typing import Any

from pydantic import BaseModel


class FusionSnapshot(BaseModel):
    """Represents a snapshot of fused sensor data at a point in time.

    Attributes:
        timestamp: When the snapshot was taken.
        sensors: Raw sensor readings, keyed by modality (force, vision, temperature, etc.).
        derived: Processed or aggregated sensor information, e.g., EMA-smoothed values, risk scores.
    """

    id: str
    timestamp: datetime
    sensors: dict[str, Any] = {}
    derived: dict[str, Any] = {}  # processed / fused result
