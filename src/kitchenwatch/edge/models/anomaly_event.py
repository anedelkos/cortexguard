from datetime import datetime
from typing import Any

from pydantic import BaseModel

from kitchenwatch.edge.models.anomaly_severity import AnomalySeverity


class AnomalyWindow(BaseModel):
    """Defines the time window over which the anomaly was detected."""

    start_ts: datetime
    end_ts: datetime | None = None  # May be None if detection is instantaneous


class AnomalyEvent(BaseModel):
    """
    Structured, self-describing event raised by a detector.
    This replaces the simple boolean/string flags on the Blackboard.
    It serves as the core artifact for the Reasoning Trace and Mayday Packet.
    """

    id: str  # Unique event UUID (evt-uuid)
    key: str  # Semantic key for the anomaly (e.g., "overheat_smoke_combo")
    timestamp: datetime
    severity: AnomalySeverity
    score: float  # Confidence/magnitude of the anomaly (0.0 to 1.0)
    contributing_detectors: list[str]  # e.g., ["HardLimitDetector", "VisionDetector"]
    metadata: dict[str, Any] = {}  # Sensor values, Z-scores, logic rule details
    window: AnomalyWindow | None = None  # Time window of the event


class AnomalyReplay(BaseModel):
    """Container for the compressed sensor data window associated with an anomaly."""

    window: AnomalyWindow
    compressed_data: str  # Base64 encoded lz4 or zstd compressed binary
    format: str = "lz4"
