"""Persistence-layer model for edge step-telemetry records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class StoredTelemetryRecord:
    """A single step-execution outcome record reported by an edge device."""

    device_id: str
    key: str
    outcome: str
    timestamp: datetime
    sensor_snapshot_json: str
