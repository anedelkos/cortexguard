from __future__ import annotations

from pydantic import BaseModel, Field


class TelemetrySnapshot(BaseModel):
    """Single sensor snapshot from a step execution, as transmitted over the wire."""

    id: str
    timestamp: str | None = None
    sensors: dict[str, float] = {}
    derived: dict[str, float] = {}


class TelemetryRecord(BaseModel):
    """A single step-execution outcome record sent from an edge device to the cloud."""

    device_id: str
    key: str
    outcome: str
    timestamp: str
    sensor_snapshot: TelemetrySnapshot | None = None


class TelemetryBatch(BaseModel):
    """Batch of telemetry records posted by an edge device to the cloud API."""

    records: list[TelemetryRecord] = Field(max_length=1000)
