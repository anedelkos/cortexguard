from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalyReplay


class SystemHealth(BaseModel):
    """Compact summary of system status during a critical event."""

    cpu_load_pct: float | None = None
    net_rtt_ms: int | None = None
    packet_loss_pct: float | None = None
    disk_pressure_pct: float | None = None


class MaydayPacket(BaseModel):
    trace_id: str
    schema_version: str = "1.0"
    device_id: str
    timestamp: datetime

    anomalies: list[AnomalyEvent] = Field(default_factory=list)
    current_plan_id: str | None = None
    current_step: str | None = None
    last_actions: list[dict[str, object]] = Field(default_factory=list)
    health: SystemHealth

    state_estimate: dict[str, object] | None = None
    scene_graph_compact: dict[str, object] | None = None
    reasoning_trace: list[str] = Field(default_factory=list)

    # Compact, not raw models (use model_dump())
    remediation_policy: dict[str, object] | None = None
    current_plan_compact: dict[str, object] | None = None

    replay_data: AnomalyReplay | None = None
