from datetime import datetime
from typing import Any

from pydantic import BaseModel

from kitchenwatch.edge.models.anomaly_event import AnomalyEvent, AnomalyReplay


class SystemHealth(BaseModel):
    """Compact summary of system status during a critical event."""

    cpu_load_pct: float | None = None
    net_rtt_ms: int | None = None
    packet_loss_pct: float | None = None
    disk_pressure_pct: float | None = None


class MaydayPacket(BaseModel):
    """
    Compact, prioritized packet sent from Edge to Cloud upon critical incident (HIGH/CRITICAL).
    This serves as the formal contract for the Cloud Agent's input.
    """

    device_id: str
    timestamp: datetime
    anomalies: list[AnomalyEvent]
    current_plan_id: str | None = None
    current_step: str | None = None
    last_actions: list[Any] = []  # List of recent step intents & statuses (ActionHistory)
    health: SystemHealth
    replay_data: AnomalyReplay | None = None  # Optional small compressed window
