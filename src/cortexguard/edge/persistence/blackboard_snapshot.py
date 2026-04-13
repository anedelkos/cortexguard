"""Pydantic DTO representing a serialisable snapshot of Blackboard durable state."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from cortexguard.edge.models.anomaly_event import AnomalyEvent
from cortexguard.edge.models.plan import Plan
from cortexguard.edge.models.remediation_policy import RemediationPolicy

CURRENT_SCHEMA_VERSION: int = 1


class BlackboardSnapshot(BaseModel):
    """Serialisable subset of Blackboard state, used for periodic persistence.

    Fields that are NOT persisted (ephemeral / self-healing):
    - latest_snapshot: 10 Hz sensor data, re-arrives within 100ms
    - latest_state_estimate: derived from online learner, not durable
    - _scene_graph: transient vision data
    - reasoning_traces: too noisy / large; not needed for restart recovery
    - current_step: derived from current_plan step index
    """

    schema_version: int = CURRENT_SCHEMA_VERSION
    captured_at: datetime
    active_anomalies: dict[str, AnomalyEvent]
    current_plan: Plan | None
    paused_plan: Plan | None
    plan_step_indices: dict[str, int]
    active_remediation_policy: RemediationPolicy | None
    # deque serialises as list; restored as deque(maxlen=100) on load
    failed_plans: list[Plan]
    recovery_status: dict[str, str]
    safety_flags: dict[str, bool]
