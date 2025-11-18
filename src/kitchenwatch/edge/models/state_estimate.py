from datetime import datetime
from typing import Any

from pydantic import BaseModel


class StateEstimate(BaseModel):
    """
    Semantic state estimate produced by the state estimator.

    Attributes:
        timestamp: When the estimate was produced.
        label: High-level state (e.g., "heating", "grasping", "slipping").
        confidence: 0..1 estimate of label correctness.
        residuals: Currently observed deviations per sensor (placeholder if no expected yet).
        uncertainty: Optional per-signal or overall uncertainty (can be None for now).
        ttd: Time-to-done for current action (optional).
        ttf: Time-to-failure for current action (optional).
        flags: Dict for anomaly or condition flags (vision degraded, impact, etc.).
        source_intent: The action or step that produced this state (from FusionSnapshot.intent).
    """

    timestamp: datetime
    label: str
    confidence: float
    residuals: dict[str, float] = {}
    uncertainty: dict[str, float] | None = None
    ttd: float | None = None
    ttf: float | None = None
    flags: dict[str, Any] = {}
    source_intent: str | None = None
