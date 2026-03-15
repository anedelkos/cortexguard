from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StateEstimate(BaseModel):
    """
    Minimal state estimate structure based on core state estimation outputs.
    """

    timestamp: datetime = Field(..., description="Timestamp of the estimate generation.")
    label: str = Field(..., description="High-level state label (e.g., 'nominal', 'slipping').")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0).")
    observations: dict[str, float] = Field(
        default_factory=dict, description="Raw observed sensor values"
    )
    residuals: dict[str, float] = Field(
        default_factory=dict, description="Observed deviations per sensor."
    )
    uncertainty: dict[str, float] | None = Field(
        default=None, description="Optional per-signal uncertainty."
    )
    z_scores: dict[str, float] | None = Field(
        default=None, description="Optional computed z-scores."
    )
    ttd: float | None = Field(default=None, description="Time-to-done for current activity.")
    ttf: float | None = Field(default=None, description="Time-to-failure for current activity.")
    flags: dict[str, Any] = Field(default_factory=dict, description="Anomaly or condition flags.")

    # Context and Diagnostics fields used in your previous nominal state function
    source_intent: str | None = Field(
        default=None, description="The simple intent or context of the action being executed."
    )
    symbolic_system_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Low-level diagnostic state (e.g., CPU, memory, motor load).",
    )
