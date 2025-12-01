from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ReasoningTraceEntry(BaseModel):
    """
    An immutable, timestamped record of an event or state change on the system's Blackboard.
    Used by Executors and Detectors to inform the Planner (LLM) of system status.
    """

    timestamp: datetime = Field(
        default_factory=datetime.now, description="The exact time the event was recorded."
    )

    source: str = Field(
        ...,
        description="The module that generated the entry (e.g., 'AnomalyDetector', 'StepExecutor', 'VisionSystem').",
    )

    event_type: str = Field(
        ...,
        description="A high-level type identifier for the event (e.g., 'ANOMALY_TRIGGERED', 'STEP_COMPLETED', 'NEW_GOAL').",
    )

    reasoning_text: str = Field(
        ...,
        description="A brief, human-readable summary of the event (e.g., 'Grill_1 temperature sensor out of range').",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data relevant to the event (e.g., device_id, severity level, step_id).",
    )
