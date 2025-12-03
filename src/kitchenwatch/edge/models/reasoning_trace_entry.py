import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceSeverity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ReasoningTraceEntry(BaseModel):
    id: str = Field(default_factory=lambda: f"trace-{uuid4().hex[:8]}")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    source: str = Field(..., description="Module that generated the entry")
    event_type: str = Field(..., description="High-level event identifier")
    reasoning_text: str = Field(..., description="Human-readable summary of the event")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Compact structured details")
    refs: dict[str, str] = Field(
        default_factory=dict, description="Cross references like scene_id, step_id"
    )
    duration_ms: int | None = Field(
        default=None, description="Elapsed time for the action in milliseconds"
    )
    severity: TraceSeverity = Field(TraceSeverity.INFO, description="Trace priority/severity")

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "id": "trace-1a2b3c4d",
                    "timestamp": "2025-12-03T15:12:34.567Z",
                    "source": "OnlineLearnerStateEstimator",
                    "event_type": "STATE_ESTIMATE_COMPUTED",
                    "severity": "INFO",
                    "reasoning_text": "State estimate computed: transient_disturbance",
                    "metadata": {
                        "label": "transient_disturbance",
                        "confidence": 0.42,
                        "max_z": 4.2,
                    },
                    "refs": {"scene_id": "sg-9f8e7d6c", "state_estimate_id": "se-1234"},
                    "duration_ms": 12,
                }
            ]
        },
    }
