from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"


class PlanType(str, Enum):
    RECIPE = "recipe"
    REMEDIATION = "remediation"


class PlanStep(BaseModel):
    """Represents a single actionable step in a Plan or remediation sequence."""

    id: str
    name: str
    intent: str
    parameters: dict[str, Any] = {}
    expected_modalities: list[str] = []
    expected_duration_s: float | None = None  # planned duration
    actual_duration_s: float | None = None  # filled at runtime
    max_duration_s: float | None = None  # safety / TTF reference
    status: StepStatus = StepStatus.PENDING


class Plan(BaseModel):
    """Represents a structured recipe remediation plan for the edge agent."""

    plan_id: str
    plan_type: PlanType
    version: str
    goal: str
    current_step_index: int = 0
    context: dict[str, Any] = {}
    priority: int = 0  # lower number = higher priority
    status: PlanStatus = PlanStatus.PENDING
    created_at: datetime
    steps: list[PlanStep]
    deadline: datetime | None = None  # optional urgency/deadline

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "Plan":
        """Load a Plan from a YAML file and validate it."""
        file_path = Path(file_path)
        with file_path.open("r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def step_ids(self) -> list[str]:
        return [s.id for s in self.steps]

    def serialize(self) -> dict[str, Any]:
        return self.model_dump()


class IntentContext(BaseModel):
    """Represents the current intent state of the edge agent."""

    goal: str
    step_id: str
    action: str
    parameters: dict[str, Any] = {}
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_actual_s: float | None = None
