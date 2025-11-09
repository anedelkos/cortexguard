from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel


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
    status: Literal["pending", "running", "completed", "failed"] = "pending"


class Plan(BaseModel):
    """Represents a structured recipe remediation plan for the edge agent."""

    plan_id: str
    plan_type: Literal["recipe", "remediation"]
    version: str
    goal: str
    context: dict[str, Any] = {}
    priority: int = 0  # higher number = higher priority
    status: Literal["pending", "running", "paused", "completed"] = "pending"
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


class IntentContext(BaseModel):
    """Represents the current intent state of the edge agent."""

    goal: str
    step_id: str
    action: str
    parameters: dict[str, Any] = {}
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_actual_s: float | None = None
