from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AnomalySpec(BaseModel):
    type: str
    severity: str | None = None
    distance_m: float | None = None
    delta_c: float | None = None
    opacity: float | None = None
    force_pct: float | None = None
    drift_mm: float | None = None
    repeat: int | None = None
    duration_s: float | None = None
    repeats: int | None = None


class Scenario(BaseModel):
    scenario_id: str
    title: str
    tier: int = Field(ge=0, le=5)
    seed: int | None = None
    duration_s: int | None = None
    baseline_recipe: str | None = None
    anomalies: list[AnomalySpec]
    expected_outcome: str


def load_scenarios(path: str | Path = "data/anomaly_scenarios.yaml") -> dict[str, Scenario]:
    """Load and validate scenarios from a YAML file."""
    with open(path, encoding="utf-8") as f:
        raw: list[dict[str, Any]] = yaml.safe_load(f)

    scenarios: dict[str, Scenario] = {}
    for entry in raw:
        scenario = Scenario.model_validate(entry)
        scenarios[scenario.scenario_id] = scenario
    return scenarios
