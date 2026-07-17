"""Provider-neutral request, response, and protocol types for planner backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from cortexguard.edge.models.plan import Plan


class PlannerRequest(BaseModel):
    anomaly_key: str
    severity: str
    escalation_summary: str
    state_summary: str
    retrieved_summaries: list[str] = Field(default_factory=list)
    capability_catalog_json: str

    reasoning_trace: str = ""
    last_actions: str = ""
    current_plan_id: str | None = None
    current_step: str | None = None
    scene_graph: str = ""
    system_health: str = ""
    anomaly_details: str = ""
    remediation_policy: str = ""
    current_plan_compact: str = ""


class PlannerResponse(BaseModel):
    candidate_plan: Plan | None
    confidence: float
    needs_human_review: bool
    rationale: str
    raw_provider_metadata: dict[str, Any] = {}


@runtime_checkable
class LLMClientProtocol(Protocol):
    async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse: ...
