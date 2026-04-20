"""Provider-neutral request, response, and protocol types for planner backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from cortexguard.edge.models.plan import Plan


class PlannerRequest(BaseModel):
    escalation_summary: str
    state_summary: str
    retrieved_summaries: list[str]
    capability_catalog_json: str
    anomaly_key: str
    severity: str


class PlannerResponse(BaseModel):
    candidate_plan: Plan | None
    confidence: float
    needs_human_review: bool
    rationale: str
    raw_provider_metadata: dict[str, Any] = {}


@runtime_checkable
class LLMClientProtocol(Protocol):
    async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse: ...
