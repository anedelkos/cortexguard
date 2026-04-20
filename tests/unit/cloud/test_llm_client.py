from __future__ import annotations

import pytest

from cortexguard.cloud.planner.llm_client import PlannerRequest, PlannerResponse
from cortexguard.cloud.planner.mock_client import MockLLMClient
from cortexguard.cloud.planner.prompts import build_planner_prompt
from cortexguard.edge.models.plan import PlanType


def _make_request(**kwargs: object) -> PlannerRequest:
    defaults: dict[str, object] = {
        "escalation_summary": "arm grip failed 3 times",
        "state_summary": "{}",
        "retrieved_summaries": [],
        "capability_catalog_json": '{"PLACE_ITEM": {}}',
        "anomaly_key": "S1.1_MISGRASP",
        "severity": "high",
    }
    defaults.update(kwargs)
    return PlannerRequest(**defaults)  # type: ignore[arg-type]


class TestMockLLMClient:
    @pytest.mark.asyncio
    async def test_returns_canned_response_by_default(self) -> None:
        client = MockLLMClient()
        response = await client.generate_structured_plan(_make_request())
        assert isinstance(response, PlannerResponse)
        assert response.candidate_plan is not None
        assert response.candidate_plan.plan_type == PlanType.REMEDIATION
        assert response.confidence == 0.5
        assert response.needs_human_review is False

    @pytest.mark.asyncio
    async def test_fixed_response_override_returns_exactly_that_response(self) -> None:
        fixed = PlannerResponse(
            candidate_plan=None,
            confidence=0.99,
            needs_human_review=True,
            rationale="override",
            raw_provider_metadata={"model": "test"},
        )
        client = MockLLMClient(fixed_response=fixed)
        response = await client.generate_structured_plan(_make_request())
        assert response is fixed


class TestBuildPlannerPrompt:
    def test_includes_capability_catalog(self) -> None:
        request = _make_request(
            capability_catalog_json='{"RESET_DEVICE": {"description": "resets"}}'
        )
        prompt = build_planner_prompt(request)
        assert "RESET_DEVICE" in prompt

    def test_includes_retrieved_summaries(self) -> None:
        request = _make_request(retrieved_summaries=["past grip slip resolved by reset"])
        prompt = build_planner_prompt(request)
        assert "past grip slip resolved by reset" in prompt

    def test_includes_hard_safety_rules(self) -> None:
        request = _make_request()
        prompt = build_planner_prompt(request)
        assert "Do not invent capabilities" in prompt
        assert "reversible" in prompt
        assert "structured JSON only" in prompt
        assert "outside the provided capability catalog" in prompt
