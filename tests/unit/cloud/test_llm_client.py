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
        assert "Generate a remediation plan as structured JSON" in prompt
        assert "Only use actions listed in the capability catalog" in prompt

    def test_xml_structure_present(self) -> None:
        request = _make_request(
            anomaly_key="OVERHEAT",
            anomaly_details='[{"key": "OVERHEAT", "severity": "high"}]',
            reasoning_trace='[{"step": "check temp", "result": "over limit"}]',
            last_actions='[{"action": "SET_POWER", "result": "completed"}]',
            current_plan_id="plan-123",
            current_step="step-2",
            scene_graph='{"objects": [{"id": "1", "label": "person"}]}',
            system_health='{"cpu_load_pct": 45.0}',
            remediation_policy='{"action": "e-stop"}',
            current_plan_compact='{"id": "plan-123"}',
        )
        prompt = build_planner_prompt(request)
        assert "<context>" in prompt
        assert "<anomaly>" in prompt
        assert "<key>OVERHEAT</key>" in prompt
        assert "<state>" in prompt
        assert "<edge_reasoning>" in prompt
        assert "<actions_tried>" in prompt
        assert "<active_plan>" in prompt
        assert "<id>plan-123</id>" in prompt
        assert "<current_step>step-2</current_step>" in prompt
        assert "<vision>" in prompt
        assert "<system_health>" in prompt
        assert "<remediation_policy>" in prompt
        assert "<capability_catalog>" in prompt
        assert "<instruction>" in prompt

    def test_empty_fields_suppressed(self) -> None:
        request = _make_request()
        prompt = build_planner_prompt(request)
        assert "<edge_reasoning>" not in prompt
        assert "<actions_tried>" not in prompt
        assert "<vision>" not in prompt
        assert "<remediation_policy>" not in prompt
        assert "<active_plan>" not in prompt
        assert "<details>" not in prompt
