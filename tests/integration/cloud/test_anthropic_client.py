from __future__ import annotations

import os

import pytest

from cortexguard.cloud.planner.anthropic_client import AnthropicLLMClient
from cortexguard.cloud.planner.llm_client import PlannerRequest, PlannerResponse


@pytest.mark.llm_slow
@pytest.mark.asyncio
async def test_anthropic_client_returns_valid_response() -> None:
    if not os.getenv("CLOUD_ANTHROPIC_API_KEY"):
        pytest.skip("CLOUD_ANTHROPIC_API_KEY not set")
    client = AnthropicLLMClient()
    request = PlannerRequest(
        escalation_summary="Robot arm repeated grip failures, 3 consecutive misses on part A",
        state_summary='{"z_score": 4.8}',
        retrieved_summaries=["S1.1: reset grip, slow approach — outcome: completed"],
        capability_catalog_json='[{"name": "PLACE_ITEM", "description": "moves item", "risk_level": "LOW"}]',
        anomaly_key="S1.1_MISGRASP",
        severity="high",
    )
    response = await client.generate_structured_plan(request)
    assert isinstance(response, PlannerResponse)
    assert response.candidate_plan is not None or response.needs_human_review is True
