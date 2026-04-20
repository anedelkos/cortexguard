"""Anthropic-backed planner client for structured remediation plan generation."""

from __future__ import annotations

import os
from typing import Any

from cortexguard.cloud.planner.llm_client import PlannerRequest, PlannerResponse
from cortexguard.cloud.planner.prompts import build_planner_prompt


class AnthropicLLMClient:
    MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str | None = None) -> None:
        import anthropic
        import instructor

        key = api_key or os.environ["CLOUD_ANTHROPIC_API_KEY"]
        raw_client = anthropic.AsyncAnthropic(api_key=key)
        self._client: Any = instructor.from_anthropic(raw_client)

    async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
        import logging

        logger = logging.getLogger(__name__)
        prompt = build_planner_prompt(request)
        try:
            response: PlannerResponse = await self._client.messages.create(
                model=self.MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                response_model=PlannerResponse,
            )
            logger.info(
                "LLM response: candidate_plan=%s confidence=%s needs_human=%s rationale=%s",
                response.candidate_plan is not None,
                response.confidence,
                response.needs_human_review,
                response.rationale,
            )
            return response
        except Exception:
            logger.exception("LLM call failed")
            raise
