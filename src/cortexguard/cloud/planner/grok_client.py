"""Grok-backed planner client for hosted open-model experiments."""

from __future__ import annotations

import logging
import os
from typing import Any

from cortexguard.cloud.planner.llm_client import PlannerRequest, PlannerResponse
from cortexguard.cloud.planner.prompts import build_planner_prompt

logger = logging.getLogger(__name__)

_XAI_BASE_URL = "https://api.x.ai/v1"
_DEFAULT_MODEL = "grok-3-mini"


class GrokLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        import instructor
        from openai import AsyncOpenAI

        key = api_key or os.environ["CLOUD_XAI_API_KEY"]
        raw_client = AsyncOpenAI(api_key=key, base_url=_XAI_BASE_URL)
        self._client: Any = instructor.from_openai(raw_client)
        self._model = model

    async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
        prompt = build_planner_prompt(request)
        try:
            response: PlannerResponse = await self._client.chat.completions.create(
                model=self._model,
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
            logger.exception("Grok LLM call failed")
            raise
