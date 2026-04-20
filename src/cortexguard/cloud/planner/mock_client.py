"""Mock planner client used for local development and deterministic tests."""

from __future__ import annotations

import uuid

from cortexguard.cloud.planner.llm_client import PlannerRequest, PlannerResponse
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.plan import Plan, PlanSource, PlanStep, PlanType


def _canned_plan() -> Plan:
    return Plan(
        plan_id=str(uuid.uuid4()),
        context=GoalContext(
            goal_id=str(uuid.uuid4()),
            user_prompt="Cloud planner safe park",
            intent="Park arm safely pending operator review",
        ),
        plan_type=PlanType.REMEDIATION,
        source=PlanSource.CLOUD_AGENT,
        steps=[
            PlanStep(
                description="Park arm in safe home position",
                action=AgentToolCall(
                    action_name="PLACE_ITEM",
                    arguments={
                        "item_name": "arm",
                        "target_location": "home",
                        "tool_id": "RoboticGripper_A",
                    },
                ),
            )
        ],
    )


class MockLLMClient:
    def __init__(self, fixed_response: PlannerResponse | None = None) -> None:
        self._fixed_response = fixed_response

    async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
        if self._fixed_response is not None:
            return self._fixed_response
        return PlannerResponse(
            candidate_plan=_canned_plan(),
            confidence=0.5,
            needs_human_review=False,
            rationale="Mock canned safe-park plan",
            raw_provider_metadata={},
        )
