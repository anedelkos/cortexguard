from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from cortexguard.cloud.graph.nodes import make_validate_candidate_plan_node, route_decision
from cortexguard.cloud.graph.state import CloudPlanningState
from cortexguard.cloud.validation.capability_adapter import CapabilityAdapter
from cortexguard.cloud.validation.plan_validator import PlanValidator
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.capability_registry import CapabilityRegistry
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth
from cortexguard.edge.models.plan import Plan, PlanSource, PlanStep, PlanType


class _MockAdapter(CapabilityAdapter):
    def __init__(self, known: set[str]) -> None:
        super().__init__(CapabilityRegistry())
        self._known = known

    def is_known_capability(self, name: str) -> bool:
        return name in self._known

    def get_risk_level(self, name: str) -> str | None:
        return "LOW" if name in self._known else None


def _make_packet() -> MaydayPacket:
    return MaydayPacket(
        trace_id=str(uuid.uuid4()),
        device_id="robot-arm-01",
        timestamp=datetime.now(UTC),
        health=SystemHealth(),
    )


def _make_plan(action_name: str = "PLACE_ITEM") -> Plan:
    return Plan(
        plan_id=str(uuid.uuid4()),
        context=GoalContext(
            goal_id=str(uuid.uuid4()),
            user_prompt="test",
            intent="test intent",
        ),
        plan_type=PlanType.REMEDIATION,
        source=PlanSource.CLOUD_AGENT,
        steps=[
            PlanStep(
                description="step",
                action=AgentToolCall(action_name=action_name, arguments={}),
            )
        ],
    )


def _make_state(plan: Plan | None, confidence: float = 0.8) -> CloudPlanningState:
    return CloudPlanningState(
        request=_make_packet(),
        incident_id="test-id",
        retrieved_incidents=[],
        candidate_plan=plan,
        validation_result=None,
        decision=None,
        rationale="test",
        confidence=confidence,
        needs_human_review=False,
        errors=[],
    )


@pytest.mark.asyncio
async def test_valid_plan_routes_to_plan_ready() -> None:
    adapter = _MockAdapter(known={"PLACE_ITEM"})
    validator = PlanValidator(adapter)
    validate_node = make_validate_candidate_plan_node(validator)
    state = _make_state(_make_plan("PLACE_ITEM"))

    after_validate = await validate_node(state)
    after_route = await route_decision(after_validate)

    assert after_route["decision"] == "plan_ready"


@pytest.mark.asyncio
async def test_invalid_plan_routes_to_needs_human() -> None:
    adapter = _MockAdapter(known={"PLACE_ITEM"})
    validator = PlanValidator(adapter)
    validate_node = make_validate_candidate_plan_node(validator)
    state = _make_state(_make_plan("UNKNOWN_CAPABILITY"))

    after_validate = await validate_node(state)
    after_route = await route_decision(after_validate)

    assert after_route["decision"] == "needs_human"


@pytest.mark.asyncio
async def test_none_plan_routes_to_no_safe_plan() -> None:
    adapter = _MockAdapter(known={"PLACE_ITEM"})
    validator = PlanValidator(adapter)
    validate_node = make_validate_candidate_plan_node(validator)
    state = _make_state(None)

    after_validate = await validate_node(state)
    after_route = await route_decision(after_validate)

    assert after_route["decision"] == "no_safe_plan"
