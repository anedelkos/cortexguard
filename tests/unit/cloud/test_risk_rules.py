from __future__ import annotations

import uuid

from cortexguard.cloud.validation.risk_rules import (
    all_capabilities_known,
    contains_estop_action,
    contains_high_risk_action,
    forbidden_combination_violations,
    invalid_argument_calls,
    plan_exceeds_max_length,
)
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.plan import Plan, PlanSource, PlanStep, PlanType


class _MockAdapter:
    def __init__(
        self, known: set[str], high_risk: set[str], e_stop: set[str] | None = None
    ) -> None:
        self._known = known
        self._high_risk = high_risk
        self._e_stop = e_stop or set()

    def is_known_capability(self, name: str) -> bool:
        return name in self._known

    def get_risk_level(self, name: str) -> str | None:
        if name in self._e_stop:
            return "E-STOP"
        if name in self._high_risk:
            return "HIGH"
        if name in self._known:
            return "LOW"
        return None

    def validate_arguments(self, name: str, arguments: dict[str, object]) -> list[str]:
        return []


def _make_plan(*action_names: str) -> Plan:
    steps = [
        PlanStep(
            description=f"step {name}",
            action=AgentToolCall(action_name=name, arguments={}),
        )
        for name in action_names
    ]
    return Plan(
        plan_id=str(uuid.uuid4()),
        context=GoalContext(
            goal_id=str(uuid.uuid4()),
            user_prompt="test",
            intent="test intent",
        ),
        plan_type=PlanType.REMEDIATION,
        source=PlanSource.CLOUD_AGENT,
        steps=steps,
    )


class TestContainsEstopAction:
    def test_detects_emergency_stop(self) -> None:
        adapter = _MockAdapter(known={"EMERGENCY_STOP"}, high_risk=set(), e_stop={"EMERGENCY_STOP"})
        plan = _make_plan("EMERGENCY_STOP")
        assert contains_estop_action(plan, adapter) is True

    def test_no_estop_returns_false(self) -> None:
        adapter = _MockAdapter(known={"PLACE_ITEM"}, high_risk=set())
        plan = _make_plan("PLACE_ITEM")
        assert contains_estop_action(plan, adapter) is False


class TestContainsHighRiskAction:
    def test_detects_high_risk(self) -> None:
        adapter = _MockAdapter(known={"RESET_DEVICE"}, high_risk={"RESET_DEVICE"})
        plan = _make_plan("RESET_DEVICE")
        assert contains_high_risk_action(plan, adapter) is True

    def test_no_high_risk_returns_false(self) -> None:
        adapter = _MockAdapter(known={"PLACE_ITEM"}, high_risk=set())
        plan = _make_plan("PLACE_ITEM")
        assert contains_high_risk_action(plan, adapter) is False


class TestPlanExceedsMaxLength:
    def test_exceeds_limit(self) -> None:
        plan = _make_plan(*[f"ACTION_{i}" for i in range(11)])
        assert plan_exceeds_max_length(plan, max_steps=10) is True

    def test_within_limit(self) -> None:
        plan = _make_plan("ACTION_1", "ACTION_2")
        assert plan_exceeds_max_length(plan, max_steps=10) is False

    def test_exactly_at_limit(self) -> None:
        plan = _make_plan(*[f"ACTION_{i}" for i in range(10)])
        assert plan_exceeds_max_length(plan, max_steps=10) is False


class TestAllCapabilitiesKnown:
    def test_returns_unknown_names(self) -> None:
        adapter = _MockAdapter(known={"PLACE_ITEM"}, high_risk=set())
        plan = _make_plan("PLACE_ITEM", "UNKNOWN_ACTION")
        unknown = all_capabilities_known(plan, adapter)
        assert unknown == ["UNKNOWN_ACTION"]

    def test_all_known_returns_empty(self) -> None:
        adapter = _MockAdapter(known={"PLACE_ITEM", "RESET_DEVICE"}, high_risk=set())
        plan = _make_plan("PLACE_ITEM", "RESET_DEVICE")
        assert all_capabilities_known(plan, adapter) == []


class _BadArgsAdapter(_MockAdapter):
    """Adapter that reports argument validation failures for named capabilities."""

    def __init__(self, known: set[str], bad_args: set[str]) -> None:
        super().__init__(known=known, high_risk=set())
        self._bad_args = bad_args

    def validate_arguments(self, name: str, arguments: dict[str, object]) -> list[str]:
        if name in self._bad_args:
            return [f"{name}: arguments failed schema validation"]
        return []


class TestInvalidArgumentCalls:
    def test_returns_empty_for_valid_args(self) -> None:
        adapter = _MockAdapter(known={"PLACE_ITEM"}, high_risk=set())
        plan = _make_plan("PLACE_ITEM")
        assert invalid_argument_calls(plan, adapter) == []

    def test_returns_error_for_bad_args(self) -> None:
        adapter = _BadArgsAdapter(known={"PLACE_ITEM"}, bad_args={"PLACE_ITEM"})
        plan = _make_plan("PLACE_ITEM")
        errors = invalid_argument_calls(plan, adapter)
        assert len(errors) == 1
        assert "PLACE_ITEM" in errors[0]

    def test_skips_unknown_capabilities(self) -> None:
        adapter = _MockAdapter(known=set(), high_risk=set())
        plan = _make_plan("UNKNOWN_CAP")
        assert invalid_argument_calls(plan, adapter) == []


class TestForbiddenCombinations:
    def test_consecutive_duplicates_detected(self) -> None:
        plan = _make_plan("PLACE_ITEM", "PLACE_ITEM")
        violations = forbidden_combination_violations(plan)
        assert any("consecutive duplicate" in v for v in violations)

    def test_deliver_order_not_last_detected(self) -> None:
        plan = _make_plan("DELIVER_ORDER", "PLACE_ITEM")
        violations = forbidden_combination_violations(plan)
        assert any("DELIVER_ORDER" in v for v in violations)

    def test_deliver_order_as_last_step_is_clean(self) -> None:
        plan = _make_plan("PLACE_ITEM", "DELIVER_ORDER")
        assert forbidden_combination_violations(plan) == []

    def test_clean_plan_returns_empty(self) -> None:
        plan = _make_plan("PLACE_ITEM", "RESET_DEVICE")
        assert forbidden_combination_violations(plan) == []
