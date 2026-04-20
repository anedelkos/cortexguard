from __future__ import annotations

import uuid

from cortexguard.cloud.validation.capability_adapter import CapabilityAdapter
from cortexguard.cloud.validation.plan_validator import PlanValidator
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.capability_registry import CapabilityRegistry
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.plan import Plan, PlanSource, PlanStep, PlanType


class _MockAdapter(CapabilityAdapter):
    def __init__(
        self, known: set[str], high_risk: set[str], e_stop: set[str] | None = None
    ) -> None:
        super().__init__(CapabilityRegistry())
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


def _clean_adapter() -> _MockAdapter:
    return _MockAdapter(known={"PLACE_ITEM", "RESET_DEVICE"}, high_risk=set())


class TestPlanValidator:
    def test_valid_plan_returns_passed(self) -> None:
        validator = PlanValidator(_clean_adapter())
        plan = _make_plan("PLACE_ITEM")
        result = validator.validate(plan, confidence=0.9, needs_human_review=False)
        assert result.passed is True
        assert result.errors == []

    def test_none_plan_returns_failed(self) -> None:
        validator = PlanValidator(_clean_adapter())
        result = validator.validate(None, confidence=0.0, needs_human_review=False)
        assert result.passed is False
        assert len(result.errors) > 0

    def test_unknown_capability_returns_failed(self) -> None:
        adapter = _MockAdapter(known={"PLACE_ITEM"}, high_risk=set())
        validator = PlanValidator(adapter)
        plan = _make_plan("UNKNOWN_CAPABILITY")
        result = validator.validate(plan, confidence=0.8, needs_human_review=False)
        assert result.passed is False
        assert any("unknown" in e.lower() for e in result.errors)

    def test_estop_action_returns_failed(self) -> None:
        adapter = _MockAdapter(known={"EMERGENCY_STOP"}, high_risk=set(), e_stop={"EMERGENCY_STOP"})
        validator = PlanValidator(adapter)
        plan = _make_plan("EMERGENCY_STOP")
        result = validator.validate(plan, confidence=0.8, needs_human_review=False)
        assert result.passed is False
        assert any("E-STOP" in e for e in result.errors)

    def test_errors_list_populated_for_all_failure_cases(self) -> None:
        validator = PlanValidator(_clean_adapter())
        for plan_arg, confidence, needs_human in [
            (None, 0.0, False),
            (_make_plan("UNKNOWN_ACTION"), 0.8, False),
        ]:
            result = validator.validate(plan_arg, confidence, needs_human)  # type: ignore[arg-type]
            assert len(result.errors) > 0

    def test_invalid_arguments_returns_failed(self) -> None:
        class _BadArgsAdapter(_MockAdapter):
            def validate_arguments(self, name: str, arguments: dict[str, object]) -> list[str]:
                return [f"{name}: bad args"] if name in self._known else []

        adapter = _BadArgsAdapter(known={"PLACE_ITEM"}, high_risk=set())
        validator = PlanValidator(adapter)
        plan = _make_plan("PLACE_ITEM")
        result = validator.validate(plan, confidence=0.9, needs_human_review=False)
        assert result.passed is False
        assert any("bad args" in e for e in result.errors)

    def test_forbidden_combination_returns_failed(self) -> None:
        validator = PlanValidator(_clean_adapter())
        plan = _make_plan("PLACE_ITEM", "PLACE_ITEM")
        result = validator.validate(plan, confidence=0.9, needs_human_review=False)
        assert result.passed is False
        assert any("consecutive duplicate" in e for e in result.errors)
