"""Risk-rule helpers applied to cloud-generated remediation plans."""

from __future__ import annotations

from cortexguard.cloud.validation.capability_adapter import CapabilityAdapterProtocol
from cortexguard.edge.models.plan import Plan

_HIGH_RISK_LEVEL = "HIGH"
_ESTOP_RISK_LEVEL = "E-STOP"
_MUST_BE_LAST = {"DELIVER_ORDER"}


def contains_estop_action(plan: Plan, adapter: CapabilityAdapterProtocol) -> bool:
    return any(
        adapter.get_risk_level(step.action.action_name) == _ESTOP_RISK_LEVEL for step in plan.steps
    )


def contains_high_risk_action(plan: Plan, adapter: CapabilityAdapterProtocol) -> bool:
    for step in plan.steps:
        risk = adapter.get_risk_level(step.action.action_name)
        if risk == _HIGH_RISK_LEVEL:
            return True
    return False


def plan_exceeds_max_length(plan: Plan, max_steps: int = 10) -> bool:
    return len(plan.steps) > max_steps


def all_capabilities_known(plan: Plan, adapter: CapabilityAdapterProtocol) -> list[str]:
    return [
        step.action.action_name
        for step in plan.steps
        if not adapter.is_known_capability(step.action.action_name)
    ]


def invalid_argument_calls(plan: Plan, adapter: CapabilityAdapterProtocol) -> list[str]:
    errors: list[str] = []
    for step in plan.steps:
        name = step.action.action_name
        if adapter.is_known_capability(name):
            errors.extend(adapter.validate_arguments(name, step.action.arguments))
    return errors


def forbidden_combination_violations(plan: Plan) -> list[str]:
    violations: list[str] = []
    names = [step.action.action_name for step in plan.steps]
    # Consecutive duplicate actions indicate bad planning
    for i in range(len(names) - 1):
        if names[i] == names[i + 1]:
            violations.append(f"consecutive duplicate action: {names[i]}")
    # Certain actions must appear only as the final step
    for i, name in enumerate(names[:-1]):
        if name in _MUST_BE_LAST:
            violations.append(f"{name} must be the final step but appears at position {i + 1}")
    return violations
