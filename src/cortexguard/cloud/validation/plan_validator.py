"""Plan-level validation for cloud-generated remediation candidates."""

from __future__ import annotations

from typing import Protocol

from cortexguard.cloud.graph.state import ValidationResult
from cortexguard.cloud.validation.capability_adapter import CapabilityAdapter
from cortexguard.cloud.validation.risk_rules import (
    all_capabilities_known,
    contains_estop_action,
    contains_high_risk_action,
    forbidden_combination_violations,
    invalid_argument_calls,
    plan_exceeds_max_length,
)
from cortexguard.edge.models.plan import Plan


class PlanValidatorProtocol(Protocol):
    def validate(
        self, plan: Plan | None, confidence: float, needs_human_review: bool
    ) -> ValidationResult: ...


class PlanValidator:
    def __init__(self, adapter: CapabilityAdapter, min_confidence: float = 0.5) -> None:
        self._adapter = adapter
        self._min_confidence = min_confidence

    def validate(
        self, plan: Plan | None, confidence: float, needs_human_review: bool
    ) -> ValidationResult:
        if plan is None:
            return ValidationResult(
                passed=False, errors=["no plan generated"], risk_level="unknown"
            )

        if confidence < self._min_confidence:
            return ValidationResult(passed=False, errors=["confidence too low"], risk_level="high")

        errors: list[str] = []

        if contains_estop_action(plan, self._adapter):
            errors.append("plan contains E-STOP action")

        if contains_high_risk_action(plan, self._adapter):
            errors.append("plan contains high-risk action")

        unknown = all_capabilities_known(plan, self._adapter)
        if unknown:
            errors.append(f"unknown capabilities: {', '.join(unknown)}")

        if plan_exceeds_max_length(plan):
            errors.append(f"plan exceeds max length of 10 steps ({len(plan.steps)} steps)")

        errors.extend(invalid_argument_calls(plan, self._adapter))
        errors.extend(forbidden_combination_violations(plan))

        if needs_human_review:
            errors.append("llm requested human review")

        if errors:
            return ValidationResult(passed=False, errors=errors, risk_level="high")

        return ValidationResult(passed=True, errors=[], risk_level="low")
