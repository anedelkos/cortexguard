"""Tests for CapabilityRegistry — regression for risk-level and serialisation bugs."""

from __future__ import annotations

import json

from cortexguard.edge.models.capability_registry import (
    CapabilityRegistry,
    FunctionSchema,
    RiskLevel,
)


def _registry_with_high_risk() -> CapabilityRegistry:
    return CapabilityRegistry(
        capabilities={
            "activate_blade": FunctionSchema(
                description="Spin up cutting blade",
                parameters={"type": "object", "properties": {}, "required": []},
                risk_level=RiskLevel.HIGH,
            )
        }
    )


def _registry_with_medium_risk() -> CapabilityRegistry:
    return CapabilityRegistry(
        capabilities={
            "open_gripper": FunctionSchema(
                description="Open gripper",
                parameters={"type": "object", "properties": {}, "required": []},
                risk_level=RiskLevel.MEDIUM,
            )
        }
    )


# ---------------------------------------------------------------------------
# m1 — validate_call must return the declared risk level, not always LOW
# ---------------------------------------------------------------------------


def test_validate_call_returns_high_risk_for_high_risk_function() -> None:
    registry = _registry_with_high_risk()
    valid, risk = registry.validate_call("activate_blade", {})
    assert valid is True
    assert risk == RiskLevel.HIGH, f"Expected HIGH, got {risk}"


def test_validate_call_returns_medium_risk_for_medium_risk_function() -> None:
    registry = _registry_with_medium_risk()
    valid, risk = registry.validate_call("open_gripper", {})
    assert valid is True
    assert risk == RiskLevel.MEDIUM, f"Expected MEDIUM, got {risk}"


def test_validate_call_returns_low_risk_for_low_risk_function() -> None:
    registry = CapabilityRegistry(
        capabilities={
            "query_sensor": FunctionSchema(
                description="Read sensor value",
                parameters={"type": "object", "properties": {}, "required": []},
                risk_level=RiskLevel.LOW,
            )
        }
    )
    valid, risk = registry.validate_call("query_sensor", {})
    assert valid is True
    assert risk == RiskLevel.LOW


# ---------------------------------------------------------------------------
# m2 — get_llm_tool_catalog must return valid JSON without raising TypeError
# ---------------------------------------------------------------------------


def test_get_llm_tool_catalog_returns_valid_json() -> None:
    registry = _registry_with_high_risk()
    catalog_str = registry.get_llm_tool_catalog()
    catalog = json.loads(catalog_str)
    assert isinstance(catalog, list)
    assert catalog[0]["name"] == "activate_blade"


def test_get_llm_tool_catalog_includes_risk_level_as_string() -> None:
    registry = _registry_with_high_risk()
    catalog_str = registry.get_llm_tool_catalog()
    catalog = json.loads(catalog_str)
    assert catalog[0]["risk_level"] == "HIGH"
