"""Unit tests for the CortexGuard MCP server handlers."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import AnyUrl

from cortexguard.cloud.orchestrator import PlanningResult
from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.plan import Plan, PlanStep, PlanType
from tests.unit.cloud.factories import make_incident

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_plan() -> Plan:
    """Return the smallest valid Plan for use in tests."""
    step = PlanStep(
        description="Test step",
        action=AgentToolCall(action_name="set_power_state", arguments={"state": "off"}),
    )
    return Plan(
        context=GoalContext(
            goal_id=str(uuid.uuid4()),
            user_prompt="test",
            intent="test intent",
        ),
        plan_type=PlanType.REMEDIATION,
        steps=[step],
    )


def _make_incident_with_plan(**overrides: object) -> IncidentRecord:
    plan = _make_minimal_plan()
    base = make_incident(
        candidate_plan_json=plan.model_dump_json(),
        decision="plan_ready",
        rationale="Automated recovery",
        confidence=0.85,
        **overrides,
    )
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_repo() -> AsyncMock:
    """AsyncMock implementing IncidentRepositoryProtocol."""
    repo = AsyncMock()
    incident1 = _make_incident_with_plan(incident_id=str(uuid.uuid4()))
    incident2 = _make_incident_with_plan(incident_id=str(uuid.uuid4()))
    repo.list_recent_incidents = AsyncMock(return_value=[incident1, incident2])
    repo.get_incident = AsyncMock(return_value=incident1)
    repo.get_incident_by_trace_id = AsyncMock(return_value=incident1)
    repo.save_incident = AsyncMock(return_value=None)
    repo.update_operator_resolution = AsyncMock(return_value=True)
    return repo


@pytest.fixture()
def mock_orchestrator(mock_repo: AsyncMock) -> AsyncMock:
    """AsyncMock CloudOrchestrator."""
    orch = AsyncMock()
    plan = _make_minimal_plan()
    orch.submit = AsyncMock(return_value="trace-123")
    orch.get_result = AsyncMock(return_value=PlanningResult(decision="plan_ready", plan=plan))
    return orch


@pytest.fixture()
def mock_explain_client() -> AsyncMock:
    """AsyncMock ExplainClientProtocol."""
    client = AsyncMock()
    client.explain = AsyncMock(return_value="Mock explanation")
    return client


@pytest.fixture()
def mock_validator() -> MagicMock:
    """MagicMock PlanValidator that returns valid."""
    from cortexguard.cloud.graph.state import ValidationResult

    validator = MagicMock()
    validator.validate = MagicMock(
        return_value=ValidationResult(passed=True, errors=[], risk_level="low")
    )
    return validator


# ---------------------------------------------------------------------------
# Resource tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resource_capability_registry_returns_capabilities(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """capability_registry resource should return JSON with a 'capabilities' key."""
    import cortexguard.cloud.mcp_server as srv

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
    ):
        results = await srv.read_resource(AnyUrl("cortexguard://capability_registry"))
        assert len(results) == 1
        data = json.loads(results[0].content)
        assert "capabilities" in data


@pytest.mark.asyncio
async def test_resource_recent_incident_summaries_returns_list(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """recent_incident_summaries should return a list of 2 items with expected keys."""
    import cortexguard.cloud.mcp_server as srv

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
    ):
        results = await srv.read_resource(AnyUrl("cortexguard://recent_incident_summaries"))
        data = json.loads(results[0].content)
        assert isinstance(data, list)
        assert len(data) == 2
        for item in data:
            for key in (
                "incident_id",
                "timestamp",
                "device_id",
                "anomaly_key",
                "severity",
                "decision",
            ):
                assert key in item


@pytest.mark.asyncio
async def test_resource_latest_planner_decision_returns_full_detail(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """latest_planner_decision should return a dict with incident_id, decision, and plan."""
    import cortexguard.cloud.mcp_server as srv

    # list_recent_incidents returns 1 item for this resource
    incident = _make_incident_with_plan()
    mock_repo.list_recent_incidents = AsyncMock(return_value=[incident])

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
    ):
        results = await srv.read_resource(AnyUrl("cortexguard://latest_planner_decision"))
        data = json.loads(results[0].content)
        assert "incident_id" in data
        assert "decision" in data
        assert "plan" in data


# ---------------------------------------------------------------------------
# Tool: validate_plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_validate_plan_valid(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """validate_plan should return valid=True and empty errors for a good plan."""
    import cortexguard.cloud.mcp_server as srv

    plan = _make_minimal_plan()
    plan_dict = plan.model_dump(mode="json")

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
    ):
        result = await srv.call_tool("validate_plan", {"plan": plan_dict})
        assert result["valid"] is True
        assert result["errors"] == []


@pytest.mark.asyncio
async def test_tool_validate_plan_invalid(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """validate_plan should return valid=False and non-empty errors for an unknown action."""
    import cortexguard.cloud.mcp_server as srv
    from cortexguard.cloud.graph.state import ValidationResult

    mock_validator.validate = MagicMock(
        return_value=ValidationResult(
            passed=False, errors=["unknown capabilities: totally_fake_action"], risk_level="high"
        )
    )

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
    ):
        plan = _make_minimal_plan()
        plan_dict = plan.model_dump(mode="json")
        # Override action name to something fake
        plan_dict["steps"][0]["action"]["name"] = "totally_fake_action"
        result = await srv.call_tool("validate_plan", {"plan": plan_dict})
        assert result["valid"] is False
        assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# Tool: create_remediation_plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_create_remediation_plan_returns_decision(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """create_remediation_plan should return decision, incident_id and mark source=mcp."""
    import cortexguard.cloud.mcp_server as srv
    from cortexguard.cloud.config import CloudConfig

    incident = _make_incident_with_plan()
    mock_repo.get_incident_by_trace_id = AsyncMock(return_value=incident)
    mock_orchestrator.submit = AsyncMock(return_value=incident.trace_id)
    mock_orchestrator.get_result = AsyncMock(
        return_value=PlanningResult(decision="plan_ready", plan=_make_minimal_plan())
    )

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_config", CloudConfig()),
    ):
        result = await srv.call_tool(
            "create_remediation_plan",
            {
                "anomaly_key": "OVERHEAT",
                "severity": "high",
                "device_id": "robot-arm-01",
                "summary": "Overheat detected",
                "state_summary": "temp=95C",
                "proposed_local_attempts": ["checked fans"],
            },
        )
        assert "decision" in result
        assert result["decision"] == "plan_ready"
        assert "incident_id" in result
        # confidence and rationale should come from the incident record, not be hardcoded
        assert result["confidence"] == incident.confidence
        assert result["rationale"] == incident.rationale
        # save_incident should have been called to update source="mcp"
        mock_repo.save_incident.assert_called_once()
        saved_record = mock_repo.save_incident.call_args[0][0]
        assert saved_record.source == "mcp"


# ---------------------------------------------------------------------------
# Tool: explain_plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_explain_plan_by_incident_id(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """explain_plan with incident_id should call explain_client.explain once."""
    import cortexguard.cloud.mcp_server as srv

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
    ):
        result = await srv.call_tool("explain_plan", {"incident_id": "some-incident-id"})
        assert "explanation" in result
        mock_explain_client.explain.assert_called_once()


@pytest.mark.asyncio
async def test_tool_explain_plan_by_direct_input(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """explain_plan with plan + rationale should return explanation without repo lookup."""
    import cortexguard.cloud.mcp_server as srv

    plan = _make_minimal_plan()
    plan_dict = plan.model_dump(mode="json")

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
    ):
        result = await srv.call_tool(
            "explain_plan",
            {"plan": plan_dict, "rationale": "Because the sensor was faulty"},
        )
        assert "explanation" in result
        assert result["explanation"] == "Mock explanation"
        mock_explain_client.explain.assert_called_once()


# ---------------------------------------------------------------------------
# Tool: propose_alternative_plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_propose_alternative_plan_sets_parent(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """propose_alternative_plan should write parent_incident_id to the new incident."""
    import cortexguard.cloud.mcp_server as srv
    from cortexguard.cloud.config import CloudConfig

    original_incident = _make_incident_with_plan(
        raw_packet_json=_make_valid_packet_json("robot-arm-01")
    )
    new_incident = _make_incident_with_plan(incident_id=str(uuid.uuid4()))

    # get_incident returns original (for lookup by incident_id)
    # get_incident_by_trace_id returns new_incident (for lookup by submitted trace_id)
    mock_repo.get_incident = AsyncMock(return_value=original_incident)
    mock_repo.get_incident_by_trace_id = AsyncMock(return_value=new_incident)
    mock_orchestrator.submit = AsyncMock(return_value=new_incident.trace_id)
    mock_orchestrator.get_result = AsyncMock(
        return_value=PlanningResult(decision="plan_ready", plan=_make_minimal_plan())
    )

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_config", CloudConfig()),
    ):
        result = await srv.call_tool(
            "propose_alternative_plan",
            {"incident_id": original_incident.incident_id, "avoid": "recalibration"},
        )
        assert "incident_id" in result
        assert result["confidence"] == new_incident.confidence
        assert result["rationale"] == new_incident.rationale
        # save_incident should have been called to set parent_incident_id
        mock_repo.save_incident.assert_called_once()
        saved = mock_repo.save_incident.call_args[0][0]
        assert saved.parent_incident_id == original_incident.incident_id
        assert saved.source == "mcp"


@pytest.mark.asyncio
async def test_tool_propose_alternative_plan_incident_not_found(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """propose_alternative_plan should return decision=error when original incident is not found."""
    import cortexguard.cloud.mcp_server as srv

    mock_repo.get_incident = AsyncMock(return_value=None)

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
    ):
        result = await srv.call_tool(
            "propose_alternative_plan",
            {"incident_id": "nonexistent-id", "avoid": "recalibration"},
        )
        assert result["decision"] == "error"
        assert "not found" in result["rationale"]
        assert result["plan"] is None
        mock_orchestrator.submit.assert_not_called()


@pytest.mark.asyncio
async def test_tool_explain_plan_unknown_incident_id(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """explain_plan with an unknown incident_id should return an explanation containing 'Error'."""
    import cortexguard.cloud.mcp_server as srv

    mock_repo.get_incident = AsyncMock(return_value=None)

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
    ):
        result = await srv.call_tool("explain_plan", {"incident_id": "nonexistent-incident-id"})
        assert "explanation" in result
        assert "Error" in result["explanation"]
        mock_explain_client.explain.assert_not_called()


# ---------------------------------------------------------------------------
# Tool: record_operator_resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_record_operator_resolution_updates_store(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """record_operator_resolution should call update_operator_resolution with correct JSON."""
    import cortexguard.cloud.mcp_server as srv

    retrieval_store = AsyncMock()
    retrieval_store.index_incident = AsyncMock(return_value=None)

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_retrieval_store", retrieval_store),
    ):
        result = await srv.call_tool(
            "record_operator_resolution",
            {
                "incident_id": "inc-001",
                "actions_taken": "Power-cycled the sensor",
                "outcome": "resolved",
                "notes": "Worked first time",
            },
        )
        assert result["recorded"] is True
        assert result["embedding_updated"] is True
        mock_repo.update_operator_resolution.assert_called_once()
        call_args = mock_repo.update_operator_resolution.call_args
        assert call_args[0][0] == "inc-001"
        resolution = json.loads(call_args[0][1])
        assert resolution["actions_taken"] == "Power-cycled the sensor"
        assert resolution["outcome"] == "resolved"


@pytest.mark.asyncio
async def test_tool_record_operator_resolution_invalid_outcome(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """record_operator_resolution with invalid outcome should return an error."""
    import cortexguard.cloud.mcp_server as srv

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
    ):
        result = await srv.call_tool(
            "record_operator_resolution",
            {
                "incident_id": "inc-001",
                "actions_taken": "Did something",
                "outcome": "invalid_outcome",
            },
        )
        assert "error" in result
        mock_repo.update_operator_resolution.assert_not_called()


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_calls_counter_incremented(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """Calling any tool should fire _post_mcp_event with the tool label."""
    import cortexguard.cloud.mcp_server as srv

    plan = _make_minimal_plan()
    plan_dict = plan.model_dump(mode="json")

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_post_mcp_event") as mock_post,
    ):
        await srv.call_tool("validate_plan", {"plan": plan_dict})

    mock_post.assert_called_once_with({"tool": "validate_plan"})


@pytest.mark.asyncio
async def test_operator_resolutions_counter_incremented(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """record_operator_resolution should fire _post_mcp_event with the outcome."""
    import cortexguard.cloud.mcp_server as srv

    retrieval_store = AsyncMock()
    retrieval_store.index_incident = AsyncMock(return_value=None)

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_retrieval_store", retrieval_store),
        patch.object(srv, "_post_mcp_event") as mock_post,
    ):
        await srv.call_tool(
            "record_operator_resolution",
            {
                "incident_id": "inc-001",
                "actions_taken": "Replaced sensor",
                "outcome": "resolved",
            },
        )

    mock_post.assert_any_call({"outcome": "resolved"})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_counter(metric: Any, labels: dict[str, str]) -> float:
    """Read the current value of a labelled Prometheus Counter."""
    try:
        return float(metric.labels(**labels)._value.get())
    except Exception:
        return 0.0


@pytest.mark.asyncio
async def test_tool_record_operator_resolution_unknown_incident_id(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """record_operator_resolution with an unknown incident_id should return recorded=False."""
    import cortexguard.cloud.mcp_server as srv

    retrieval_store = AsyncMock()
    retrieval_store.index_incident = AsyncMock(return_value=None)

    # update_operator_resolution returns False (not found), get_incident returns None
    mock_repo.update_operator_resolution = AsyncMock(return_value=False)
    mock_repo.get_incident = AsyncMock(return_value=None)

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_retrieval_store", retrieval_store),
    ):
        result = await srv.call_tool(
            "record_operator_resolution",
            {
                "incident_id": "nonexistent-incident-id",
                "actions_taken": "Did something",
                "outcome": "resolved",
            },
        )
        assert result["recorded"] is False
        assert "error" in result
        # update_operator_resolution was still called (before the not-found check)
        mock_repo.update_operator_resolution.assert_called_once()
        # index_incident should NOT have been called
        retrieval_store.index_incident.assert_not_called()


@pytest.mark.asyncio
async def test_tool_create_remediation_plan_timeout(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """create_remediation_plan should return decision=timeout when _poll_result times out."""
    import cortexguard.cloud.mcp_server as srv
    from cortexguard.cloud.config import CloudConfig

    # get_result always returns "pending" so _poll_result times out immediately
    mock_orchestrator.get_result = AsyncMock(return_value="pending")

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_config", CloudConfig(cloud_llm_timeout_s=0.0)),
        patch.object(srv, "_poll_result", AsyncMock(return_value=None)),
    ):
        result = await srv.call_tool(
            "create_remediation_plan",
            {
                "anomaly_key": "OVERHEAT",
                "severity": "high",
                "device_id": "robot-arm-01",
            },
        )
        assert result["decision"] == "timeout"
        assert result["plan"] is None
        assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_tool_propose_alternative_plan_timeout(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """propose_alternative_plan should return decision=timeout when _poll_result times out."""
    import cortexguard.cloud.mcp_server as srv
    from cortexguard.cloud.config import CloudConfig

    original_incident = _make_incident_with_plan(
        raw_packet_json=_make_valid_packet_json("robot-arm-01")
    )
    mock_repo.get_incident = AsyncMock(return_value=original_incident)
    mock_orchestrator.submit = AsyncMock(return_value="new-trace-id")

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_config", CloudConfig(cloud_llm_timeout_s=0.0)),
        patch.object(srv, "_poll_result", AsyncMock(return_value=None)),
    ):
        result = await srv.call_tool(
            "propose_alternative_plan",
            {"incident_id": original_incident.incident_id, "avoid": "recalibration"},
        )
        assert result["decision"] == "timeout"
        assert result["plan"] is None


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resource_latest_planner_decision_empty_repo(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """latest_planner_decision should return empty dict when no incidents exist."""
    import cortexguard.cloud.mcp_server as srv

    mock_repo.list_recent_incidents = AsyncMock(return_value=[])

    with patch.object(srv, "_repo", mock_repo):
        results = await srv.read_resource(AnyUrl("cortexguard://latest_planner_decision"))
    data = json.loads(results[0].content)
    assert data == {}


@pytest.mark.asyncio
async def test_resource_latest_planner_decision_malformed_plan_json(
    mock_repo: AsyncMock,
) -> None:
    """latest_planner_decision should handle malformed candidate_plan_json gracefully."""
    import cortexguard.cloud.mcp_server as srv

    incident = make_incident(candidate_plan_json="not-valid-json")
    mock_repo.list_recent_incidents = AsyncMock(return_value=[incident])

    with patch.object(srv, "_repo", mock_repo):
        results = await srv.read_resource(AnyUrl("cortexguard://latest_planner_decision"))
    data = json.loads(results[0].content)
    assert data["plan"] is None


@pytest.mark.asyncio
async def test_resource_latest_planner_decision_malformed_retrieved_incidents_json(
    mock_repo: AsyncMock,
) -> None:
    """latest_planner_decision should handle malformed retrieved_incidents_json gracefully."""
    import cortexguard.cloud.mcp_server as srv

    incident = make_incident(retrieved_incidents_json="not-valid-json")
    mock_repo.list_recent_incidents = AsyncMock(return_value=[incident])

    with patch.object(srv, "_repo", mock_repo):
        results = await srv.read_resource(AnyUrl("cortexguard://latest_planner_decision"))
    data = json.loads(results[0].content)
    assert data["retrieved_incidents"] == []


@pytest.mark.asyncio
async def test_resource_unknown_uri_raises(
    mock_repo: AsyncMock,
) -> None:
    """read_resource should raise ValueError for an unknown URI."""
    import cortexguard.cloud.mcp_server as srv

    with patch.object(srv, "_repo", mock_repo), pytest.raises(ValueError, match="Unknown resource"):
        await srv.read_resource(AnyUrl("cortexguard://nonexistent_resource"))


@pytest.mark.asyncio
async def test_list_tools_returns_all_tools(
    mock_repo: AsyncMock,
) -> None:
    """list_tools should return all registered tools."""
    import cortexguard.cloud.mcp_server as srv

    tools = await srv.list_tools()
    tool_names = {t.name for t in tools}
    assert "validate_plan" in tool_names
    assert "create_remediation_plan" in tool_names
    assert "explain_plan" in tool_names
    assert "propose_alternative_plan" in tool_names
    assert "record_operator_resolution" in tool_names


@pytest.mark.asyncio
async def test_call_tool_unknown_tool_returns_error(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """call_tool with an unknown tool name should return an error dict."""
    import cortexguard.cloud.mcp_server as srv

    with patch.object(srv, "_repo", mock_repo):
        result = await srv.call_tool("nonexistent_tool", {})
    assert "error" in result
    assert "nonexistent_tool" in result["error"]


@pytest.mark.asyncio
async def test_tool_validate_plan_exception_path(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """validate_plan should return valid=False when Plan.model_validate raises."""
    import cortexguard.cloud.mcp_server as srv

    with patch.object(srv, "_repo", mock_repo), patch.object(srv, "_validator", mock_validator):
        result = await srv.call_tool("validate_plan", {"plan": {"steps": "not-a-list"}})
    assert result["valid"] is False
    assert len(result["errors"]) > 0


@pytest.mark.asyncio
async def test_tool_create_remediation_plan_unknown_severity(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """create_remediation_plan with unknown severity should fall back to MEDIUM."""
    import cortexguard.cloud.mcp_server as srv
    from cortexguard.cloud.config import CloudConfig

    incident = _make_incident_with_plan()
    mock_repo.get_incident_by_trace_id = AsyncMock(return_value=incident)
    mock_orchestrator.submit = AsyncMock(return_value=incident.trace_id)
    mock_orchestrator.get_result = AsyncMock(
        return_value=PlanningResult(decision="plan_ready", plan=_make_minimal_plan())
    )

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_config", CloudConfig()),
    ):
        result = await srv.call_tool(
            "create_remediation_plan",
            {
                "anomaly_key": "OVERHEAT",
                "severity": "completely_unknown_severity",
                "device_id": "device-01",
            },
        )
    assert "decision" in result


@pytest.mark.asyncio
async def test_tool_explain_plan_no_input_returns_error(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """explain_plan with neither incident_id nor plan should return error."""
    import cortexguard.cloud.mcp_server as srv

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_explain_client", mock_explain_client),
    ):
        result = await srv.call_tool("explain_plan", {})
    assert "explanation" in result
    assert "Error" in result["explanation"]


@pytest.mark.asyncio
async def test_tool_create_remediation_plan_incident_not_found_after_poll(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """create_remediation_plan uses submitted_trace_id when incident lookup fails."""
    import cortexguard.cloud.mcp_server as srv
    from cortexguard.cloud.config import CloudConfig

    mock_repo.get_incident_by_trace_id = AsyncMock(return_value=None)
    mock_orchestrator.submit = AsyncMock(return_value="submitted-trace")
    mock_orchestrator.get_result = AsyncMock(
        return_value=PlanningResult(decision="plan_ready", plan=None)
    )

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_config", CloudConfig()),
    ):
        result = await srv.call_tool(
            "create_remediation_plan",
            {
                "anomaly_key": "OVERHEAT",
                "severity": "high",
                "device_id": "device-01",
            },
        )
    assert result["decision"] == "plan_ready"
    assert result["incident_id"] == "submitted-trace"
    assert result["confidence"] == 0.0
    assert result["plan"] is None
    mock_repo.save_incident.assert_not_called()


@pytest.mark.asyncio
async def test_tool_explain_plan_incident_with_no_plan_json(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """explain_plan with incident that has no candidate_plan_json shows (no steps)."""
    import cortexguard.cloud.mcp_server as srv

    incident = make_incident(candidate_plan_json=None)
    mock_repo.get_incident = AsyncMock(return_value=incident)

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_explain_client", mock_explain_client),
    ):
        result = await srv.call_tool("explain_plan", {"incident_id": incident.incident_id})
    assert "explanation" in result
    assert result["explanation"] == "Mock explanation"
    prompt_call = mock_explain_client.explain.call_args[0][0]
    assert "(no steps)" in prompt_call


@pytest.mark.asyncio
async def test_tool_explain_plan_incident_with_malformed_plan_json(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """explain_plan with malformed candidate_plan_json falls back to (no steps)."""
    import cortexguard.cloud.mcp_server as srv

    incident = make_incident(candidate_plan_json="not-valid-json")
    mock_repo.get_incident = AsyncMock(return_value=incident)

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_explain_client", mock_explain_client),
    ):
        result = await srv.call_tool("explain_plan", {"incident_id": incident.incident_id})
    assert "explanation" in result
    prompt_call = mock_explain_client.explain.call_args[0][0]
    assert "(no steps)" in prompt_call


@pytest.mark.asyncio
async def test_tool_propose_alternative_plan_malformed_packet_json(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """propose_alternative_plan should return decision=error when raw_packet_json is invalid."""
    import cortexguard.cloud.mcp_server as srv

    incident = make_incident(raw_packet_json="not-valid-json")
    mock_repo.get_incident = AsyncMock(return_value=incident)

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
    ):
        result = await srv.call_tool(
            "propose_alternative_plan",
            {"incident_id": incident.incident_id, "avoid": "recalibration"},
        )
    assert result["decision"] == "error"
    assert "deserialize" in result["rationale"].lower() or "failed" in result["rationale"].lower()
    mock_orchestrator.submit.assert_not_called()


@pytest.mark.asyncio
async def test_tool_propose_alternative_plan_new_incident_not_found(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """propose_alternative_plan returns incident_id=trace when new incident lookup fails."""
    import cortexguard.cloud.mcp_server as srv
    from cortexguard.cloud.config import CloudConfig

    original_incident = _make_incident_with_plan(
        raw_packet_json=_make_valid_packet_json("device-01")
    )
    mock_repo.get_incident = AsyncMock(return_value=original_incident)
    mock_repo.get_incident_by_trace_id = AsyncMock(return_value=None)
    mock_orchestrator.submit = AsyncMock(return_value="submitted-trace")
    mock_orchestrator.get_result = AsyncMock(
        return_value=PlanningResult(decision="plan_ready", plan=_make_minimal_plan())
    )

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_config", CloudConfig()),
    ):
        result = await srv.call_tool(
            "propose_alternative_plan",
            {"incident_id": original_incident.incident_id, "avoid": "recalibration"},
        )
    assert result["decision"] == "plan_ready"
    assert result["incident_id"] == "submitted-trace"
    assert result["confidence"] == 0.0
    mock_repo.save_incident.assert_not_called()


@pytest.mark.asyncio
async def test_tool_record_operator_resolution_get_incident_returns_none(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """record_operator_resolution with successful update but no record should not re-embed."""
    import cortexguard.cloud.mcp_server as srv

    retrieval_store = AsyncMock()
    retrieval_store.index_incident = AsyncMock(return_value=None)

    # update succeeds but get_incident returns None (record deleted between update and get)
    mock_repo.update_operator_resolution = AsyncMock(return_value=True)
    mock_repo.get_incident = AsyncMock(return_value=None)

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_retrieval_store", retrieval_store),
    ):
        result = await srv.call_tool(
            "record_operator_resolution",
            {
                "incident_id": "inc-001",
                "actions_taken": "Did something",
                "outcome": "resolved",
            },
        )
    assert result["recorded"] is True
    assert result["embedding_updated"] is False
    retrieval_store.index_incident.assert_not_called()


@pytest.mark.asyncio
async def test_tool_record_operator_resolution_index_incident_raises(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """record_operator_resolution should still return recorded=True if re-embedding fails."""
    import cortexguard.cloud.mcp_server as srv

    retrieval_store = AsyncMock()
    retrieval_store.index_incident = AsyncMock(side_effect=RuntimeError("embed failed"))

    mock_repo.update_operator_resolution = AsyncMock(return_value=True)

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_orchestrator", mock_orchestrator),
        patch.object(srv, "_explain_client", mock_explain_client),
        patch.object(srv, "_validator", mock_validator),
        patch.object(srv, "_retrieval_store", retrieval_store),
    ):
        result = await srv.call_tool(
            "record_operator_resolution",
            {
                "incident_id": "inc-001",
                "actions_taken": "Did something",
                "outcome": "resolved",
            },
        )
    assert result["recorded"] is True
    assert result["embedding_updated"] is False


@pytest.mark.asyncio
async def test_tool_get_latest_incident_returns_detail(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """get_latest_incident should return incident detail including decision and plan."""
    import cortexguard.cloud.mcp_server as srv

    incident = make_incident(anomaly_key="ft_impact_impulse", decision="needs_human")
    mock_repo.list_recent_incidents = AsyncMock(return_value=[incident])

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_post_mcp_event"),
    ):
        result = await srv.call_tool("get_latest_incident", {})

    assert result["incident_id"] == incident.incident_id
    assert result["anomaly_key"] == "ft_impact_impulse"
    assert result["decision"] == "needs_human"
    assert "plan" in result
    assert "retrieved_incidents" in result


@pytest.mark.asyncio
async def test_tool_get_latest_incident_no_incidents(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_explain_client: AsyncMock,
    mock_validator: MagicMock,
) -> None:
    """get_latest_incident should return an error when no incidents exist."""
    import cortexguard.cloud.mcp_server as srv

    mock_repo.list_recent_incidents = AsyncMock(return_value=[])

    with (
        patch.object(srv, "_repo", mock_repo),
        patch.object(srv, "_post_mcp_event"),
    ):
        result = await srv.call_tool("get_latest_incident", {})

    assert "error" in result


@pytest.mark.asyncio
async def test_post_mcp_event_creates_task_when_loop_running() -> None:
    """_post_mcp_event schedules a task on the running event loop."""
    import cortexguard.cloud.mcp_server as srv

    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        srv._post_mcp_event({"tool": "validate_plan"})

    mock_loop.create_task.assert_called_once()


def test_post_mcp_event_no_running_loop_is_silent() -> None:
    """_post_mcp_event does nothing when called outside an event loop."""
    import cortexguard.cloud.mcp_server as srv

    with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
        srv._post_mcp_event({"tool": "validate_plan"})  # must not raise


def _make_valid_packet_json(device_id: str) -> str:
    """Return a minimal valid MaydayPacket JSON string for testing."""
    from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
    from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth

    packet = MaydayPacket(
        trace_id=str(uuid.uuid4()),
        device_id=device_id,
        timestamp=datetime.now(UTC),
        health=SystemHealth(),
        anomalies=[
            AnomalyEvent(
                id=str(uuid.uuid4()),
                key="test_anomaly",
                severity=AnomalySeverity.HIGH,
                timestamp=datetime.now(UTC),
                metadata={},
                score=1.0,
                contributing_detectors=[],
            )
        ],
    )
    return packet.model_dump_json()
