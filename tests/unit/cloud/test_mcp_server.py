"""Unit tests for the CortexGuard MCP server handlers."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import AnyUrl

from cortexguard.cloud.config import CloudConfig
from cortexguard.cloud.mcp_server import (
    _VALID_OUTCOMES,
    CloudDependencies,
    CreateRemediationPlanHandler,
    GetLatestIncidentHandler,
    MetricsReporter,
    ProposeAlternativePlanHandler,
    RecordOperatorResolutionHandler,
    ResourceHandler,
    ToolRegistry,
    ValidatePlanHandler,
    create_mcp_server,
)
from cortexguard.cloud.orchestrator import PlanningResult
from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.plan import Plan, PlanStep, PlanType
from tests.unit.cloud.factories import make_incident

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NULL_METRICS = MetricsReporter()


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
def mock_orchestrator() -> AsyncMock:
    """AsyncMock CloudOrchestrator."""
    orch = AsyncMock()
    plan = _make_minimal_plan()
    orch.submit = AsyncMock(return_value="trace-123")
    orch.get_result = AsyncMock(return_value=PlanningResult(decision="plan_ready", plan=plan))
    return orch


@pytest.fixture()
def mock_validator() -> MagicMock:
    """MagicMock PlanValidator that returns valid."""
    from cortexguard.cloud.graph.state import ValidationResult

    validator = MagicMock()
    validator.validate = MagicMock(
        return_value=ValidationResult(passed=True, errors=[], risk_level="low")
    )
    return validator


@pytest.fixture()
def config() -> CloudConfig:
    return CloudConfig()


# ---------------------------------------------------------------------------
# create_mcp_server integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_mcp_server_registers_all_tools_when_deps_available(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_validator: MagicMock,
    config: CloudConfig,
) -> None:
    deps = CloudDependencies(
        repo=mock_repo,
        orchestrator=mock_orchestrator,
        validator=mock_validator,
        retrieval_store=AsyncMock(),
        config=config,
        sqs_queue=None,
        metrics=_NULL_METRICS,
        capability_registry_json='{"capabilities": {}}',
    )
    server = create_mcp_server(deps)
    assert server is not None


@pytest.mark.asyncio
async def test_create_mcp_server_skips_tools_when_deps_missing(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    config: CloudConfig,
) -> None:
    deps = CloudDependencies(
        repo=mock_repo,
        orchestrator=mock_orchestrator,
        validator=None,
        retrieval_store=None,
        config=config,
        sqs_queue=None,
        metrics=_NULL_METRICS,
        capability_registry_json='{"capabilities": {}}',
    )
    server = create_mcp_server(deps)
    assert server is not None


# ---------------------------------------------------------------------------
# ResourceHandler tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resource_capability_registry_returns_capabilities(
    mock_repo: AsyncMock,
) -> None:
    handler = ResourceHandler(
        repo=mock_repo,
        capability_registry_json='{"capabilities": {"test_cap": {"description": "Test", "risk_level": "low"}}}',
    )
    results = await handler.handle(AnyUrl("cortexguard://capability_registry"))
    assert len(results) == 1
    data = json.loads(results[0].content)
    assert "capabilities" in data


@pytest.mark.asyncio
async def test_resource_recent_incident_summaries_returns_list(
    mock_repo: AsyncMock,
) -> None:
    handler = ResourceHandler(repo=mock_repo, capability_registry_json='{"capabilities": {}}')
    results = await handler.handle(AnyUrl("cortexguard://recent_incident_summaries"))
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
) -> None:
    handler = ResourceHandler(repo=mock_repo, capability_registry_json='{"capabilities": {}}')
    incident = _make_incident_with_plan()
    mock_repo.list_recent_incidents = AsyncMock(return_value=[incident])

    results = await handler.handle(AnyUrl("cortexguard://latest_planner_decision"))
    data = json.loads(results[0].content)
    assert "incident_id" in data
    assert "decision" in data
    assert "plan" in data


@pytest.mark.asyncio
async def test_resource_latest_planner_decision_empty_repo(
    mock_repo: AsyncMock,
) -> None:
    handler = ResourceHandler(repo=mock_repo, capability_registry_json='{"capabilities": {}}')
    mock_repo.list_recent_incidents = AsyncMock(return_value=[])

    results = await handler.handle(AnyUrl("cortexguard://latest_planner_decision"))
    data = json.loads(results[0].content)
    assert data == {}


@pytest.mark.asyncio
async def test_resource_latest_planner_decision_malformed_plan_json(
    mock_repo: AsyncMock,
) -> None:
    handler = ResourceHandler(repo=mock_repo, capability_registry_json='{"capabilities": {}}')
    incident = make_incident(candidate_plan_json="not-valid-json")
    mock_repo.list_recent_incidents = AsyncMock(return_value=[incident])

    results = await handler.handle(AnyUrl("cortexguard://latest_planner_decision"))
    data = json.loads(results[0].content)
    assert data["plan"] is None


@pytest.mark.asyncio
async def test_resource_latest_planner_decision_malformed_retrieved_incidents_json(
    mock_repo: AsyncMock,
) -> None:
    handler = ResourceHandler(repo=mock_repo, capability_registry_json='{"capabilities": {}}')
    incident = make_incident(retrieved_incidents_json="not-valid-json")
    mock_repo.list_recent_incidents = AsyncMock(return_value=[incident])

    results = await handler.handle(AnyUrl("cortexguard://latest_planner_decision"))
    data = json.loads(results[0].content)
    assert data["retrieved_incidents"] == []


@pytest.mark.asyncio
async def test_resource_unknown_uri_raises(mock_repo: AsyncMock) -> None:
    handler = ResourceHandler(repo=mock_repo, capability_registry_json='{"capabilities": {}}')
    with pytest.raises(ValueError, match="Unknown resource"):
        await handler.handle(AnyUrl("cortexguard://nonexistent_resource"))


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_registry_list_tools(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_validator: MagicMock,
    config: CloudConfig,
) -> None:
    registry = ToolRegistry()
    registry.register(GetLatestIncidentHandler(repo=mock_repo, metrics=_NULL_METRICS))
    registry.register(ValidatePlanHandler(validator=mock_validator, metrics=_NULL_METRICS))
    registry.register(
        CreateRemediationPlanHandler(
            orchestrator=mock_orchestrator, repo=mock_repo, config=config, metrics=_NULL_METRICS
        )
    )
    registry.register(
        ProposeAlternativePlanHandler(
            orchestrator=mock_orchestrator, repo=mock_repo, config=config, metrics=_NULL_METRICS
        )
    )
    registry.register(
        RecordOperatorResolutionHandler(
            repo=mock_repo, retrieval_store=AsyncMock(), sqs_queue=None, metrics=_NULL_METRICS
        )
    )

    tools = registry.list_tools()
    tool_names = {t.name for t in tools}
    assert "validate_plan" in tool_names
    assert "create_remediation_plan" in tool_names
    assert "propose_alternative_plan" in tool_names
    assert "record_operator_resolution" in tool_names


@pytest.mark.asyncio
async def test_tool_registry_unknown_tool_returns_error() -> None:
    registry = ToolRegistry()
    result = await registry.call_tool("nonexistent_tool", {})
    assert "error" in result
    assert "nonexistent_tool" in result["error"]


# ---------------------------------------------------------------------------
# Tool: validate_plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_validate_plan_valid(mock_validator: MagicMock) -> None:
    handler = ValidatePlanHandler(validator=mock_validator, metrics=_NULL_METRICS)
    plan = _make_minimal_plan()
    plan_dict = plan.model_dump(mode="json")

    result = await handler.handle({"plan": plan_dict})
    assert result["valid"] is True
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_tool_validate_plan_invalid(mock_validator: MagicMock) -> None:
    from cortexguard.cloud.graph.state import ValidationResult

    mock_validator.validate = MagicMock(
        return_value=ValidationResult(
            passed=False, errors=["unknown capabilities: totally_fake_action"], risk_level="high"
        )
    )
    handler = ValidatePlanHandler(validator=mock_validator, metrics=_NULL_METRICS)
    plan = _make_minimal_plan()
    plan_dict = plan.model_dump(mode="json")
    plan_dict["steps"][0]["action"]["name"] = "totally_fake_action"

    result = await handler.handle({"plan": plan_dict})
    assert result["valid"] is False
    assert len(result["errors"]) > 0


@pytest.mark.asyncio
async def test_tool_validate_plan_exception_path(mock_validator: MagicMock) -> None:
    handler = ValidatePlanHandler(validator=mock_validator, metrics=_NULL_METRICS)
    result = await handler.handle({"plan": {"steps": "not-a-list"}})
    assert result["valid"] is False
    assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# Tool: create_remediation_plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_create_remediation_plan_returns_decision(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    config: CloudConfig,
) -> None:
    incident = _make_incident_with_plan()
    mock_repo.get_incident_by_trace_id = AsyncMock(return_value=incident)
    mock_orchestrator.submit = AsyncMock(return_value=incident.trace_id)
    mock_orchestrator.get_result = AsyncMock(
        return_value=PlanningResult(decision="plan_ready", plan=_make_minimal_plan())
    )

    handler = CreateRemediationPlanHandler(
        orchestrator=mock_orchestrator, repo=mock_repo, config=config, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {
            "anomaly_key": "OVERHEAT",
            "severity": "high",
            "device_id": "robot-arm-01",
            "summary": "Overheat detected",
            "state_summary": "temp=95C",
            "proposed_local_attempts": ["checked fans"],
        }
    )
    assert "decision" in result
    assert result["decision"] == "plan_ready"
    assert "incident_id" in result
    assert result["confidence"] == incident.confidence
    assert result["rationale"] == incident.rationale
    mock_repo.save_incident.assert_called_once()
    saved_record = mock_repo.save_incident.call_args[0][0]
    assert saved_record.source == "mcp"


@pytest.mark.asyncio
async def test_tool_create_remediation_plan_timeout(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    mock_orchestrator.get_result = AsyncMock(return_value="pending")

    handler = CreateRemediationPlanHandler(
        orchestrator=mock_orchestrator,
        repo=mock_repo,
        config=CloudConfig(cloud_llm_timeout_s=0.0),
        metrics=_NULL_METRICS,
    )
    with patch("cortexguard.cloud.mcp_server._poll_result", AsyncMock(return_value=None)):
        result = await handler.handle(
            {
                "anomaly_key": "OVERHEAT",
                "severity": "high",
                "device_id": "robot-arm-01",
            }
        )
    assert result["decision"] == "timeout"
    assert result["plan"] is None
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_tool_create_remediation_plan_unknown_severity(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    config: CloudConfig,
) -> None:
    incident = _make_incident_with_plan()
    mock_repo.get_incident_by_trace_id = AsyncMock(return_value=incident)
    mock_orchestrator.submit = AsyncMock(return_value=incident.trace_id)
    mock_orchestrator.get_result = AsyncMock(
        return_value=PlanningResult(decision="plan_ready", plan=_make_minimal_plan())
    )

    handler = CreateRemediationPlanHandler(
        orchestrator=mock_orchestrator, repo=mock_repo, config=config, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {
            "anomaly_key": "OVERHEAT",
            "severity": "completely_unknown_severity",
            "device_id": "device-01",
        }
    )
    assert "decision" in result


@pytest.mark.asyncio
async def test_tool_create_remediation_plan_incident_not_found_after_poll(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    config: CloudConfig,
) -> None:
    mock_repo.get_incident_by_trace_id = AsyncMock(return_value=None)
    mock_orchestrator.submit = AsyncMock(return_value="submitted-trace")
    mock_orchestrator.get_result = AsyncMock(
        return_value=PlanningResult(decision="plan_ready", plan=None)
    )

    handler = CreateRemediationPlanHandler(
        orchestrator=mock_orchestrator, repo=mock_repo, config=config, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {
            "anomaly_key": "OVERHEAT",
            "severity": "high",
            "device_id": "device-01",
        }
    )
    assert result["decision"] == "plan_ready"
    assert result["incident_id"] == "submitted-trace"
    assert result["confidence"] == 0.0
    assert result["plan"] is None
    mock_repo.save_incident.assert_not_called()


# ---------------------------------------------------------------------------
# Tool: propose_alternative_plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_propose_alternative_plan_sets_parent(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    config: CloudConfig,
) -> None:
    original_incident = _make_incident_with_plan(
        raw_packet_json=_make_valid_packet_json("robot-arm-01")
    )
    new_incident = _make_incident_with_plan(incident_id=str(uuid.uuid4()))

    mock_repo.get_incident = AsyncMock(return_value=original_incident)
    mock_repo.get_incident_by_trace_id = AsyncMock(return_value=new_incident)
    mock_orchestrator.submit = AsyncMock(return_value=new_incident.trace_id)
    mock_orchestrator.get_result = AsyncMock(
        return_value=PlanningResult(decision="plan_ready", plan=_make_minimal_plan())
    )

    handler = ProposeAlternativePlanHandler(
        orchestrator=mock_orchestrator, repo=mock_repo, config=config, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {"incident_id": original_incident.incident_id, "avoid": "recalibration"}
    )
    assert "incident_id" in result
    assert result["confidence"] == new_incident.confidence
    assert result["rationale"] == new_incident.rationale
    mock_repo.save_incident.assert_called_once()
    saved = mock_repo.save_incident.call_args[0][0]
    assert saved.parent_incident_id == original_incident.incident_id
    assert saved.source == "mcp"


@pytest.mark.asyncio
async def test_tool_propose_alternative_plan_incident_not_found(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    config: CloudConfig,
) -> None:
    mock_repo.get_incident = AsyncMock(return_value=None)
    handler = ProposeAlternativePlanHandler(
        orchestrator=mock_orchestrator, repo=mock_repo, config=config, metrics=_NULL_METRICS
    )

    result = await handler.handle({"incident_id": "nonexistent-id", "avoid": "recalibration"})
    assert result["decision"] == "error"
    assert "not found" in result["rationale"]
    assert result["plan"] is None
    mock_orchestrator.submit.assert_not_called()


@pytest.mark.asyncio
async def test_tool_propose_alternative_plan_timeout(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    original_incident = _make_incident_with_plan(
        raw_packet_json=_make_valid_packet_json("robot-arm-01")
    )
    mock_repo.get_incident = AsyncMock(return_value=original_incident)
    mock_orchestrator.submit = AsyncMock(return_value="new-trace-id")

    handler = ProposeAlternativePlanHandler(
        orchestrator=mock_orchestrator,
        repo=mock_repo,
        config=CloudConfig(cloud_llm_timeout_s=0.0),
        metrics=_NULL_METRICS,
    )
    with patch("cortexguard.cloud.mcp_server._poll_result", AsyncMock(return_value=None)):
        result = await handler.handle(
            {"incident_id": original_incident.incident_id, "avoid": "recalibration"}
        )
    assert result["decision"] == "timeout"
    assert result["plan"] is None


@pytest.mark.asyncio
async def test_tool_propose_alternative_plan_malformed_packet_json(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    config: CloudConfig,
) -> None:
    incident = make_incident(raw_packet_json="not-valid-json")
    mock_repo.get_incident = AsyncMock(return_value=incident)
    handler = ProposeAlternativePlanHandler(
        orchestrator=mock_orchestrator, repo=mock_repo, config=config, metrics=_NULL_METRICS
    )

    result = await handler.handle({"incident_id": incident.incident_id, "avoid": "recalibration"})
    assert result["decision"] == "error"
    assert "deserialize" in result["rationale"].lower() or "failed" in result["rationale"].lower()
    mock_orchestrator.submit.assert_not_called()


@pytest.mark.asyncio
async def test_tool_propose_alternative_plan_new_incident_not_found(
    mock_repo: AsyncMock,
    mock_orchestrator: AsyncMock,
    config: CloudConfig,
) -> None:
    original_incident = _make_incident_with_plan(
        raw_packet_json=_make_valid_packet_json("device-01")
    )
    mock_repo.get_incident = AsyncMock(return_value=original_incident)
    mock_repo.get_incident_by_trace_id = AsyncMock(return_value=None)
    mock_orchestrator.submit = AsyncMock(return_value="submitted-trace")
    mock_orchestrator.get_result = AsyncMock(
        return_value=PlanningResult(decision="plan_ready", plan=_make_minimal_plan())
    )

    handler = ProposeAlternativePlanHandler(
        orchestrator=mock_orchestrator, repo=mock_repo, config=config, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {"incident_id": original_incident.incident_id, "avoid": "recalibration"}
    )
    assert result["decision"] == "plan_ready"
    assert result["incident_id"] == "submitted-trace"
    assert result["confidence"] == 0.0
    mock_repo.save_incident.assert_not_called()


# ---------------------------------------------------------------------------
# Tool: get_latest_incident
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_get_latest_incident_returns_detail(mock_repo: AsyncMock) -> None:
    incident = make_incident(anomaly_key="ft_impact_impulse", decision="needs_human")
    mock_repo.list_recent_incidents = AsyncMock(return_value=[incident])

    handler = GetLatestIncidentHandler(repo=mock_repo, metrics=_NULL_METRICS)
    result = await handler.handle({})

    assert result["incident_id"] == incident.incident_id
    assert result["anomaly_key"] == "ft_impact_impulse"
    assert result["decision"] == "needs_human"
    assert "plan" in result
    assert "retrieved_incidents" in result


@pytest.mark.asyncio
async def test_tool_get_latest_incident_no_incidents(mock_repo: AsyncMock) -> None:
    mock_repo.list_recent_incidents = AsyncMock(return_value=[])

    handler = GetLatestIncidentHandler(repo=mock_repo, metrics=_NULL_METRICS)
    result = await handler.handle({})

    assert "error" in result


# ---------------------------------------------------------------------------
# Tool: record_operator_resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_record_operator_resolution_updates_store(
    mock_repo: AsyncMock,
) -> None:
    retrieval_store = AsyncMock()
    retrieval_store.index_incident = AsyncMock(return_value=None)

    handler = RecordOperatorResolutionHandler(
        repo=mock_repo, retrieval_store=retrieval_store, sqs_queue=None, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {
            "incident_id": "inc-001",
            "actions_taken": "Power-cycled the sensor",
            "outcome": "resolved",
            "notes": "Worked first time",
        }
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
) -> None:
    handler = RecordOperatorResolutionHandler(
        repo=mock_repo, retrieval_store=AsyncMock(), sqs_queue=None, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {
            "incident_id": "inc-001",
            "actions_taken": "Did something",
            "outcome": "invalid_outcome",
        }
    )
    assert "error" in result
    mock_repo.update_operator_resolution.assert_not_called()


@pytest.mark.asyncio
async def test_tool_record_operator_resolution_unknown_incident_id(
    mock_repo: AsyncMock,
) -> None:
    retrieval_store = AsyncMock()
    retrieval_store.index_incident = AsyncMock(return_value=None)
    mock_repo.update_operator_resolution = AsyncMock(return_value=False)
    mock_repo.get_incident = AsyncMock(return_value=None)

    handler = RecordOperatorResolutionHandler(
        repo=mock_repo, retrieval_store=retrieval_store, sqs_queue=None, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {
            "incident_id": "nonexistent-incident-id",
            "actions_taken": "Did something",
            "outcome": "resolved",
        }
    )
    assert result["recorded"] is False
    assert "error" in result
    mock_repo.update_operator_resolution.assert_called_once()
    retrieval_store.index_incident.assert_not_called()


@pytest.mark.asyncio
async def test_tool_record_operator_resolution_get_incident_returns_none(
    mock_repo: AsyncMock,
) -> None:
    retrieval_store = AsyncMock()
    retrieval_store.index_incident = AsyncMock(return_value=None)
    mock_repo.update_operator_resolution = AsyncMock(return_value=True)
    mock_repo.get_incident = AsyncMock(return_value=None)

    handler = RecordOperatorResolutionHandler(
        repo=mock_repo, retrieval_store=retrieval_store, sqs_queue=None, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {
            "incident_id": "inc-001",
            "actions_taken": "Did something",
            "outcome": "resolved",
        }
    )
    assert result["recorded"] is True
    assert result["embedding_updated"] is False
    retrieval_store.index_incident.assert_not_called()


@pytest.mark.asyncio
async def test_tool_record_operator_resolution_index_incident_raises(
    mock_repo: AsyncMock,
) -> None:
    retrieval_store = AsyncMock()
    retrieval_store.index_incident = AsyncMock(side_effect=RuntimeError("embed failed"))
    mock_repo.update_operator_resolution = AsyncMock(return_value=True)

    handler = RecordOperatorResolutionHandler(
        repo=mock_repo, retrieval_store=retrieval_store, sqs_queue=None, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {
            "incident_id": "inc-001",
            "actions_taken": "Did something",
            "outcome": "resolved",
        }
    )
    assert result["recorded"] is True
    assert result["embedding_updated"] is False


@pytest.mark.asyncio
async def test_tool_record_operator_resolution_no_retrieval_store(
    mock_repo: AsyncMock,
) -> None:
    """record_operator_resolution should skip re-embedding when retrieval_store is None."""
    mock_repo.update_operator_resolution = AsyncMock(return_value=True)

    handler = RecordOperatorResolutionHandler(
        repo=mock_repo, retrieval_store=None, sqs_queue=None, metrics=_NULL_METRICS
    )
    result = await handler.handle(
        {
            "incident_id": "inc-001",
            "actions_taken": "Did something",
            "outcome": "resolved",
        }
    )
    assert result["recorded"] is True
    assert result["embedding_updated"] is False


# ---------------------------------------------------------------------------
# MetricsReporter tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_reporter_record_tool_call() -> None:
    reporter = MetricsReporter()
    with patch.object(reporter, "_post") as mock_post:
        reporter.record_tool_call("validate_plan")
    mock_post.assert_called_once_with({"tool": "validate_plan"})


@pytest.mark.asyncio
async def test_metrics_reporter_record_resolution() -> None:
    reporter = MetricsReporter()
    with patch.object(reporter, "_post") as mock_post:
        reporter.record_resolution("resolved")
    mock_post.assert_called_once_with({"outcome": "resolved"})


@pytest.mark.asyncio
async def test_metrics_reporter_post_creates_task_when_loop_running() -> None:
    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        reporter = MetricsReporter()
        reporter._post({"tool": "validate_plan"})
    mock_loop.create_task.assert_called_once()


def test_metrics_reporter_post_no_running_loop_is_silent() -> None:
    with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
        reporter = MetricsReporter()
        reporter._post({"tool": "validate_plan"})  # must not raise


@pytest.mark.asyncio
async def test_metrics_reporter_custom_url() -> None:
    reporter = MetricsReporter(api_url="http://custom:9999/internal/mcp-event")
    with patch("httpx.AsyncClient"):
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.create_task = MagicMock(side_effect=lambda c: None)
            reporter._post({"tool": "test"})


# ---------------------------------------------------------------------------
# _VALID_OUTCOMES coverage
# ---------------------------------------------------------------------------


def test_valid_outcomes_known_values() -> None:
    assert _VALID_OUTCOMES == {
        "resolved",
        "escalated_further",
        "hardware_replaced",
        "aborted",
    }
