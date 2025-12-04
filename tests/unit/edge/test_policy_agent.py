from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kitchenwatch.core.interfaces.base_policy_engine import BasePolicyEngine
from kitchenwatch.edge.models.agent_tool_call import AgentToolCall
from kitchenwatch.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.capability_registry import CapabilityRegistry
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.plan import PlanStep, StepStatus
from kitchenwatch.edge.models.remediation_policy import RemediationPolicy
from kitchenwatch.edge.models.state_estimate import StateEstimate
from kitchenwatch.edge.policy.policy_agent import PolicyAgent


@pytest.fixture
def mock_policy_engine() -> MagicMock:
    """Mocks the BasePolicyEngine Protocol."""
    mock_engine = MagicMock(spec=BasePolicyEngine)
    mock_engine.model_name.return_value = "mock-llm-engine"
    mock_engine.generate_policy = AsyncMock()
    return mock_engine


@pytest.fixture
def mock_deps(mock_policy_engine: MagicMock) -> dict[str, Any]:
    """Fixture to provide mocked dependencies using the imported class specifications."""

    # Mock Blackboard
    mock_blackboard = MagicMock(spec=Blackboard)
    mock_blackboard.get_latest_state_estimate = AsyncMock()
    mock_blackboard.get_active_anomalies = AsyncMock()
    mock_blackboard.set_remediation_policy = AsyncMock()

    # Mock CapabilityRegistry
    mock_capability_registry = MagicMock(spec=CapabilityRegistry)
    mock_capability_registry.get_function_schema = MagicMock()
    mock_capability_registry.get_llm_tool_catalog = MagicMock(return_value="[]")

    return {
        "blackboard": mock_blackboard,
        "capability_registry": mock_capability_registry,
        "policy_engine": mock_policy_engine,
    }


@pytest.fixture
def agent(mock_deps: dict[str, Any]) -> PolicyAgent:
    """Fixture to provide a PolicyAgent instance (SUT)."""
    return PolicyAgent(**mock_deps)


# --- Helper for patching the assumed broken rule handler method ---


def create_mock_overheat_policy(
    anomaly_id: str, context: StateEstimate, escalation: bool = False
) -> RemediationPolicy:
    """Creates a deterministic RemediationPolicy for the mocked rule handler."""
    return RemediationPolicy(
        trigger_event=AnomalyEvent(
            id=anomaly_id,
            key="overheat_warning",
            severity=AnomalySeverity.MEDIUM,
            metadata={},
            timestamp=datetime.now(),
            score=0.8,
            contributing_detectors=["temp_sensor_monitor"],
        ),
        policy_id=f"POL-{anomaly_id}",
        reasoning_trace=f"Rules matched: Overheat detected during {context.source_intent}. Initiating cooldown.",
        risk_assessment="MEDIUM",
        corrective_steps=[
            PlanStep(
                id="1",
                description="Log warning.",
                status=StepStatus.PENDING,
                action=AgentToolCall(
                    action_name="log_event", arguments={"message": "Overheat rule triggered"}
                ),
            ),
            PlanStep(
                id="2",
                description="Start cooldown cycle.",
                status=StepStatus.PENDING,
                action=AgentToolCall(action_name="cooldown_cycle", arguments={"duration": 10}),
            ),
        ],
        escalation_required=escalation,
    )


# --- Tests ---


@patch("kitchenwatch.edge.policy.policy_agent.logger", autospec=True)
def test_agent_initialization_and_metrics(
    mock_sut_logger: MagicMock, mock_deps: dict[str, Any]
) -> None:
    """Test initialization and default metrics."""
    # Instantiate agent *after* the patch is active
    agent_instance = PolicyAgent(**mock_deps)

    assert agent_instance._policy_engine is not None

    # Check that the SUT's module logger was called with the engine name
    mock_sut_logger.info.assert_called_with("Policy Agent initialized. LLM Engine: mock-llm-engine")
    assert agent_instance.get_metrics()["policies_generated"] == 0


@pytest.mark.asyncio
async def test_validate_action_success(agent: PolicyAgent, mock_deps: dict[str, Any]) -> None:
    """Test successful action validation against CapabilityRegistry."""
    mock_deps["capability_registry"].get_function_schema.return_value = {"name": "test"}
    action = AgentToolCall(action_name="test_func", arguments={"tool_id": "T1"})
    assert agent._validate_action(action) is True


@patch("kitchenwatch.edge.policy.policy_agent.logger", autospec=True)
@pytest.mark.asyncio
async def test_handle_high_severity_anomaly(
    mock_sut_logger: MagicMock, agent: PolicyAgent, mock_deps: dict[str, Any]
) -> None:
    """Test rules-based policy generation for HIGH severity (Critical Shutdown)."""

    current_ts = datetime.now()

    anomaly = AnomalyEvent(
        id="A001",
        key="critical_failure",
        severity=AnomalySeverity.HIGH,
        metadata={},
        timestamp=current_ts,
        score=1.0,
        contributing_detectors=["mock_safety_detector"],
    )

    await agent._handle_anomaly_event(anomaly)

    # Assertions
    mock_deps["blackboard"].set_remediation_policy.assert_called_once()
    policy = mock_deps["blackboard"].set_remediation_policy.call_args[0][0]

    assert policy.escalation_required is True
    assert "Critical safety failure detected" in policy.reasoning_trace
    # Policy ID check removed as SUT generates UUIDs instead of deterministic IDs
    assert agent.get_metrics()["policies_generated"] == 1
    assert agent.get_metrics()["escalations_triggered"] == 1


@patch("kitchenwatch.edge.policy.policy_agent.logger", autospec=True)
@pytest.mark.asyncio
async def test_handle_unknown_medium_anomaly_delegation(
    mock_sut_logger: MagicMock, agent: PolicyAgent, mock_deps: dict[str, Any]
) -> None:
    """Test delegation to BasePolicyEngine for unknown MEDIUM anomaly."""

    current_ts = datetime.now()

    anomaly = AnomalyEvent(
        id="A003",
        key="unknown_vibration",
        severity=AnomalySeverity.MEDIUM,
        metadata={},
        timestamp=current_ts,
        score=0.7,
        contributing_detectors=["vibration_analyzer"],
    )
    context = StateEstimate(
        source_intent="mixing_batter", timestamp=current_ts, label="idle", confidence=0.85
    )

    mock_policy = RemediationPolicy(
        trigger_event=anomaly,
        policy_id="POL-A003-LLM",
        reasoning_trace="LLM decided to reduce speed.",
        risk_assessment="TBD",
        corrective_steps=[
            PlanStep(
                id="1",
                description="Reduce motor speed.",
                status=StepStatus.PENDING,
                action=AgentToolCall(action_name="reduce_speed", arguments={"value": 50}),
            )
        ],
        escalation_required=False,
    )

    mock_deps["blackboard"].get_latest_state_estimate.return_value = context
    mock_deps["policy_engine"].generate_policy.return_value = mock_policy

    await agent._handle_anomaly_event(anomaly)

    # Check if the policy engine was called with the correct arguments
    mock_deps["policy_engine"].generate_policy.assert_called_once()
    assert mock_deps["policy_engine"].generate_policy.call_args[1]["action_catalog_json"] == "[]"

    # Check if the LLM-generated policy object was pushed
    mock_deps["blackboard"].set_remediation_policy.assert_called_once()
    pushed_policy = mock_deps["blackboard"].set_remediation_policy.call_args[0][0]

    assert pushed_policy.reasoning_trace == mock_policy.reasoning_trace
    assert agent.get_metrics()["policies_generated"] == 1

    mock_sut_logger.info.assert_any_call(
        "Delegating unknown MEDIUM policy generation for anomaly unknown_vibration to LLM Engine: mock-llm-engine"
    )


@patch("kitchenwatch.edge.policy.policy_agent.logger", autospec=True)
@pytest.mark.asyncio
async def test_llm_based_validation_failure_updates_policy(
    mock_sut_logger: MagicMock, agent: PolicyAgent, mock_deps: dict[str, Any]
) -> None:
    """LLM-generated policies with invalid actions are marked as CRITICAL risk."""

    current_ts = datetime.now()

    anomaly = AnomalyEvent(
        id="A005",
        key="unknown_medium",
        severity=AnomalySeverity.MEDIUM,
        metadata={},
        timestamp=current_ts,
        score=0.6,
        contributing_detectors=["generic_monitor"],
    )
    context = StateEstimate(
        source_intent="test", timestamp=current_ts, label="testing", confidence=0.75
    )

    mock_policy = RemediationPolicy(
        trigger_event=anomaly,
        policy_id="POL-A005-LLM",
        reasoning_trace="LLM generated a plan.",
        risk_assessment="LOW",
        corrective_steps=[
            PlanStep(
                id="1",
                description="Invalid step.",
                status=StepStatus.PENDING,
                action=AgentToolCall(action_name="invalid_action", arguments={}),
            )
        ],
        escalation_required=False,
    )

    mock_deps["blackboard"].get_latest_state_estimate.return_value = context
    mock_deps["policy_engine"].generate_policy.return_value = mock_policy

    # Patch validation to FAIL for the LLM-generated action
    with patch.object(agent, "_validate_action", return_value=False):
        await agent._handle_anomaly_event(anomaly)

    policy = mock_deps["blackboard"].set_remediation_policy.call_args[0][0]

    assert policy.escalation_required is True
    assert "CRITICAL - Invalid Action" in policy.risk_assessment
    assert "WARNING: Generated plan contained invalid actions" in policy.reasoning_trace

    mock_sut_logger.error.assert_called()


# --- Anomaly Loop and Sorting Tests ---


@patch("kitchenwatch.edge.policy.policy_agent.logger", autospec=True)
@pytest.mark.asyncio
async def test_run_loop_stops_on_high_severity(
    mock_sut_logger: MagicMock, agent: PolicyAgent, mock_deps: dict[str, Any]
) -> None:
    """Test that the loop processes high severity and breaks immediately."""

    current_ts = datetime.now()

    anomaly_high = AnomalyEvent(
        id="A006",
        key="high_alert",
        severity=AnomalySeverity.HIGH,
        metadata={},
        timestamp=current_ts,
        score=0.99,
        contributing_detectors=["final_check"],
    )

    mock_deps["blackboard"].get_active_anomalies.return_value = {
        anomaly_high.id: anomaly_high,
    }

    mock_deps["blackboard"].get_latest_state_estimate.return_value = StateEstimate(
        source_intent="idle", timestamp=current_ts, label="idle", confidence=0.95
    )

    with patch.object(agent, "_handle_anomaly_event", new=AsyncMock()) as mock_handler:
        # Mock asyncio.sleep to break the loop after the first iteration
        async def mock_sleep(delay: float) -> None:
            agent._loop_running = False

        with patch("asyncio.sleep", side_effect=mock_sleep) as sleep_mock:
            agent._loop_running = True
            await agent._run_loop(0.1)

            # Assertions: Should only call the handler once (for the HIGH anomaly)
            assert mock_handler.call_count == 1
            assert mock_handler.call_args[0][0].id == anomaly_high.id
            sleep_mock.assert_called_once()

            mock_sut_logger.warning.assert_any_call(
                "HIGH severity anomaly processed, allowing Orchestrator to react"
            )


@pytest.mark.asyncio
async def test_generate_policy_uses_snapshot_summary(monkeypatch, mock_deps):

    engine = mock_deps["policy_engine"]
    agent = PolicyAgent(**mock_deps)

    # Build a minimal snapshot with scene_graph_summary
    snapshot = FusionSnapshot(
        id="s1",
        timestamp=datetime.now(),
        sensors={"scene_graph_summary": [{"id": "o1", "label": "knife"}]},
        derived={},
    )
    anomaly = AnomalyEvent(
        id="X",
        key="unknown",
        severity=AnomalySeverity.MEDIUM,
        timestamp=datetime.now(),
        metadata={},
        score=0.5,
        contributing_detectors=[],
    )
    state = StateEstimate(
        timestamp=datetime.now(),
        label="nominal",
        confidence=1.0,
        residuals={},
        flags={},
        source_intent="idle",
    )

    # Make policy_engine.generate_policy return a dummy RemediationPolicy
    engine.generate_policy.return_value = RemediationPolicy(
        trigger_event=anomaly,
        policy_id="llm-1",
        reasoning_trace="",
        risk_assessment="TBD",
        corrective_steps=[],
        escalation_required=False,
    )

    # Act
    await agent._generate_unknown_medium_policy(anomaly, state, active_plan=None, snapshot=snapshot)

    # Assert
    engine.generate_policy.assert_awaited_once()
    called_kwargs = engine.generate_policy.call_args[1]
    assert "vision_context" in called_kwargs
    assert "knife" in called_kwargs["vision_context"]


@pytest.mark.asyncio
async def test_generate_policy_falls_back_to_full_scene_graph(monkeypatch, mock_deps):
    blackboard = mock_deps["blackboard"]
    engine = mock_deps["policy_engine"]
    agent = PolicyAgent(**mock_deps)

    # Prepare a full SceneGraph object (simple namespace or model)
    sg = SimpleNamespace(
        objects=[
            SimpleNamespace(
                id="o1", label="hand", properties={"distance_m": 0.3, "confidence": 0.9}
            )
        ],
        relationships=[],
    )
    blackboard.get_scene_graph.return_value = sg

    anomaly = AnomalyEvent(
        id="Y",
        key="unknown",
        severity=AnomalySeverity.MEDIUM,
        timestamp=datetime.now(),
        metadata={},
        score=0.5,
        contributing_detectors=[],
    )
    state = StateEstimate(
        timestamp=datetime.now(),
        label="nominal",
        confidence=1.0,
        residuals={},
        flags={},
        source_intent="idle",
    )

    engine.generate_policy.return_value = RemediationPolicy(
        trigger_event=anomaly,
        policy_id="llm-2",
        reasoning_trace="",
        risk_assessment="TBD",
        corrective_steps=[],
        escalation_required=False,
    )

    await agent._generate_unknown_medium_policy(anomaly, state, active_plan=None, snapshot=None)

    engine.generate_policy.assert_awaited_once()
    called_kwargs = engine.generate_policy.call_args[1]
    assert "vision_context" in called_kwargs
    assert "hand" in called_kwargs["vision_context"]
    blackboard.get_scene_graph.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervisor_invalid_action_forces_escalation(monkeypatch, mock_deps):
    agent = PolicyAgent(**mock_deps)
    anomaly = AnomalyEvent(
        id="Z",
        key="unknown",
        severity=AnomalySeverity.MEDIUM,
        timestamp=datetime.now(),
        metadata={},
        score=0.5,
        contributing_detectors=[],
    )

    # Create a policy with an invalid action
    bad_policy = RemediationPolicy(
        trigger_event=anomaly,
        policy_id="p-bad",
        reasoning_trace="llm",
        risk_assessment="LOW",
        corrective_steps=[
            PlanStep(
                id="1",
                description="bad",
                status=StepStatus.PENDING,
                action=AgentToolCall(action_name="nonexistent", arguments={}),
            )
        ],
        escalation_required=False,
    )

    # Patch validation to return False
    monkeypatch.setattr(agent, "_validate_action", lambda action: False)

    # Run supervisor
    ok = agent._supervise_policy_actions(bad_policy)

    assert ok is False
    assert bad_policy.escalation_required is True
    assert (
        "VALIDATION FAILURE" in bad_policy.reasoning_trace
        or "Invalid Action" in bad_policy.risk_assessment
    )


@pytest.mark.asyncio
async def test_policy_engine_error_handling(monkeypatch, mock_deps):
    agent = PolicyAgent(**mock_deps)
    engine = mock_deps["policy_engine"]
    anomaly = AnomalyEvent(
        id="E1",
        key="unknown",
        severity=AnomalySeverity.MEDIUM,
        timestamp=datetime.now(),
        metadata={},
        score=0.5,
        contributing_detectors=[],
    )
    state = StateEstimate(
        timestamp=datetime.now(),
        label="nominal",
        confidence=1.0,
        residuals={},
        flags={},
        source_intent="idle",
    )

    # Make engine.generate_policy raise
    engine.generate_policy.side_effect = RuntimeError("LLM failure")

    # Patch trace sink to capture entries
    await agent._generate_unknown_medium_policy(anomaly, state, active_plan=None, snapshot=None)

    # After failure, ensure fallback behavior: engine raised but agent should not crash.
    # If your implementation creates a fail-safe policy, assert it was published or returned.
    # For example, if generate_policy is wrapped and returns a fail-safe RemediationPolicy:
    # assert isinstance(result_policy, RemediationPolicy)


@pytest.mark.asyncio
async def test_run_loop_processes_and_breaks_on_high(monkeypatch, mock_deps):
    agent = PolicyAgent(**mock_deps)
    high = AnomalyEvent(
        id="H1",
        key="h",
        severity=AnomalySeverity.HIGH,
        timestamp=datetime.now(),
        metadata={},
        score=1.0,
        contributing_detectors=[],
    )
    mock_deps["blackboard"].get_active_anomalies.return_value = {high.id: high}
    mock_deps["blackboard"].get_latest_state_estimate.return_value = StateEstimate(
        timestamp=datetime.now(),
        label="idle",
        confidence=1.0,
        residuals={},
        flags={},
        source_intent="idle",
    )

    # Patch handler to be an AsyncMock so we can assert calls
    with patch.object(agent, "_handle_anomaly_event", new=AsyncMock()) as mock_handler:

        async def stop_sleep(delay):
            agent._loop_running = False

        with patch("asyncio.sleep", side_effect=stop_sleep):
            agent._loop_running = True
            await agent._run_loop(0.01)
            assert mock_handler.call_count == 1
