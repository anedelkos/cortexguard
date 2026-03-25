import json
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.remediation_policy import RemediationPolicy
from cortexguard.edge.models.state_estimate import StateEstimate

# Using the import path specified by the user
from cortexguard.edge.policy.mistral_policy_engine import MistralLLMPolicyEngine

# --- Fixtures ---


@pytest.fixture
def temp_high_event() -> AnomalyEvent:
    return AnomalyEvent(
        id="anomaly_1",
        key="TEMP_HIGH",
        severity=AnomalySeverity.HIGH,
        timestamp=datetime.now(UTC),
        metadata={"current_temp": 150.5, "target_temp": 120.0},
        score=1.0,
        contributing_detectors=[],
    )


@pytest.fixture
def other_event() -> AnomalyEvent:
    return AnomalyEvent(
        id="anomaly_2",
        key="OTHER_ERROR",
        severity=AnomalySeverity.LOW,
        timestamp=datetime.now(UTC),
        metadata={},
        score=0.5,
        contributing_detectors=[],
    )


@pytest.fixture
def state_estimate() -> StateEstimate:
    return StateEstimate(
        timestamp=datetime.now(UTC),
        label="NORMAL",
        confidence=1.0,
        residuals={},
        flags={},
        source_intent="IDLE",
        ttf=999.0,
    )


@pytest.fixture
def tool_catalog() -> str:
    # Adding necessary actions for policy engine logic (SET_POWER_LEVEL, SEND_NOTIFICATION, EMERGENCY_STOP)
    return json.dumps(
        [
            {
                "tool_id": "Cooling_Unit_001",
                "capabilities": [{"capability_name": "SET_POWER_LEVEL"}],
            },
            {"tool_id": "utility_tool", "capabilities": [{"capability_name": "SEND_NOTIFICATION"}]},
            {"tool_id": "grill_001", "capabilities": [{"capability_name": "EMERGENCY_STOP"}]},
        ],
        indent=2,
    )


# --- Patching for Initialization Tests ---
#
# torch / transformers are now imported lazily inside __init__ (not at module level).
# We inject mocks via sys.modules so the `import torch` / `from transformers import ...`
# statements inside __init__ pick up the mocks instead of the real packages.


def _make_torch_mock(cuda_available: bool) -> Any:
    """Build a minimal torch mock with cuda.is_available stubbed."""
    from unittest.mock import MagicMock

    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = cuda_available
    torch_mock.bfloat16 = "bfloat16"  # used as dtype argument
    return torch_mock


def _make_transformers_mock() -> tuple[Any, Any, Any]:
    """Return (MockAutoModel, MockAutoTokenizer, MockBitsAndBytesConfig)."""
    from unittest.mock import MagicMock

    return MagicMock(), MagicMock(), MagicMock()


class TestMistralLLMInit:
    def test_init_mock_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        """Tests the initialization path when use_mock=True.

        torch / transformers must NOT be imported when use_mock=True.
        """
        engine = MistralLLMPolicyEngine(use_mock=True)
        assert engine._use_mock is True
        assert "LLM is running in MOCK mode" in caplog.text

    def test_init_real_cuda_mode(self) -> None:
        """Tests real initialization on a CUDA device."""
        import sys
        import types

        torch_mock = _make_torch_mock(cuda_available=True)
        MockModel, MockTokenizer, MockBnb = _make_transformers_mock()
        transformers_mock = types.ModuleType("transformers")
        transformers_mock.__dict__["AutoModelForCausalLM"] = MockModel
        transformers_mock.__dict__["AutoTokenizer"] = MockTokenizer
        transformers_mock.__dict__["BitsAndBytesConfig"] = MockBnb

        with patch.dict(sys.modules, {"torch": torch_mock, "transformers": transformers_mock}):
            engine = MistralLLMPolicyEngine(use_mock=False)

        assert engine.device == "cuda"
        MockModel.from_pretrained.assert_called_once()
        assert "quantization_config" in MockModel.from_pretrained.call_args[1]

    def test_init_real_cpu_mode_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Tests real initialization on CPU with the expected warning."""
        import sys
        import types

        torch_mock = _make_torch_mock(cuda_available=False)
        MockModel, MockTokenizer, MockBnb = _make_transformers_mock()
        transformers_mock = types.ModuleType("transformers")
        transformers_mock.__dict__["AutoModelForCausalLM"] = MockModel
        transformers_mock.__dict__["AutoTokenizer"] = MockTokenizer
        transformers_mock.__dict__["BitsAndBytesConfig"] = MockBnb

        with patch.dict(sys.modules, {"torch": torch_mock, "transformers": transformers_mock}):
            engine = MistralLLMPolicyEngine(use_mock=False)

        assert engine.device == "cpu"
        MockModel.from_pretrained.assert_called_once()
        assert "quantization_config" in MockModel.from_pretrained.call_args[1]
        assert MockModel.from_pretrained.call_args[1]["quantization_config"] is None
        assert "Loading Mistral on CPU will be extremely slow" in caplog.text


# --- Unit Tests for Logic ---


@pytest.fixture
def mock_engine() -> MistralLLMPolicyEngine:
    """Fixture for an engine in mock mode."""
    return MistralLLMPolicyEngine(use_mock=True)


def test_format_prompt_structure(
    mock_engine: MistralLLMPolicyEngine,
    temp_high_event: AnomalyEvent,
    state_estimate: StateEstimate,
    tool_catalog: str,
) -> None:
    """Tests the structure and content of the generated prompt."""
    prompt = mock_engine._format_prompt(temp_high_event, state_estimate, tool_catalog)

    # Check Mistral instruction format
    assert prompt.startswith("[INST]")
    assert prompt.endswith("[/INST]")

    # Check System Instruction presence
    assert "You are the Policy Selector Agent." in prompt
    assert "MUST output ONLY valid JSON" in prompt

    # Check for content insertion
    assert "TEMP_HIGH" in prompt  # Anomaly key
    assert "NORMAL" in prompt  # State estimate label
    assert "Cooling_Unit_001" in prompt  # Tool catalog content

    # Check for the rule structure (FIXED: changed underscore to space to match the source string)
    assert "EMERGENCY FALLBACK sequence" in prompt  # Policy rules

    # Check for mandatory schema instructions
    assert '"corrective_steps"' in prompt
    assert '"action_name"' in prompt


def test_mock_llm_call_temp_high(
    mock_engine: MistralLLMPolicyEngine, temp_high_event: AnomalyEvent, tool_catalog: str
) -> None:
    """Tests the mock response for TEMP_HIGH (Success Case)."""
    response_json = mock_engine._mock_llm_call("dummy_prompt", temp_high_event, tool_catalog)
    response_data = json.loads(response_json)

    # Assertions updated for simplified trace and risk assessment
    assert "Rule 1.A/B was TRUE" in response_data["reasoning_trace"]
    assert response_data["risk_assessment"] == "MEDIUM"
    assert response_data["escalation_required"] is False
    assert len(response_data["corrective_steps"]) == 2

    # Check Step 1: SET_POWER_LEVEL
    step1_action = response_data["corrective_steps"][0]["action"]
    # Function name is now action_name in the mock, consistent with the parser/schema
    assert step1_action["action_name"] == "SET_POWER_LEVEL"
    assert step1_action["arguments"]["level"] == 1.0
    assert step1_action["arguments"]["device_id"] == "Cooling_Unit_001"

    # Check Step 2: SEND_NOTIFICATION
    step2_action = response_data["corrective_steps"][1]["action"]
    assert step2_action["action_name"] == "SEND_NOTIFICATION"


def test_mock_llm_call_other_error(
    mock_engine: MistralLLMPolicyEngine, other_event: AnomalyEvent, tool_catalog: str
) -> None:
    """Tests the mock fallback response (Unknown Event Case)."""
    response_json = mock_engine._mock_llm_call("dummy_prompt", other_event, tool_catalog)
    response_data = json.loads(response_json)

    # Assertions updated for simplified trace and risk assessment
    assert "Rule 2 was applied" in response_data["reasoning_trace"]
    assert response_data["risk_assessment"] == "LOW"
    assert response_data["escalation_required"] is True
    assert len(response_data["corrective_steps"]) == 1

    # Check Step 1: LOG_EVENT
    step1_action = response_data["corrective_steps"][0]["action"]
    # Function name is now action_name in the mock, consistent with the parser/schema
    assert step1_action["action_name"] == "LOG_EVENT"


def test_parse_llm_response_success(
    mock_engine: MistralLLMPolicyEngine, temp_high_event: AnomalyEvent
) -> None:
    """Tests successful parsing and Pydantic model construction."""
    # The mock response uses action_name, so this test's manual mock must also use action_name
    mock_response = """
    ```json
    {
      "reasoning_trace": "Parsing check success.",
      "risk_assessment": "LOW",
      "escalation_required": true,
      "corrective_steps": [
        {
          "description": "Do the first thing.",
          "action": {
            "action_name": "ACTION_A",
            "arguments": {"tool_id": "tool_a", "key": 1}
          }
        },
        {
          "description": "Do the second thing.",
          "action": {
            "action_name": "ACTION_B",
            "arguments": {"tool_id": "tool_b"}
          }
        }
      ]
    }
    ```
    """
    policy = mock_engine._parse_llm_response(mock_response, temp_high_event)

    assert isinstance(policy, RemediationPolicy)
    assert policy.risk_assessment == "LOW"
    assert policy.escalation_required is True
    assert len(policy.corrective_steps) == 2

    # Check action name (now 'function_name' in the AgentToolCall model)
    assert policy.corrective_steps[0].action.action_name == "ACTION_A"
    # Check arguments
    assert policy.corrective_steps[1].action.arguments["tool_id"] == "tool_b"


def test_parse_llm_response_invalid_json(
    mock_engine: MistralLLMPolicyEngine, other_event: AnomalyEvent, caplog: pytest.LogCaptureFixture
) -> None:
    """Tests the failure path with invalid JSON to hit the fail-safe policy."""
    invalid_response = "This is not valid JSON, it's just plain text"

    with caplog.at_level(logging.ERROR):
        policy = mock_engine._parse_llm_response(invalid_response, other_event)

    # Check for logging of the critical error
    assert "CRITICAL PARSING FAILURE" in policy.reasoning_trace
    assert "Failed to parse LLM JSON response" in caplog.text

    # Check that the fail-safe policy was created
    assert policy.risk_assessment == "HIGH - System safety cannot be guaranteed."
    assert policy.escalation_required is True
    # Check function name on the fail-safe step
    assert policy.corrective_steps[0].action.action_name == "EMERGENCY_SHUTDOWN"


def test_parse_llm_response_missing_fields(
    mock_engine: MistralLLMPolicyEngine, other_event: AnomalyEvent, caplog: pytest.LogCaptureFixture
) -> None:
    """Tests robust parsing when optional fields are missing."""
    minimal_response = """
    {
      "reasoning_trace": "Minimal test.",
      "risk_assessment": "UNKNOWN",
      "escalation_required": false,
      "corrective_steps": []
    }
    """
    policy = mock_engine._parse_llm_response(minimal_response, other_event)

    assert isinstance(policy, RemediationPolicy)
    assert policy.risk_assessment == "UNKNOWN"
    assert len(policy.corrective_steps) == 0
    # Check that it still successfully created a policy, even with an empty list of steps


@pytest.mark.asyncio
async def test_generate_policy_mock_path(
    mock_engine: MistralLLMPolicyEngine,
    temp_high_event: AnomalyEvent,
    state_estimate: StateEstimate,
    tool_catalog: str,
) -> None:
    """Tests the generate_policy method taking the mock execution path."""
    policy = await mock_engine.generate_policy(
        temp_high_event, state_estimate, tool_catalog, active_plan_context=""
    )

    assert isinstance(policy, RemediationPolicy)
    # Check success based on mock logic
    assert policy.risk_assessment == "MEDIUM"
    assert policy.escalation_required is False
    assert len(policy.corrective_steps) == 2

    # Check arguments on the first step (SET_POWER_LEVEL)
    assert policy.corrective_steps[0].action.action_name == "SET_POWER_LEVEL"
    assert policy.corrective_steps[0].action.arguments["device_id"] == "Cooling_Unit_001"
    assert policy.corrective_steps[0].action.arguments["level"] == 1.0

    # Check the second step (SEND_NOTIFICATION)
    assert policy.corrective_steps[1].action.action_name == "SEND_NOTIFICATION"
