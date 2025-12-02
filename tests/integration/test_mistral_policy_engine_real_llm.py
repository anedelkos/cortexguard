import asyncio
import json
import os
from datetime import datetime

import pytest

from kitchenwatch.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from kitchenwatch.edge.models.state_estimate import StateEstimate
from kitchenwatch.edge.policy.mistral_policy_engine import MistralLLMPolicyEngine

# --- Configuration for Integration Tests ---
MAX_LATENCY_SECONDS = 20.0
# Define the environment variable that MUST be set to run these tests
RUN_SLOW_LLM_TESTS = os.environ.get("RUN_SLOW_LLM_TESTS", "False").lower() == "true"


# Fixture setup for the necessary inputs
@pytest.fixture
def anomaly_event_fixture() -> AnomalyEvent:
    """Provides a sample anomaly event for testing (High Temp)."""
    return AnomalyEvent(
        id="anomaly_1",
        key="TEMP_HIGH",
        severity=AnomalySeverity.HIGH,
        timestamp=datetime.now(),
        metadata={
            "sensor_id": "temp_sensor_001",
            "current_temp": 150.5,
            "target_temp": 120.0,
            "units": "C",
        },
        score=1.0,
        contributing_detectors=["HardLimitDetector"],
    )


@pytest.fixture
def state_estimate_fixture() -> StateEstimate:
    """
    Provides a detailed state estimate context for testing.
    """
    return StateEstimate(
        timestamp=datetime.now(),
        label="HIGH_TEMPERATURE_DEVIATION",
        confidence=0.98,
        residuals={
            "temp_sensor_001_deviation": 30.5,
            "temp_sensor_001_value": 150.5,
        },
        flags={
            "system_mode": "ACTIVE_PROCESS",
            "device_mapping": {
                "Grill_Station_1": {
                    "purpose": "Primary heat source for COOKING_PHASE_1",
                    "controlled_by": ["temp_sensor_001"],
                },
                "Cooling_Unit_001": {
                    "purpose": "Thermal regulation",
                    "status": "OPERATIONAL",
                    "power_level": 0.5,
                },
            },
        },
        source_intent="COOKING_PHASE_1",
        ttf=60.0,
    )


@pytest.fixture
def action_catalog_json_fixture() -> str:
    """
    Provides a rich JSON string representing the functions catalog
    required by the LLM Policy Engine for function calling, using the new set
    of kitchen automation and remediation capabilities.
    """
    functions = [
        {
            "action_name": "SLICE_ITEM",
            "description": "Slices a specified food item using an available cutting tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                    "tool_id": {
                        "type": "string",
                        "enum": ["Knife_Station_A", "Laser_Cutter_B"],
                    },
                },
                "required": ["item_name", "tool_id"],
            },
        },
        {
            "action_name": "GRILL_ITEM",
            "description": "Cooks a food item on a specific grill for a defined duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                    "duration_s": {"type": "integer"},
                    "tool_id": {
                        "type": "string",
                        "enum": ["Grill_Station_1", "Grill_Station_2", "Backup_Grill"],
                    },
                },
                "required": ["item_name", "duration_s", "tool_id"],
            },
        },
        {
            "action_name": "PLACE_ITEM",
            "description": "Picks up an item and places it at a target location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                    "target_location": {"type": "string"},
                    "tool_id": {
                        "type": "string",
                        "enum": ["RoboticGripper_A", "RoboticGripper_B", "RoboticSpatula_B"],
                    },
                },
                "required": ["item_name", "target_location", "tool_id"],
            },
        },
        {
            "action_name": "ALIGN_AND_STACK",
            "description": "Stacks two components for final assembly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bottom_layer": {"type": "string"},
                    "top_component": {"type": "string"},
                    "tool_id": {"type": "string", "enum": ["RoboticArm_Mover"]},
                },
                "required": ["bottom_layer", "top_component", "tool_id"],
            },
        },
        {
            "action_name": "DELIVER_ORDER",
            "description": "Transfers a completed food item to the final service area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "pickup_area": {"type": "string"},
                    "tool_id": {"type": "string", "enum": ["ServingTray_Arm"]},
                },
                "required": ["order_id", "pickup_area", "tool_id"],
            },
        },
        # --- Anomaly Remediation Capabilities ---
        {
            "action_name": "EMERGENCY_STOP",
            "description": "Immediately cuts power to a critical device when an extreme hazard is detected.",
            "parameters": {
                "type": "object",
                "properties": {"device_id": {"type": "string"}},
                "required": ["device_id"],
            },
        },
        {
            "action_name": "RESET_DEVICE",
            "description": "Performs a safe power cycle on a non-critical device to clear transient errors.",
            "parameters": {
                "type": "object",
                "properties": {"device_id": {"type": "string"}},
                "required": ["device_id"],
            },
        },
        {
            "action_name": "SEND_ALERT",
            "description": "Sends a notification to a human supervisor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "enum": ["supervisor", "log_system"]},
                    "message": {"type": "string"},
                },
                "required": ["recipient", "message"],
            },
        },
    ]
    return json.dumps(functions, indent=2)


@pytest.fixture
def action_catalog_with_cooling_action_fixture(action_catalog_json_fixture: str) -> str:
    """
    Provides the action catalog *including* the required SET_POWER_LEVEL action
    that enables controlled cooling.
    """
    base_actions = json.loads(action_catalog_json_fixture)
    new_action = {
        "action_name": "SET_POWER_LEVEL",
        "description": "Sets the operational power level (0.0 to 1.0) for a device like a cooling unit.",
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "power_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["device_id", "power_level"],
        },
    }
    base_actions.append(new_action)
    return json.dumps(base_actions, indent=2)


@pytest.fixture(scope="module")
def real_llm_engine() -> MistralLLMPolicyEngine:
    """
    Initializes the real MistralLLMPolicyEngine. This is the slowest part.
    It will skip if the RUN_SLOW_LLM_TESTS environment variable is not set to 'true'.
    """
    if not RUN_SLOW_LLM_TESTS:
        pytest.skip("Skipping real LLM tests. Set RUN_SLOW_LLM_TESTS=True to enable.")

    try:
        # Setting use_mock=False forces the loading of the full model.
        engine = MistralLLMPolicyEngine(use_mock=False)
        return engine
    except Exception as e:
        # Skip if initialization fails (e.g., VRAM/CUDA unavailable)
        pytest.skip(f"Failed to initialize real LLM (VRAM/CUDA issue?): {e}")


@pytest.mark.llm_slow
@pytest.mark.integration
def test_real_llm_schema_validation(
    real_llm_engine: MistralLLMPolicyEngine,
    anomaly_event_fixture: AnomalyEvent,
    state_estimate_fixture: StateEstimate,
    action_catalog_json_fixture: str,
) -> None:
    """
    Verifies that the LLM's raw JSON output is correctly parsed into the
    RemediationPolicy Pydantic model.
    """
    print("\n--- Starting Real LLM Schema Validation Test ---")

    # 1. Run the generation
    policy = asyncio.run(
        real_llm_engine.generate_policy(
            anomaly_event_fixture,
            state_estimate_fixture,
            action_catalog_json=action_catalog_json_fixture,
            active_plan_context="",
        )
    )

    # 2. Assert detailed schema fields are populated
    assert policy.policy_id is not None
    assert isinstance(policy.reasoning_trace, str)
    assert policy.reasoning_trace, "Reasoning trace should not be empty."
    assert policy.risk_assessment in [
        "LOW",
        "MEDIUM",
        "HIGH",
        "UNKNOWN",
    ], "Risk assessment must be categorized."
    assert policy.corrective_steps, "LLM must suggest at least one corrective step."

    # 3. Assert deep nested structure (PlanStep and Action)
    # The anomaly is TEMP_HIGH, so we expect a remediation action (EMERGENCY_STOP or RESET_DEVICE)
    expected_remediation_actions = ["EMERGENCY_STOP", "RESET_DEVICE", "SEND_ALERT"]
    found_remediation = False

    for step in policy.corrective_steps:
        assert isinstance(step.description, str)
        assert step.action.arguments, "Action arguments must be populated."

        assert step.action.action_name != "NO_OP", "LLM should suggest a specific capability."

        if step.action.action_name in expected_remediation_actions:
            found_remediation = True

            # For EMERGENCY_STOP/RESET_DEVICE, check if device_id is present
            if step.action.action_name in ["EMERGENCY_STOP", "RESET_DEVICE"]:
                assert "device_id" in step.action.arguments
                # Check that the device_id argument value is plausible
                assert isinstance(step.action.arguments["device_id"], str)

            # For SEND_ALERT, check if recipient is present
            if step.action.action_name == "SEND_ALERT":
                assert "recipient" in step.action.arguments

    # Assert that at least one of the expected remediation actions was called
    assert found_remediation, (
        f"LLM must use one of the remediation capabilities: {expected_remediation_actions}. "
        f"Found: {[s.action.action_name for s in policy.corrective_steps]}"
    )

    print(
        "Schema Validation Passed: LLM output adheres to the Policy and Action model structure and uses the new actions."
    )


@pytest.mark.llm_slow
@pytest.mark.integration
def test_real_llm_cooling_policy(
    real_llm_engine: MistralLLMPolicyEngine,
    anomaly_event_fixture: AnomalyEvent,
    state_estimate_fixture: StateEstimate,
    action_catalog_with_cooling_action_fixture: str,
) -> None:
    """
    Verifies that when the required cooling control action (SET_POWER_LEVEL) is present,
    the LLM prioritizes controlled mitigation (cooling) before resorting to
    EMERGENCY_STOP, given the anomaly is still in the early stages (TTF=60s).
    """
    print("\n--- Starting Real LLM Cooling Policy Test (With Cooling Action Available) ---")

    # 1. Run the generation
    policy = asyncio.run(
        real_llm_engine.generate_policy(
            anomaly_event_fixture,
            state_estimate_fixture,
            action_catalog_json=action_catalog_with_cooling_action_fixture,
            active_plan_context="",
        )
    )

    # 2. Assert that the LLM generated a controlled cooling plan
    assert policy.corrective_steps, "LLM must suggest at least one corrective step."

    # Step 1: SET_POWER_LEVEL on the cooler
    step1 = policy.corrective_steps[0]

    # Assert LLM correctly chose the controlled cooling action first
    assert (
        step1.action.action_name == "SET_POWER_LEVEL"
    ), "Step 1 must be SET_POWER_LEVEL to attempt cooling."
    assert (
        step1.action.arguments.get("device_id") == "Cooling_Unit_001"
    ), "SET_POWER_LEVEL must target the cooling unit."
    # We expect it to be set to max power (1.0)
    assert (
        step1.action.arguments.get("power_level") == 1.0
    ), "Cooling power level must be set to 1.0 (max)."

    # Step 2 (Optional, but likely): Check for a follow-up action (like an alert or a stop)
    if len(policy.corrective_steps) > 1:
        step2 = policy.corrective_steps[1]
        # The second step should be an ALERT or an EMERGENCY_STOP as a fail-safe
        assert step2.action.action_name in [
            "SEND_ALERT",
            "EMERGENCY_STOP",
        ], "Step 2 must be an alert or a fail-safe stop."

    print(
        "Cooling Policy Test Passed: LLM correctly prioritized SET_POWER_LEVEL when the action was available."
    )


@pytest.mark.llm_slow
@pytest.mark.integration
def test_real_llm_emergency_stop_fallback(
    real_llm_engine: MistralLLMPolicyEngine,
    anomaly_event_fixture: AnomalyEvent,
    state_estimate_fixture: StateEstimate,
    action_catalog_json_fixture: str,
) -> None:
    """
    Verifies the "Emergency Fallback" path (Decision Hierarchy Step 2) is followed:
    action is missing, so LLM escalates to EMERGENCY_STOP on the Primary Heat Source.
    """
    print("\n--- Starting Real LLM Emergency Stop Fallback Test (Cooling Action Missing) ---")

    # 1. Run the generation
    policy = asyncio.run(
        real_llm_engine.generate_policy(
            anomaly_event_fixture,
            state_estimate_fixture,
            action_catalog_json=action_catalog_json_fixture,
            active_plan_context="",
        )
    )

    assert policy.corrective_steps, "LLM must suggest at least one corrective step."

    # Step 1: EMERGENCY_STOP on the heat source
    step1 = policy.corrective_steps[0]

    # Assert LLM correctly chose the emergency action first
    assert (
        step1.action.action_name == "EMERGENCY_STOP"
    ), "Step 1 must be EMERGENCY_STOP when SET_POWER_LEVEL is missing."
    assert (
        step1.action.arguments.get("device_id") == "Grill_Station_1"
    ), "EMERGENCY_STOP must target the primary heat source."

    # Assert the risk is correctly assessed as HIGH for this escalation
    assert policy.risk_assessment == "HIGH"

    # Assert a follow-up alert is included
    assert any(
        step.action.action_name == "SEND_ALERT" for step in policy.corrective_steps
    ), "A SEND_ALERT action must be included in the fallback plan."

    print(
        "Emergency Fallback Test Passed: LLM correctly escalated to EMERGENCY_STOP on the heat source."
    )


@pytest.mark.llm_slow
@pytest.mark.integration
def test_real_llm_actuator_target_grounding(
    real_llm_engine: MistralLLMPolicyEngine,
    anomaly_event_fixture: AnomalyEvent,
    state_estimate_fixture: StateEstimate,
    action_catalog_with_cooling_action_fixture: str,
) -> None:
    """
    Verifies the grounding rule: 'NEVER target a sensor... Only target Actuators.'
    The LLM must use the sensor ID ('temp_sensor_001') in the anomaly metadata
    to infer the related ACTUATOR ('Cooling_Unit_001') for the action argument.
    """
    print("\n--- Starting Real LLM Actuator Target Grounding Test ---")

    # 1. Run the generation (using the cooling action catalog)
    policy = asyncio.run(
        real_llm_engine.generate_policy(
            anomaly_event_fixture,
            state_estimate_fixture,
            action_catalog_json=action_catalog_with_cooling_action_fixture,
            active_plan_context="",
        )
    )

    # We expect the controlled fix (SET_POWER_LEVEL)
    assert policy.corrective_steps, "LLM must suggest at least one corrective step."

    step1 = policy.corrective_steps[0]

    # 2. Assert the action targets an actuator, not the sensor ID.
    assert step1.action.action_name == "SET_POWER_LEVEL", "Must use controlled fix."

    device_id = step1.action.arguments.get("device_id")

    # Assert the target is the correct actuator ID from the device_mapping
    assert device_id == "Cooling_Unit_001", "Action must target the actuator ID (Cooling_Unit_001)."

    # Assert the target is NOT the sensor ID from the anomaly metadata
    assert device_id != anomaly_event_fixture.metadata.get(
        "sensor_id"
    ), "Action must NOT target the sensor ID (temp_sensor_001) as per grounding rules."

    print(
        "Actuator Target Grounding Test Passed: LLM correctly identified and targeted the actuator."
    )
