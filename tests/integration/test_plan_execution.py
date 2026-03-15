import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import pytest
import yaml

from cortexguard.core.interfaces.base_controller import BaseController
from cortexguard.core.interfaces.base_step_classifier import BaseStepClassifier
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.capability_registry import CapabilityRegistry, RiskLevel
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.plan import Plan, PlanStatus, PlanStep, PlanType, StepStatus
from cortexguard.edge.step_executor import StepExecutor

logger = logging.getLogger(__name__)

CAPABILITY_REGISTRY_YAML_CONTENT = """
SLICE_ITEM:
  description: "Slices a specified food item."
  parameters:
    type: object
    properties:
      item_name: {type: string}
      tool_id: {type: string}
    required: [item_name, tool_id]

PLACE_ITEM:
  description: "Picks up an item and places it at a target location."
  parameters:
    type: object
    properties:
      item_name: {type: string}
      target_location: {type: string}
      tool_id: {type: string}
    required: [item_name, target_location, tool_id]

ALIGN_AND_STACK:
  description: "Stacks two components for final assembly."
  parameters:
    type: object
    properties:
      bottom_layer: {type: string}
      top_component: {type: string}
      tool_id: {type: string}
    required: [bottom_layer, top_component, tool_id]

DELIVER_ORDER:
  description: "Transfers a completed food item to the final service area."
  parameters:
    type: object
    properties:
      order_id: {type: string}
      pickup_area: {type: string}
    required: [order_id, pickup_area]
"""

BURGER_PLAN_YAML_CONTENT = """
goal_id: "CHZ-BURGER-002"
context:
  user_prompt: "Prepare a cheeseburger."
steps:
  - step_id: "STEP-BUN-001"
    description: "Slice the bun in half."
    function_name: "SLICE_ITEM"
    arguments:
      tool_id: "Knife_Station_A"
      item_name: "BUN"

  - step_id: "STEP-PATTY-002"
    description: "Move raw beef patty onto Grill Slot 1."
    function_name: "PLACE_ITEM"
    arguments:
      tool_id: "RoboticGripper_A"
      item_name: "RAW_PATTY"
      target_location: "Grill_Slot_1"

  - step_id: "STEP-CHEESE-003"
    description: "Place cheese slice onto the patty."
    function_name: "PLACE_ITEM"
    arguments:
      tool_id: "RoboticGripper_A"
      item_name: "CHEDDAR_SLICE"
      target_location: "Grill_Slot_1"

  - step_id: "STEP-SERVE-004"
    description: "Transfer the completed burger to the customer service window."
    function_name: "DELIVER_ORDER"
    arguments:
      order_id: "CHZ-BURGER-002"
      pickup_area: "ServiceWindow_1"
"""


class MockController(BaseController):
    """Logs executed calls for assertion."""

    def __init__(
        self, execute_override: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    ) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self._execute_override = execute_override

    async def execute(self, function_name: str, parameters: dict[str, Any]) -> None:
        self.executed.append((function_name, parameters))
        if self._execute_override:
            await self._execute_override(function_name, parameters)


class MockFailingController(BaseController):
    """
    Fails a specific action on the first call, succeeds after.
    The test should assert on 2 executions (1 fail, 1 success).
    """

    def __init__(self, fail_function_name: str) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self._fail_function_name = fail_function_name
        self._attempt_count: dict[str, int] = {}

    async def execute(self, function_name: str, parameters: dict[str, Any]) -> None:
        self.executed.append((function_name, parameters))

        current_attempts = self._attempt_count.get(function_name, 0) + 1
        self._attempt_count[function_name] = current_attempts

        # Fail only on the first attempt
        if function_name == self._fail_function_name and current_attempts == 1:
            raise ConnectionError(f"Simulated failure for {function_name}")

        # Succeed otherwise (on attempt 2 and higher)
        return


class MockStepClassifier(BaseStepClassifier):
    """Always returns COMPLETED."""

    def classify_completion_status(self, step: PlanStep) -> StepStatus:
        return StepStatus.COMPLETED


class MockValidatingRegistry(CapabilityRegistry):
    """
    Registry that can be configured to fail validation for a specific function.
    """

    def __init__(self, validation_fail_function: str | None = None, **data: Any) -> None:
        super().__init__(**data)
        self._validation_fail_function = validation_fail_function

    def validate_call(
        self, function_name: str, arguments: dict[str, Any]
    ) -> tuple[bool, RiskLevel]:
        # Simulate a schema validation failure for the configured function
        if function_name == self._validation_fail_function:
            # Return a conservative denial with HIGH risk
            return False, RiskLevel.HIGH

        # Otherwise delegate to the real implementation and return its result
        return super().validate_call(function_name, arguments)


def _create_plan_from_yaml(plan_yaml_content: str) -> Plan:
    """Helper to create a Plan instance by loading YAML from the provided content string."""
    raw_plan = yaml.safe_load(plan_yaml_content)

    plan_steps = []
    for s in raw_plan.get("steps", []):
        tool_call = AgentToolCall(
            action_name=s["function_name"],
            arguments=s["arguments"],
        )
        plan_steps.append(
            PlanStep(
                id=s["step_id"],
                description=s["description"],
                action=tool_call,
                status=StepStatus.PENDING,
            )
        )

    context = GoalContext(
        goal_id=raw_plan["goal_id"],
        user_prompt=raw_plan["context"]["user_prompt"],
        intent="cook a recipe",
    )

    return Plan(
        plan_id="test-plan-1",
        context=context,
        plan_type=PlanType.RECIPE,
        steps=plan_steps,
        status=PlanStatus.PENDING,
    )


def _create_registry(
    registry_yaml_content: str, mock_fail_function: str | None = None
) -> MockValidatingRegistry:
    """
    Helper to create and load the CapabilityRegistry from YAML content string.
    """
    capabilities_dict = yaml.safe_load(registry_yaml_content)

    registry = MockValidatingRegistry(
        validation_fail_function=mock_fail_function, capabilities=capabilities_dict
    )
    return registry


@pytest.mark.asyncio
async def test_successful_plan_execution() -> None:
    """Tests the StepExecutor successfully executes all steps in a plan."""
    blackboard = Blackboard()
    controller = MockController()
    classifier = MockStepClassifier()
    registry = _create_registry(registry_yaml_content=CAPABILITY_REGISTRY_YAML_CONTENT)
    plan = _create_plan_from_yaml(plan_yaml_content=BURGER_PLAN_YAML_CONTENT)

    executor = StepExecutor(
        blackboard=blackboard,
        step_classifier=classifier,
        capability_registry=registry,
        controller=controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )
    await executor.start()

    for step in plan.steps:
        await blackboard.set_current_step(step)

        await executor.execute_step(step)

        assert step.status == StepStatus.COMPLETED

    await executor.stop()

    expected_calls = [
        ("SLICE_ITEM", {"tool_id": "Knife_Station_A", "item_name": "BUN"}),
        (
            "PLACE_ITEM",
            {
                "tool_id": "RoboticGripper_A",
                "item_name": "RAW_PATTY",
                "target_location": "Grill_Slot_1",
            },
        ),
        (
            "PLACE_ITEM",
            {
                "tool_id": "RoboticGripper_A",
                "item_name": "CHEDDAR_SLICE",
                "target_location": "Grill_Slot_1",
            },
        ),
        ("DELIVER_ORDER", {"order_id": "CHZ-BURGER-002", "pickup_area": "ServiceWindow_1"}),
    ]

    assert len(controller.executed) == 4
    for i, (name, args) in enumerate(expected_calls):
        assert controller.executed[i][0] == name
        assert controller.executed[i][1] == args


@pytest.mark.asyncio
async def test_step_retry_on_controller_failure() -> None:
    """Tests that a step automatically retries after a controller execution failure."""

    # 1. Setup (unchanged)
    blackboard = Blackboard()
    # MockFailingController must be set up to fail the first call and succeed the second
    failing_controller = MockFailingController(fail_function_name="PLACE_ITEM")
    classifier = MockStepClassifier()
    registry = _create_registry(registry_yaml_content=CAPABILITY_REGISTRY_YAML_CONTENT)
    plan = _create_plan_from_yaml(plan_yaml_content=BURGER_PLAN_YAML_CONTENT)

    retry_step = plan.steps[1]

    executor = StepExecutor(
        blackboard=blackboard,
        step_classifier=classifier,
        capability_registry=registry,
        controller=failing_controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )

    # 2. Start the background loop and submit the step
    await executor.start()
    # This makes the step visible to the background executor loop.
    await blackboard.set_current_step(retry_step)

    # 3. Wait for the background loop to finish the step
    timeout = 1.0
    start_time = asyncio.get_event_loop().time()
    current_status = StepStatus.PENDING

    # Poll until the step status is terminal (COMPLETED or FAILED)
    while (
        current_status not in (StepStatus.COMPLETED, StepStatus.FAILED)
        and (asyncio.get_event_loop().time() - start_time) < timeout
    ):
        # Wait slightly longer than the executor's poll interval (0.05s) to ensure the loop runs
        await asyncio.sleep(0.06)
        current_step = await blackboard.get_current_step()
        if current_step is not None:
            current_status = current_step.status

    # Stop the executor loop
    await executor.stop()

    # 4. Assertions

    # Should be 2 executions: 1 failed attempt, 1 successful attempt
    assert len(failing_controller.executed) == 2
    assert failing_controller.executed[0][0] == "PLACE_ITEM"
    assert failing_controller.executed[1][0] == "PLACE_ITEM"

    final_step = await blackboard.get_current_step()
    assert final_step is not None
    assert final_step.status == StepStatus.COMPLETED
    assert final_step.attempts == 2

    trace = blackboard.reasoning_traces
    execution_fail_entries = [e for e in trace if e.event_type == "EXECUTION_FAILED"]
    step_completed_entries = [e for e in trace if e.event_type == "STEP_COMPLETED"]

    # Only one failure entry is expected (from the first attempt)
    assert len(execution_fail_entries) == 1
    assert execution_fail_entries[0].reasoning_text.startswith("Controller failed on PLACE_ITEM")

    assert len(step_completed_entries) == 1
    assert step_completed_entries[0].reasoning_text.startswith(
        f"Step {retry_step.id} successfully completed."
    )


@pytest.mark.asyncio
async def test_execution_blocked_by_anomaly() -> None:
    """Tests that execution is blocked when a medium-severity anomaly is present."""
    blackboard = Blackboard()
    controller = MockController()
    classifier = MockStepClassifier()
    registry = _create_registry(registry_yaml_content=CAPABILITY_REGISTRY_YAML_CONTENT)
    plan = _create_plan_from_yaml(plan_yaml_content=BURGER_PLAN_YAML_CONTENT)
    target_step = plan.steps[0]

    anomaly = AnomalyEvent(
        id="CRITICAL_TEMP_001",
        key="CRITICAL_TEMP",
        timestamp=datetime.now(UTC),
        severity=AnomalySeverity.MEDIUM,
        score=0.9,
        contributing_detectors=["temp_sensor_detector_v1"],
        metadata={"description": "Grill temperature too high, stopping execution."},
    )
    await blackboard.set_anomaly(anomaly)

    executor = StepExecutor(
        blackboard=blackboard,
        step_classifier=classifier,
        capability_registry=registry,
        controller=controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )
    await executor.start()
    await blackboard.set_current_step(target_step)

    await executor.execute_step(target_step)

    await executor.stop()

    assert len(controller.executed) == 0
    assert target_step.status == StepStatus.PENDING

    trace = blackboard.reasoning_traces
    assert len(trace) == 1
    assert trace[0].event_type == "ANOMALY_MEDIUM"


@pytest.mark.asyncio
async def test_validation_failure() -> None:
    """Tests that execution is halted if the CapabilityRegistry rejects the call schema."""
    blackboard = Blackboard()
    controller = MockController()
    registry = _create_registry(
        registry_yaml_content=CAPABILITY_REGISTRY_YAML_CONTENT, mock_fail_function="SLICE_ITEM"
    )
    classifier = MockStepClassifier()
    plan = _create_plan_from_yaml(plan_yaml_content=BURGER_PLAN_YAML_CONTENT)
    target_step = plan.steps[0]

    executor = StepExecutor(
        blackboard=blackboard,
        step_classifier=classifier,
        capability_registry=registry,
        controller=controller,
        default_max_retries=1,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_retry_delay=0.01,
    )
    await executor.start()
    await blackboard.set_current_step(target_step)

    await executor.execute_step(target_step)

    await executor.stop()

    assert len(controller.executed) == 0
    assert target_step.status == StepStatus.FAILED

    trace = blackboard.reasoning_traces
    validation_fail_entries = [e for e in trace if e.event_type == "VALIDATION_FAILED"]

    assert len(validation_fail_entries) == 1
    assert validation_fail_entries[0].reasoning_text.startswith("Command for SLICE_ITEM invalid:")
