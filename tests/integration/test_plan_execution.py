from io import StringIO
from typing import Any

import pytest
import yaml

from kitchenwatch.core.action_registry import ActionRegistry
from kitchenwatch.core.interfaces.base_controller import BaseController
from kitchenwatch.core.interfaces.base_step_classifier import BaseStepClassifier
from kitchenwatch.edge.models.action import Action, ActionStatus
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.goal import GoalContext
from kitchenwatch.edge.models.plan import Plan, PlanStatus, PlanStep, PlanType, StepStatus
from kitchenwatch.edge.step_executor import StepExecutor


# ----------------------------
# Mocks
# ----------------------------
class MockController(BaseController):
    """Deterministic mock controller that just logs executed primitives"""

    def __init__(self) -> None:
        self.executed: list[Any] = []

    # Use explicit Dict for mypy compatibility
    async def execute(self, primitive_name: str, parameters: dict[str, Any]) -> None:
        self.executed.append((primitive_name, parameters))
        # No sleeps here for deterministic execution


class MockStepClassifier(BaseStepClassifier):
    """Always returns COMPLETED"""

    def classify_completion_status(self, step: PlanStep) -> StepStatus:
        return StepStatus.COMPLETED


# ----------------------------
# In-memory YAML Definitions
# ----------------------------
ACTION_REGISTRY_YAML = """
flip_burger:
  - primitive: move_to_burger
  - primitive: lower_tool
  - primitive: grasp_burger
  - primitive: flip_motion
  - primitive: release_burger
  - primitive: retreat
"""

RECIPE_YAML = """
- id: step1
  capability_name: flip_burger
- id: step2
  capability_name: flip_burger
"""


# Mypy fix: Explicitly define the type parameter for dict in the argument list
def _create_plan_steps(recipe_steps: list[dict[str, Any]]) -> list[PlanStep]:
    """Helper function to create the correct PlanSteps from YAML."""
    plan_steps = []
    for s in recipe_steps:
        # 1. Create the Action instance (using the new key 'capability_name')
        action_instance = Action(
            id=f"{s['id']}_action",
            tool_id="spatula_robot_arm",
            capability=s["capability_name"],
            arguments={"target_item": "burger", "power_level": "medium"},
            status=ActionStatus.PENDING,
        )

        # 2. Create the PlanStep instance, adhering to the new schema
        plan_steps.append(
            PlanStep(
                id=s["id"],
                description=f"Execute the capability: {action_instance.capability} using {action_instance.tool_id}",
                action=action_instance,
                status=StepStatus.PENDING,
            )
        )
    return plan_steps


# ----------------------------------------------------------------------
# Deterministic integration test (Manual Step Advancement)
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_executor_manual_step_advancement() -> None:
    """
    Tests the StepExecutor by manually setting the current step on the Blackboard
    and advancing through the plan, simulating the Orchestrator's control flow.
    """
    # Blackboard
    blackboard = Blackboard()

    # Mocks
    controller = MockController()
    classifier = MockStepClassifier()

    # Load action registry
    registry = ActionRegistry(controller)
    registry._actions = yaml.safe_load(StringIO(ACTION_REGISTRY_YAML))

    # Create a mock GoalContext
    mock_context = GoalContext(
        goal_id="test_goal_123", user_prompt="Please cook the recipe for me."
    )

    # Create plan steps using the helper
    recipe_steps = yaml.safe_load(StringIO(RECIPE_YAML))
    plan_steps = _create_plan_steps(recipe_steps)

    # Create plan
    plan = Plan(
        plan_id="plan1",
        context=mock_context,
        plan_type=PlanType.RECIPE,
        steps=plan_steps,
        status=PlanStatus.PENDING,
    )

    # Setup executor
    executor = StepExecutor(
        blackboard=blackboard, step_classifier=classifier, action_registry=registry
    )
    await executor.start()

    # ----------------------------------------------------------------------
    # MANUAL EXECUTION FLOW (Simulates Orchestrator advancement)
    # ----------------------------------------------------------------------

    # 1. Simulate the Orchestrator picking up the plan and setting it to RUNNING
    plan.status = PlanStatus.RUNNING
    await blackboard.set_current_plan(plan)  # Ensure plan is registered

    # 2. Iterate through steps manually
    for i, step in enumerate(plan.steps):
        # A. Simulate Orchestrator setting the current step on the Blackboard
        await blackboard.set_current_step(step)

        # B. The StepExecutor picks up and executes the step
        current_step = await blackboard.get_current_step()

        # Mypy fix: Assert current_step is not None to narrow the type
        assert (
            current_step is not None
        ), f"Current step should not be None before executing step {i + 1}"

        # Execute the step using the StepExecutor's core method
        await executor.execute_step(current_step)

        # C. Simulate Orchestrator/StepExecutor completing the step
        plan.steps[i].status = StepStatus.COMPLETED

        # D. Simulate Orchestrator clearing the step before setting the next one
        await blackboard.set_current_step(None)

    # 3. Finalize the plan status
    plan.status = PlanStatus.COMPLETED

    # Stop the executor
    await executor.stop()

    # ----------------------------------------------------------------------
    # Assertions
    # ----------------------------------------------------------------------

    # Assert plan completion
    assert plan.status == PlanStatus.COMPLETED
    for step in plan.steps:
        assert step.status == StepStatus.COMPLETED

    # Assert primitives executed
    expected_primitives = [
        "move_to_burger",
        "lower_tool",
        "grasp_burger",
        "flip_motion",
        "release_burger",
        "retreat",
    ] * len(plan.steps)

    executed_primitives = [p[0] for p in controller.executed]
    assert executed_primitives == expected_primitives
