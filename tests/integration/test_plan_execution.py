import asyncio
from datetime import datetime
from io import StringIO
from typing import Any

import pytest
import yaml

from kitchenwatch.core.action_registry import ActionRegistry
from kitchenwatch.core.interfaces.base_controller import BaseController
from kitchenwatch.core.interfaces.base_step_classifier import BaseStepClassifier
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.plan import Plan, PlanStep, PlanType, StepStatus
from kitchenwatch.edge.orchestrator import Orchestrator
from kitchenwatch.edge.step_executor import StepExecutor


# ----------------------------
# Mocks
# ----------------------------
class MockController(BaseController):
    """Deterministic mock controller that just logs executed primitives"""

    def __init__(self) -> None:
        self.executed: list[Any] = []

    async def execute(self, primitive_name: str, parameters: dict[str, Any]) -> None:
        self.executed.append((primitive_name, parameters))
        print(f"[MockController] Executing: {primitive_name} with {parameters}")


class MockStepClassifier(BaseStepClassifier):
    """Always returns COMPLETED"""

    def classify_completion_status(self, step: PlanStep) -> StepStatus:
        return StepStatus.COMPLETED


# ----------------------------
# In-memory YAML
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
  name: flip_burger
- id: step2
  name: flip_burger
"""


# ----------------------------
# Deterministic integration test
# ----------------------------
@pytest.mark.asyncio
async def test_orchestrator_executor_integration_deterministic() -> None:
    # Blackboard
    blackboard = Blackboard()

    # Mocks
    controller = MockController()
    classifier = MockStepClassifier()

    # Load action registry
    registry = ActionRegistry(controller)
    registry._actions = yaml.safe_load(StringIO(ACTION_REGISTRY_YAML))

    # Create plan steps
    recipe_steps = yaml.safe_load(StringIO(RECIPE_YAML))
    plan_steps = [PlanStep(id=s["id"], name=s["name"]) for s in recipe_steps]

    # Create plan
    plan = Plan(
        plan_id="plan1",
        steps=plan_steps,
        plan_type=PlanType.RECIPE,
        version="0.1.0",
        goal="Test goal",
        created_at=datetime.now(),
    )

    # Setup orchestrator
    orchestrator = Orchestrator(blackboard)

    # Setup executor
    executor = StepExecutor(
        blackboard=blackboard, step_classifier=classifier, action_registry=registry
    )
    await executor.start()

    # Submit plan to blackboard/orchestrator
    await orchestrator.add_plan(plan)
    await orchestrator.start(tick_interval=0.01)

    # Allow the orchestrator to pull plan & set first step
    await asyncio.sleep(0.05)

    # Deterministic loop: execute each pending step manually
    while any(step.status != StepStatus.COMPLETED for step in plan.steps):
        current_step = await blackboard.get_current_step()
        if current_step:
            await executor.execute_step(current_step)
            await asyncio.sleep(0.01)  # allow orchestrator to advance
        else:
            await asyncio.sleep(0.01)

    await orchestrator.stop()
    await executor.stop()

    # Assertions
    for step in plan.steps:
        assert step.status == StepStatus.COMPLETED

    # Each primitive executed for both steps
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
