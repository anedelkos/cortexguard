from datetime import UTC, datetime
from typing import Any

import pytest

from kitchenwatch.edge.arbiter import Arbiter
from kitchenwatch.edge.models.agent_tool_call import AgentToolCall
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.capability_registry import (
    CapabilityRegistry,
    FunctionSchema,
    RiskLevel,
)
from kitchenwatch.edge.models.plan import PlanStep, StepStatus
from kitchenwatch.edge.models.scene_graph import SceneGraph, SceneObject, SceneRelationship
from kitchenwatch.edge.models.state_estimate import StateEstimate
from kitchenwatch.edge.orchestrator import Orchestrator
from kitchenwatch.edge.safety_agent import SafetyAgent
from kitchenwatch.edge.step_executor import StepExecutor


# --- Dummy controller and classifier ---
class DummyController:
    async def execute(self, name: str, args: dict[str, Any]) -> None:
        return

    async def emergency_stop(self) -> None:
        return


class DummyClassifier:
    def classify_completion_status(self, step: PlanStep) -> StepStatus:
        return StepStatus.COMPLETED


class DummyCapabilityRegistry(CapabilityRegistry):
    def validate_call(self, name: str, args: dict[str, Any]) -> tuple[bool, Any]:
        return True, type("Risk", (), {"name": "LOW"})()

    def get_function_schema(self, name: str) -> FunctionSchema:
        # Return a minimal valid FunctionSchema instance
        return FunctionSchema(
            description=f"Dummy schema for {name}",
            parameters={"type": "object", "properties": {}},
            risk_level=RiskLevel.LOW,
            pre_conditions=[],
            post_effects=[],
        )


# --- Integration orchestrator wrapper ---
class SafetyOrchestrator(Orchestrator):
    def __init__(
        self,
        blackboard: Blackboard,
        arbiter: Arbiter,
        safety_agent: SafetyAgent,
        executor: StepExecutor,
    ) -> None:
        # Forward arbiter and safety_agent to the base Orchestrator
        super().__init__(blackboard, arbiter, safety_agent, tick_interval=1)
        self.arbiter = arbiter
        self.safety_agent = safety_agent
        self.executor = executor

    async def run_step(self, step: PlanStep, state: StateEstimate) -> str:
        cmd = await self.safety_agent.execute_safety_check(state)
        if cmd.action != "NOMINAL":
            await self.arbiter.emergency_stop(reason=cmd.reason or "hazard detected")
            return "STOPPED"
        await self.executor.execute_step(step)
        return step.status.name


# --- Tests ---
@pytest.mark.asyncio
async def test_safety_agent_triggers_emergency_stop_in_orchestrator():
    blackboard = Blackboard()
    controller = DummyController()
    classifier = DummyClassifier()
    registry = DummyCapabilityRegistry()
    arbiter = Arbiter(blackboard, registry, controller)
    safety_agent = SafetyAgent(blackboard)
    executor = StepExecutor(
        blackboard,
        classifier,
        registry,
        controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )
    orchestrator = SafetyOrchestrator(blackboard, arbiter, safety_agent, executor)

    # Hazardous scene: hand near blade
    hand = SceneObject(
        id="hand1",
        label="Human_Hand",
        properties={},
        location_2d=[0.0, 0.0, 0.1, 0.1],
        pose_3d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    blade = SceneObject(
        id="blade1",
        label="Blade",
        properties={"safety_critical": True},
        location_2d=[0.2, 0.2, 0.3, 0.3],
        pose_3d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    rel = SceneRelationship(source_id="hand1", target_id="blade1", relationship="near")
    scene = SceneGraph(objects=[hand, blade], relationships=[rel])
    await blackboard.set_scene_graph(scene)

    step = PlanStep(
        id="s1",
        description="test step",
        action=AgentToolCall(action_name="noop", arguments={}),
        status=StepStatus.PENDING,
    )
    state = StateEstimate(
        timestamp=datetime.now(UTC),
        label="test",
        confidence=1.0,
        symbolic_system_state={},
    )

    result = await orchestrator.run_step(step, state)

    assert result == "STOPPED"
    assert await blackboard.get_safety_flag("emergency_stop") is True


@pytest.mark.asyncio
async def test_nominal_plan_executes_without_safety_stop():
    blackboard = Blackboard()
    controller = DummyController()
    classifier = DummyClassifier()
    registry = DummyCapabilityRegistry()
    arbiter = Arbiter(blackboard, registry, controller)
    safety_agent = SafetyAgent(blackboard)
    executor = StepExecutor(
        blackboard,
        classifier,
        registry,
        controller,
        default_poll_interval=0.1,
        default_idle_interval=0.1,
        default_max_retries=3,
        default_retry_delay=0.01,
    )
    orchestrator = SafetyOrchestrator(blackboard, arbiter, safety_agent, executor)

    # Safe scene: no hazards
    safe_scene = SceneGraph(objects=[], relationships=[])
    await blackboard.set_scene_graph(safe_scene)

    step = PlanStep(
        id="s2",
        description="safe step",
        action=AgentToolCall(action_name="noop", arguments={}),
        status=StepStatus.PENDING,
    )
    state = StateEstimate(
        timestamp=datetime.now(UTC),
        label="test",
        confidence=1.0,
        symbolic_system_state={},
    )

    result = await orchestrator.run_step(step, state)

    assert result == "COMPLETED"
    assert await blackboard.get_safety_flag("emergency_stop") is not True
