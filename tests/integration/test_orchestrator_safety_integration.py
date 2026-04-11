from datetime import UTC, datetime
from typing import Any

import pytest

from cortexguard.edge.arbiter import Arbiter
from cortexguard.edge.edge_fusion import _build_scene_graph_from_vision
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.capability_registry import (
    CapabilityRegistry,
    FunctionSchema,
    RiskLevel,
)
from cortexguard.edge.models.plan import PlanStep, StepStatus
from cortexguard.edge.models.scene_graph import SceneGraph, SceneObject, SceneRelationship
from cortexguard.edge.models.state_estimate import StateEstimate
from cortexguard.edge.orchestrator import Orchestrator
from cortexguard.edge.safety_agent import SafetyAgent
from cortexguard.edge.step_executor import StepExecutor


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

    # Hazardous scene: hand near blade.
    # Labels are lowercased — matching what EdgeFusion._to_scene_object produces.
    hand = SceneObject(
        id="hand1",
        label="human_hand",
        properties={},
        location_2d=[0.0, 0.0, 0.1, 0.1],
        pose_3d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    blade = SceneObject(
        id="blade1",
        label="blade",
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
async def test_safety_agent_triggers_estop_via_fusion_scene_graph():
    """SceneGraph built via _build_scene_graph_from_vision applies label lowercasing before safety rule evaluation."""
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

    # Raw vision objects as a sensor payload would produce (mixed-case labels).
    # _build_scene_graph_from_vision lowercases them, exactly as EdgeFusion does.
    # distance_m <= 0.5 on both objects ensures _is_near() infers a "near" relationship.
    raw_vision_objects = [
        {"bbox_id": "hand1", "label": "Human_Hand", "distance_m": 0.2, "confidence": 0.9},
        {"bbox_id": "blade1", "label": "Blade", "distance_m": 0.2, "confidence": 0.9},
    ]
    scene = _build_scene_graph_from_vision(
        timestamp=datetime.now(UTC),
        vision_objects=raw_vision_objects,
        ema_state={},
    )

    # Verify the fusion path has lowercased the labels (this is what makes the rule testable).
    labels = {obj.label for obj in scene.objects}
    assert labels == {"human_hand", "blade"}, f"EdgeFusion must lowercase labels; got {labels}"

    await blackboard.set_scene_graph(scene)

    step = PlanStep(
        id="s3",
        description="test step via fusion path",
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

    assert result == "STOPPED", (
        "Safety rule did not fire for lowercased labels produced by the real fusion path. "
        "Check _rule_human_hand_near_hazard label comparisons."
    )
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
