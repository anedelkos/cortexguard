import asyncio
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

import pytest

from kitchenwatch.core.interfaces.base_controller import BaseController
from kitchenwatch.edge.arbiter import Arbiter
from kitchenwatch.edge.models.agent_tool_call import AgentToolCall
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.capability_registry import CapabilityRegistry
from kitchenwatch.edge.models.goal import GoalContext
from kitchenwatch.edge.models.plan import Plan, PlanStatus, PlanStep, PlanType, StepStatus
from kitchenwatch.edge.models.state_estimate import StateEstimate
from kitchenwatch.edge.orchestrator import Orchestrator
from kitchenwatch.edge.safety_agent import SafetyAgent


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def blackboard() -> Blackboard:
    return Blackboard()


class DummyController(BaseController):
    async def execute(self, name: str, args: dict[str, Any]) -> None:
        return

    async def emergency_stop(self) -> None:
        return


@pytest.fixture
def orchestrator(blackboard: Blackboard) -> Orchestrator:
    controller = DummyController()
    capability_registry = CapabilityRegistry()
    arbiter = Arbiter(blackboard, capability_registry, controller)
    safety_agent = SafetyAgent(blackboard)
    return Orchestrator(blackboard, arbiter, safety_agent)


def make_plan(
    plan_id: str,
    priority: int = 5,
    step_status: StepStatus = StepStatus.PENDING,
) -> Plan:
    steps = [
        PlanStep(
            id="step_1",
            description="Execute the first test movement and sensor reading.",
            action=AgentToolCall(  # Replaced Action with AgentToolCall
                action_name="MOCK_CAPABILITY_A",  # capability changed to function_name
                arguments={
                    "tool_id": "mock_tool_1",
                    "duration": 1.5,
                    "speed": "high",
                },  # tool_id moved to arguments
            ),
            status=step_status,
        ),
        PlanStep(
            id="step_2",
            description="Execute the second test measurement with a different tool.",
            action=AgentToolCall(  # Replaced Action with AgentToolCall
                action_name="MOCK_CAPABILITY_B",  # capability changed to function_name
                arguments={
                    "tool_id": "mock_tool_2",
                    "sensor_id": "temp_01",
                },  # tool_id moved to arguments
            ),
            status=step_status,
        ),
    ]

    goal_context = GoalContext(
        goal_id=f"goal_{plan_id}",
        user_prompt="Run a standard two-step test assembly process.",
        priority=priority,
        intent="Do stuff",
    )

    return Plan(
        plan_id=plan_id,
        context=goal_context,
        plan_type=PlanType.RECIPE,
        steps=steps,
        status=PlanStatus.PENDING,
        created_at=datetime.now(),
    )


@pytest.mark.asyncio
async def test_add_plan_sets_status_and_queues(orchestrator: Orchestrator) -> None:
    plan = make_plan("plan_001")
    await orchestrator.add_plan(plan)
    # Corrected assertion to use Enum value
    assert plan.status == PlanStatus.PENDING


@pytest.mark.asyncio
async def test_start_next_plan_sets_blackboard_state(
    orchestrator: Orchestrator, blackboard: Blackboard
) -> None:
    plan = make_plan("plan_002")
    await orchestrator.add_plan(plan)
    await orchestrator._start_next_plan()

    current_plan = await blackboard.get_current_plan()
    assert current_plan is not None
    assert current_plan.plan_id == "plan_002"

    current_step = await blackboard.get_current_step()

    assert current_step is not None
    # Corrected step field name to match new model
    assert current_step.id == "step_1"


@pytest.mark.asyncio
async def test_advance_plan_completes_successful_plan(
    orchestrator: Orchestrator, blackboard: Blackboard
) -> None:
    plan = make_plan("plan_003")
    await orchestrator.add_plan(plan)
    await orchestrator._start_next_plan()

    # Step 1: Complete
    step = await blackboard.get_current_step()
    assert step is not None
    assert step.id == "step_1"
    step.status = StepStatus.COMPLETED

    # Advance to step 2
    await orchestrator._advance_plan_or_handle_failure()

    # Step 2: Complete
    step = await blackboard.get_current_step()
    assert step is not None
    assert step.id == "step_2"
    step.status = StepStatus.COMPLETED

    # Advance, which should complete the plan
    await orchestrator._advance_plan_or_handle_failure()

    current_plan = await blackboard.get_current_plan()
    assert current_plan is None
    assert plan.status == PlanStatus.COMPLETED


@pytest.mark.asyncio
async def test_advance_plan_pauses_after_failed_step(
    orchestrator: Orchestrator, blackboard: Blackboard
) -> None:
    plan = make_plan("plan_004")
    await orchestrator.add_plan(plan)
    await orchestrator._start_next_plan()

    step = await blackboard.get_current_step()
    assert step is not None
    step.status = StepStatus.FAILED

    await orchestrator._advance_plan_or_handle_failure()
    current_plan = await blackboard.get_current_plan()
    assert current_plan is None
    paused_plan = await blackboard.get_paused_plan()
    assert paused_plan is not None
    assert paused_plan.status == PlanStatus.PAUSED
    assert paused_plan.plan_id == "plan_004"


@pytest.mark.asyncio
async def test_run_loop_starts_and_completes_plan(
    orchestrator: Orchestrator, blackboard: Blackboard
) -> None:
    # Seed a nominal state estimate so the loop doesn’t skip
    await blackboard.update_state_estimate(
        StateEstimate(
            timestamp=datetime.now(UTC),
            label="test",
            confidence=1.0,
            symbolic_system_state={},
        )
    )

    plan = make_plan("plan_006")
    await orchestrator.add_plan(plan)

    run_task = asyncio.create_task(orchestrator.start(tick_interval=0.01))
    await asyncio.sleep(0.03)

    # Step 1: Complete
    step = await blackboard.get_current_step()
    assert step is not None
    step.status = StepStatus.COMPLETED

    await asyncio.sleep(0.03)  # Wait for advance

    # Step 2: Complete
    step = await blackboard.get_current_step()
    assert step is not None
    step.status = StepStatus.COMPLETED

    await asyncio.sleep(0.03)  # Wait for completion
    await orchestrator.stop()
    await run_task

    current_plan = await blackboard.get_current_plan()
    assert current_plan is None
    assert plan.status == PlanStatus.COMPLETED


@pytest.mark.asyncio
async def test_start_next_plan_with_empty_queue(orchestrator: Orchestrator) -> None:
    # Should gracefully handle no plans
    await orchestrator._start_next_plan()
    assert orchestrator._current_plan is None


@pytest.mark.asyncio
async def test_add_plan_handles_exception(
    orchestrator: Orchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = make_plan("plan_008")

    # Corrected signature: put takes priority and item
    async def broken_put(priority: int, item: Plan) -> None:
        raise RuntimeError("Queue error")

    monkeypatch.setattr(orchestrator._plan_queue, "put", broken_put)
    # Should catch and log the exception without crashing
    await orchestrator.add_plan(plan)
    # Still marked pending because add_plan sets it before putting
    assert plan.status == PlanStatus.PENDING


@pytest.mark.asyncio
async def test_start_next_plan_handles_exception(
    orchestrator: Orchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = make_plan("plan_009")
    await orchestrator.add_plan(plan)

    # Corrected signature: pop takes block
    async def broken_pop(block: bool = False) -> Plan | None:
        raise RuntimeError("Queue pop failed")

    monkeypatch.setattr(orchestrator._plan_queue, "pop", broken_pop)

    await orchestrator._start_next_plan()
    # Should recover and reset current plan
    assert orchestrator._current_plan is None


@pytest.mark.asyncio
async def test_preemption_resumes_previous_plan(
    orchestrator: "Orchestrator", blackboard: "Blackboard"
) -> None:
    # Normal plan with lower priority
    low_plan = make_plan("low_priority_plan", priority=5)
    await orchestrator.add_plan(low_plan)
    # Start the plan. Orchestrator should set index to 0 and current step to step_1.
    await orchestrator._start_next_plan()

    # Confirm it's running
    current_plan = await blackboard.get_current_plan()
    assert current_plan is not None
    assert current_plan.plan_id == "low_priority_plan"

    # Assert initial index is 0 on the Blackboard
    assert await blackboard.get_step_index_for_plan("low_priority_plan") == 0

    # Complete step 1 (index 0)
    step = await blackboard.get_current_step()
    assert step is not None
    step.status = StepStatus.COMPLETED
    # Orchestrator._advance_plan_or_handle_failure() will:
    # 1. Read index 0 from Blackboard.
    # 2. Increment index to 1.
    # 3. Write index 1 to Blackboard.
    # 4. Set current step to low_plan.steps[1] (step_2).
    await orchestrator._advance_plan_or_handle_failure()

    # Check state after advancement
    current_plan = await blackboard.get_current_plan()
    assert current_plan is low_plan
    # Assert index is now 1 on the Blackboard
    assert await blackboard.get_step_index_for_plan("low_priority_plan") == 1

    # The current step should be step_2
    current_step = await blackboard.get_current_step()
    assert current_step is not None
    assert current_step.id == "step_2"

    # Urgent plan comes in with higher priority
    urgent_plan = make_plan("urgent_plan", priority=1)
    await orchestrator.add_plan(urgent_plan)

    # Simulate tick that checks for preemption
    top_priority_plan = await orchestrator._check_for_preemption()
    assert urgent_plan == top_priority_plan
    # This call should:
    # 1. Set low_plan status to PREEMPTED.
    # 2. Save low_plan state (index 1 is already saved).
    # 3. Set current_plan to urgent_plan.
    # 4. Set urgent_plan index to 0 on Blackboard.
    await orchestrator._start_next_plan(top_priority_plan)

    # Urgent plan should be running now
    current_plan = await blackboard.get_current_plan()
    assert current_plan is not None
    assert current_plan.plan_id == "urgent_plan"
    assert current_plan.status == PlanStatus.RUNNING

    # Check that low_plan's state is preserved on Blackboard
    assert await blackboard.get_step_index_for_plan("low_priority_plan") == 1
    # Check that urgent_plan started at index 0
    assert await blackboard.get_step_index_for_plan("urgent_plan") == 0

    # Complete step 1 (index 0) of urgent plan
    step = await blackboard.get_current_step()
    assert step is not None
    step.status = StepStatus.COMPLETED
    # Advances urgent plan index from 0 to 1
    await orchestrator._advance_plan_or_handle_failure()
    assert await blackboard.get_step_index_for_plan("urgent_plan") == 1

    # Complete step 2 (index 1) of urgent plan
    step = await blackboard.get_current_step()
    assert step is not None
    step.status = StepStatus.COMPLETED
    # Advances urgent plan. Since index 2 >= len(steps) (which is 2), it completes the plan.
    # It also calls clear_step_index_for_plan("urgent_plan")
    await orchestrator._advance_plan_or_handle_failure()

    # Assert urgent plan state is cleared
    assert await blackboard.get_step_index_for_plan("urgent_plan") is None

    # Start the next plan in the queue (which is the preempted low_plan)
    # Orchestrator should:
    # 1. Find low_plan in the queue.
    # 2. Call get_step_index_for_plan("low_priority_plan") and get 1.
    # 3. Set current_plan to low_plan.
    # 4. Set current_step to low_plan.steps[1] (step_2).
    await orchestrator._start_next_plan()

    # After urgent plan completes, the low priority plan should resume
    current_plan = await blackboard.get_current_plan()
    assert current_plan is not None
    assert current_plan.plan_id == "low_priority_plan"
    assert current_plan.status == PlanStatus.RUNNING

    # Ensure it resumes from the correct step (step 2 - index 1)
    current_step = await blackboard.get_current_step()
    assert current_step is not None
    assert current_step.id == "step_2"
    # Ensure index state is still 1
    assert await blackboard.get_step_index_for_plan("low_priority_plan") == 1


@pytest.mark.asyncio
async def test_start_next_plan_with_invalid_status(orchestrator: Orchestrator) -> None:
    plan = make_plan("invalid_plan")
    plan.status = PlanStatus.COMPLETED  # Not PENDING or PREEMPTED
    await orchestrator.add_plan(plan)
    await orchestrator._start_next_plan()
    assert orchestrator._current_plan is None


@pytest.mark.asyncio
async def test_run_loop_handles_cancel(orchestrator: Orchestrator) -> None:
    await orchestrator.add_plan(make_plan("cancel_plan"))
    await orchestrator.start(tick_interval=0.01)
    await asyncio.sleep(0.02)

    # Cancel the actual loop task
    assert orchestrator._task is not None
    orchestrator._task.cancel()
    try:
        await orchestrator._task
    except asyncio.CancelledError:
        pass

    assert not orchestrator._loop_running


@pytest.mark.asyncio
async def test_start_twice_does_not_restart(orchestrator: Orchestrator) -> None:
    # We rely on the internal logging/warning for this, but the function call sequence
    # should be safe.
    await orchestrator.start()
    await orchestrator.start()  # Should log warning and return
    await orchestrator.stop()


@pytest.mark.asyncio
async def test_stop_flush_failure(
    orchestrator: Orchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = make_plan("flush_fail_plan")
    await orchestrator.add_plan(plan)
    await orchestrator._start_next_plan()

    # Ensure the monkeypatch target matches the method signature
    async def broken_set(key: str, value: object) -> None:
        raise RuntimeError("flush failed")

    monkeypatch.setattr(orchestrator._blackboard, "set", broken_set)
    await orchestrator.stop()  # Should log warning but not crash
