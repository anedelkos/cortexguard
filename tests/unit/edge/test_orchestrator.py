import asyncio
from collections.abc import Generator
from datetime import datetime

import pytest

from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.plan import Plan, PlanStatus, PlanStep, PlanType, StepStatus
from kitchenwatch.edge.orchestrator import Orchestrator


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def blackboard() -> Blackboard:
    return Blackboard()


@pytest.fixture
def orchestrator(blackboard: Blackboard) -> Orchestrator:
    return Orchestrator(blackboard)


def make_plan(
    plan_id: str,
    priority: int = 5,
    step_status: StepStatus = StepStatus.PENDING,
) -> Plan:
    steps = [
        PlanStep(id="step_1", name="test_step_1", parameters={}, status=step_status),
        PlanStep(id="step_2", name="test_step_2", parameters={}, status=step_status),
    ]
    return Plan(
        plan_id=plan_id,
        plan_type=PlanType.RECIPE,
        version="1.0",
        created_at=datetime.now(),
        goal="test_goal",
        priority=priority,
        steps=steps,
        status=PlanStatus.PENDING,
        current_step_index=0,
    )


@pytest.mark.asyncio
async def test_add_plan_sets_status_and_queues(orchestrator: Orchestrator) -> None:
    plan = make_plan("plan_001")
    await orchestrator.add_plan(plan)
    assert plan.status == "pending"


@pytest.mark.asyncio
async def test_start_next_plan_sets_blackboard_state(
    orchestrator: Orchestrator, blackboard: Blackboard
) -> None:
    plan = make_plan("plan_002")
    await orchestrator.add_plan(plan)
    await orchestrator._start_next_plan()

    current_plan = await blackboard.get_current_plan()
    current_step = await blackboard.get_current_step()

    assert current_plan is not None
    assert current_plan.plan_id == "plan_002"
    assert current_step is not None
    assert current_step.id == "step_1"


@pytest.mark.asyncio
async def test_advance_plan_completes_successful_plan(
    orchestrator: Orchestrator, blackboard: Blackboard
) -> None:
    plan = make_plan("plan_003")
    await orchestrator.add_plan(plan)
    await orchestrator._start_next_plan()

    step = await blackboard.get_current_step()
    assert step is not None
    step.status = StepStatus.COMPLETED

    # Plan has 2 steps
    await orchestrator._advance_plan_or_handle_failure()
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
    plan = make_plan("plan_006")
    await orchestrator.add_plan(plan)

    run_task = asyncio.create_task(orchestrator.start(tick_interval=0.01))
    await asyncio.sleep(0.03)

    step = await blackboard.get_current_step()
    assert step is not None
    step.status = StepStatus.COMPLETED

    await asyncio.sleep(0.03)
    step = await blackboard.get_current_step()
    assert step is not None
    step.status = StepStatus.COMPLETED

    await asyncio.sleep(0.03)
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

    async def broken_put() -> None:
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

    async def broken_pop() -> None:
        raise RuntimeError("Queue pop failed")

    monkeypatch.setattr(orchestrator._plan_queue, "pop", broken_pop)

    await orchestrator._start_next_plan()
    # Should recover and reset current plan
    assert orchestrator._current_plan is None


@pytest.mark.asyncio
async def test_preemption_resumes_previous_plan(
    orchestrator: Orchestrator, blackboard: Blackboard
) -> None:
    # Normal plan with lower priority
    low_plan = make_plan("low_priority_plan", priority=5)
    await orchestrator.add_plan(low_plan)
    await orchestrator._start_next_plan()

    # Confirm it's running
    current_plan = await blackboard.get_current_plan()
    assert current_plan is not None
    assert current_plan.plan_id == "low_priority_plan"

    await orchestrator._advance_plan_or_handle_failure()
    current_plan = await blackboard.get_current_plan()
    assert current_plan is low_plan
    assert current_plan.current_step_index == 1

    # Urgent plan comes in with higher priority
    urgent_plan = make_plan("urgent_plan", priority=1)
    await orchestrator.add_plan(urgent_plan)

    # Simulate tick that checks for preemption
    top_priority_plan = await orchestrator._check_for_preemption()
    assert urgent_plan == top_priority_plan
    await orchestrator._start_next_plan(top_priority_plan)

    # Urgent plan should be running now
    current_plan = await blackboard.get_current_plan()
    assert current_plan is not None
    assert current_plan.plan_id == "urgent_plan"
    assert current_plan.status == PlanStatus.RUNNING

    # Previous plan should be preempted and still in queue
    assert low_plan.status == PlanStatus.PREEMPTED
    queued_plans = await orchestrator._plan_queue.get_all_items()
    assert any(plan.plan_id == "low_priority_plan" for plan in queued_plans)

    # Complete urgent plan
    step = await blackboard.get_current_step()
    assert step is not None
    step.status = StepStatus.COMPLETED
    # Run twice to complete both plan steps
    await orchestrator._advance_plan_or_handle_failure()
    await orchestrator._advance_plan_or_handle_failure()
    await orchestrator._start_next_plan()

    # After urgent plan completes, the low priority plan should resume
    current_plan = await blackboard.get_current_plan()
    assert current_plan is not None
    assert current_plan.plan_id == "low_priority_plan"
    assert current_plan.status == "running"

    # Ensure it resumes from the correct step (step 1)
    current_step = await blackboard.get_current_step()
    assert current_step is not None
    assert current_step.id == "step_2"


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

    async def broken_set(key: str, value: object) -> None:
        raise RuntimeError("flush failed")

    monkeypatch.setattr(orchestrator._blackboard, "set", broken_set)
    await orchestrator.stop()  # Should log warning but not crash
