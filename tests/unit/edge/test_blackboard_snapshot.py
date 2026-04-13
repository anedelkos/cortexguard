"""Unit tests for Blackboard.capture_snapshot() and Blackboard.restore_from_snapshot()."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime

import pytest

from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.plan import Plan, PlanStatus, PlanStep, PlanType
from cortexguard.edge.persistence.blackboard_snapshot import BlackboardSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anomaly(key: str, severity: AnomalySeverity = AnomalySeverity.MEDIUM) -> AnomalyEvent:
    return AnomalyEvent(
        id=f"evt-{key}",
        key=key,
        timestamp=datetime.now(UTC),
        severity=severity,
        score=0.75,
        contributing_detectors=["HardLimitDetector"],
        metadata={},
    )


def _make_plan(plan_id: str = "plan-test") -> Plan:
    step = PlanStep(
        id="step-1",
        description="Test step",
        action=AgentToolCall(action_name="MOCK_ACTION", arguments={}),
    )
    goal = GoalContext(goal_id="goal-1", user_prompt="test", intent="Test intent", priority=5)
    return Plan(
        plan_id=plan_id,
        context=goal,
        plan_type=PlanType.RECIPE,
        steps=[step],
        status=PlanStatus.RUNNING,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_snapshot_empty_blackboard() -> None:
    """Snapshot of a fresh Blackboard has empty collections and None plans."""
    bb = Blackboard()
    snapshot = await bb.capture_snapshot()

    assert isinstance(snapshot, BlackboardSnapshot)
    assert snapshot.active_anomalies == {}
    assert snapshot.current_plan is None
    assert snapshot.paused_plan is None
    assert snapshot.plan_step_indices == {}
    assert snapshot.active_remediation_policy is None
    assert snapshot.failed_plans == []
    assert snapshot.recovery_status == {}
    assert snapshot.safety_flags == {}
    assert snapshot.schema_version == 1


@pytest.mark.asyncio
async def test_capture_snapshot_with_active_anomalies() -> None:
    """Captured snapshot includes deep copies of active anomalies."""
    bb = Blackboard()
    anomaly = _make_anomaly("OVERHEAT", AnomalySeverity.HIGH)
    await bb.set_anomaly(anomaly)

    snapshot = await bb.capture_snapshot()

    assert "OVERHEAT" in snapshot.active_anomalies
    captured = snapshot.active_anomalies["OVERHEAT"]
    assert captured.severity == AnomalySeverity.HIGH
    assert captured.score == 0.75

    # Mutation of snapshot must not affect blackboard
    snapshot.active_anomalies["OVERHEAT"].metadata["mutated"] = True
    bb_anomaly = await bb.get_active_anomaly("OVERHEAT")
    assert bb_anomaly is not None
    assert "mutated" not in bb_anomaly.metadata


@pytest.mark.asyncio
async def test_restore_from_snapshot_round_trips_plan() -> None:
    """A plan written to a snapshot can be restored faithfully."""
    bb_source = Blackboard()
    plan = _make_plan("round-trip-plan")
    await bb_source.set_current_plan(plan)
    await bb_source.set_step_index_for_plan("round-trip-plan", 2)

    snapshot = await bb_source.capture_snapshot()

    bb_dest = Blackboard()
    await bb_dest.restore_from_snapshot(snapshot)

    restored_plan = await bb_dest.get_current_plan()
    assert restored_plan is not None
    assert restored_plan.plan_id == "round-trip-plan"
    assert restored_plan.status == PlanStatus.RUNNING

    step_idx = await bb_dest.get_step_index_for_plan("round-trip-plan")
    assert step_idx == 2


@pytest.mark.asyncio
async def test_restore_recomputes_max_anomaly_severity() -> None:
    """After restore, _max_anomaly_severity reflects restored anomalies."""
    bb_source = Blackboard()
    await bb_source.set_anomaly(_make_anomaly("LOW_EVENT", AnomalySeverity.LOW))
    await bb_source.set_anomaly(_make_anomaly("HIGH_EVENT", AnomalySeverity.HIGH))

    snapshot = await bb_source.capture_snapshot()

    bb_dest = Blackboard()
    await bb_dest.restore_from_snapshot(snapshot)

    max_sev = await bb_dest.get_highest_anomaly_severity()
    assert max_sev == AnomalySeverity.HIGH


@pytest.mark.asyncio
async def test_capture_snapshot_failed_plans_serialises_as_list() -> None:
    """failed_plans deque is serialised as a plain list in the snapshot."""
    bb = Blackboard()
    plan = _make_plan("failed-plan")
    await bb.set_failed_plan(plan)

    snapshot = await bb.capture_snapshot()

    assert isinstance(snapshot.failed_plans, list)
    assert len(snapshot.failed_plans) == 1
    assert snapshot.failed_plans[0].plan_id == "failed-plan"


@pytest.mark.asyncio
async def test_restore_failed_plans_becomes_deque() -> None:
    """failed_plans list in snapshot is re-wrapped as deque(maxlen=100) on restore."""
    bb_source = Blackboard()
    for i in range(3):
        await bb_source.set_failed_plan(_make_plan(f"failed-{i}"))

    snapshot = await bb_source.capture_snapshot()

    bb_dest = Blackboard()
    await bb_dest.restore_from_snapshot(snapshot)

    assert isinstance(bb_dest.failed_plans, deque)
    assert bb_dest.failed_plans.maxlen == 100
    assert len(bb_dest.failed_plans) == 3
