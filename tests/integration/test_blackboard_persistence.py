"""Integration tests: full EdgeRuntime restart restores Blackboard state from disk."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.plan import Plan, PlanStep, PlanType
from cortexguard.edge.runtime import EdgeRuntime, RuntimeConfig


def _make_anomaly(key: str) -> AnomalyEvent:
    return AnomalyEvent(
        id=f"evt-{key}",
        key=key,
        timestamp=datetime.now(UTC),
        severity=AnomalySeverity.HIGH,
        score=0.9,
        contributing_detectors=["IntegrationTest"],
        metadata={},
    )


def _make_plan(plan_id: str) -> Plan:
    step = PlanStep(
        id="step-1",
        description="Integration test step",
        action=AgentToolCall(action_name="MOCK_ACTION", arguments={}),
    )
    goal = GoalContext(
        goal_id="goal-int", user_prompt="integration", intent="Integrate", priority=5
    )
    return Plan(
        plan_id=plan_id,
        context=goal,
        plan_type=PlanType.RECIPE,
        steps=[step],
    )


def _make_config(persistence_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        orchestrator_tick_interval=0.05,
        anomaly_check_interval=60.0,  # disable detector loop during test
        persistence_enabled=True,
        persistence_file_path=str(persistence_path),
        persistence_snapshot_interval=9999.0,  # disable auto-loop; rely on shutdown flush
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_restores_anomaly_after_restart(tmp_path: Path) -> None:
    """An anomaly injected before clean shutdown is present after restart."""
    snap_path = tmp_path / "blackboard.json"
    config = _make_config(snap_path)

    # --- First runtime: inject anomaly then shut down cleanly ---
    runtime1 = EdgeRuntime(config)
    await runtime1.start()
    anomaly = _make_anomaly("RESTART_ANOMALY")
    await runtime1.blackboard.set_anomaly(anomaly)
    await runtime1.stop()  # triggers final flush

    assert snap_path.exists(), "Snapshot file must be written on clean shutdown"

    # --- Second runtime: should restore the anomaly ---
    runtime2 = EdgeRuntime(config)
    await runtime2.start()
    try:
        anomalies = await runtime2.blackboard.get_active_anomalies()
        assert (
            "RESTART_ANOMALY" in anomalies
        ), f"Expected RESTART_ANOMALY in restored anomalies, got: {list(anomalies.keys())}"
        assert anomalies["RESTART_ANOMALY"].severity == AnomalySeverity.HIGH
    finally:
        await runtime2.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_restores_current_plan_after_restart(tmp_path: Path) -> None:
    """A running plan is restored from the snapshot on restart."""
    snap_path = tmp_path / "blackboard.json"
    config = _make_config(snap_path)

    # --- First runtime: set a current plan then shut down ---
    runtime1 = EdgeRuntime(config)
    await runtime1.start()
    plan = _make_plan("integration-plan-42")
    await runtime1.blackboard.set_current_plan(plan)
    await runtime1.blackboard.set_step_index_for_plan("integration-plan-42", 1)
    await runtime1.stop()

    # --- Second runtime: plan should be restored ---
    runtime2 = EdgeRuntime(config)
    await runtime2.start()
    try:
        restored_plan = await runtime2.blackboard.get_current_plan()
        assert restored_plan is not None, "current_plan must be restored from snapshot"
        assert restored_plan.plan_id == "integration-plan-42"

        step_idx = await runtime2.blackboard.get_step_index_for_plan("integration-plan-42")
        assert step_idx == 1
    finally:
        await runtime2.stop()
