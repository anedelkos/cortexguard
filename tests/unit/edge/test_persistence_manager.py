"""Unit tests for PersistenceManager."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.plan import Plan, PlanStatus, PlanStep, PlanType
from cortexguard.edge.persistence.blackboard_snapshot import (
    CURRENT_SCHEMA_VERSION,
    BlackboardSnapshot,
)
from cortexguard.edge.persistence.persistence_manager import PersistenceManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anomaly(key: str = "TEST_ANOMALY") -> AnomalyEvent:
    return AnomalyEvent(
        id="evt-test",
        key=key,
        timestamp=datetime.now(UTC),
        severity=AnomalySeverity.MEDIUM,
        score=0.8,
        contributing_detectors=["HardLimitDetector"],
        metadata={},
    )


def _make_plan(plan_id: str = "test-plan") -> Plan:
    step = PlanStep(
        id="step-1",
        description="Persistence test step",
        action=AgentToolCall(action_name="MOCK_ACTION", arguments={}),
    )
    goal = GoalContext(goal_id="goal-1", user_prompt="persist me", intent="Persist", priority=3)
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
async def test_start_creates_parent_directory(tmp_path: Path) -> None:
    """PersistenceManager.start() creates missing parent directories."""
    deep_path = tmp_path / "a" / "b" / "c" / "blackboard.json"
    bb = Blackboard()
    pm = PersistenceManager(bb, deep_path, snapshot_interval=60.0)
    await pm.start()
    await pm.stop()  # clean up loop task

    assert deep_path.parent.exists()


@pytest.mark.asyncio
async def test_persist_and_restore_roundtrip(tmp_path: Path) -> None:
    """A snapshot persisted to disk can be restored into a fresh Blackboard."""
    snap_path = tmp_path / "blackboard.json"

    # Build source state
    bb_source = Blackboard()
    await bb_source.set_anomaly(_make_anomaly("ROUNDTRIP"))
    await bb_source.set_current_plan(_make_plan("rtrip-plan"))
    await bb_source.set_safety_flag("E_STOP", True)

    pm_source = PersistenceManager(bb_source, snap_path, snapshot_interval=60.0)
    await pm_source.start()
    await pm_source.stop()  # triggers final flush

    # Restore into a fresh Blackboard
    bb_dest = Blackboard()
    pm_dest = PersistenceManager(bb_dest, snap_path, snapshot_interval=60.0)
    restored = await pm_dest.restore()

    assert restored is True
    anomalies = await bb_dest.get_active_anomalies()
    assert "ROUNDTRIP" in anomalies

    current_plan = await bb_dest.get_current_plan()
    assert current_plan is not None
    assert current_plan.plan_id == "rtrip-plan"

    flag = await bb_dest.get_safety_flag("E_STOP")
    assert flag is True


@pytest.mark.asyncio
async def test_restore_returns_false_on_missing_file(tmp_path: Path) -> None:
    """restore() returns False when no snapshot file exists."""
    bb = Blackboard()
    pm = PersistenceManager(bb, tmp_path / "nonexistent.json", snapshot_interval=60.0)
    result = await pm.restore()
    assert result is False


@pytest.mark.asyncio
async def test_restore_returns_false_on_schema_version_mismatch(tmp_path: Path) -> None:
    """restore() returns False when snapshot schema version doesn't match current."""
    snap_path = tmp_path / "blackboard.json"

    # Write a snapshot with a future schema version
    snapshot = BlackboardSnapshot(
        schema_version=999,
        captured_at=datetime.now(UTC),
        active_anomalies={},
        current_plan=None,
        paused_plan=None,
        plan_step_indices={},
        active_remediation_policy=None,
        failed_plans=[],
        recovery_status={},
        safety_flags={},
    )
    snap_path.write_text(snapshot.model_dump_json(), encoding="utf-8")

    bb = Blackboard()
    pm = PersistenceManager(bb, snap_path, snapshot_interval=60.0)
    result = await pm.restore()
    assert result is False


@pytest.mark.asyncio
async def test_restore_returns_true_on_valid_snapshot(tmp_path: Path) -> None:
    """restore() returns True for a valid, version-matched snapshot."""
    snap_path = tmp_path / "blackboard.json"
    snapshot = BlackboardSnapshot(
        schema_version=CURRENT_SCHEMA_VERSION,
        captured_at=datetime.now(UTC),
        active_anomalies={},
        current_plan=None,
        paused_plan=None,
        plan_step_indices={},
        active_remediation_policy=None,
        failed_plans=[],
        recovery_status={},
        safety_flags={},
    )
    snap_path.write_text(snapshot.model_dump_json(), encoding="utf-8")

    bb = Blackboard()
    pm = PersistenceManager(bb, snap_path, snapshot_interval=60.0)
    result = await pm.restore()
    assert result is True


@pytest.mark.asyncio
async def test_stop_flushes_final_snapshot(tmp_path: Path) -> None:
    """stop() writes the snapshot file even when no loop tick has occurred."""
    snap_path = tmp_path / "blackboard.json"
    bb = Blackboard()
    await bb.set_safety_flag("PAUSED", True)

    pm = PersistenceManager(bb, snap_path, snapshot_interval=9999.0)
    await pm.start()
    # File should not exist yet (interval is huge)
    assert not snap_path.exists()

    await pm.stop()
    assert snap_path.exists()

    data = json.loads(snap_path.read_text())
    assert data["safety_flags"]["PAUSED"] is True


@pytest.mark.asyncio
async def test_snapshot_loop_writes_file(tmp_path: Path) -> None:
    """The background loop writes the snapshot file after the interval elapses."""
    snap_path = tmp_path / "blackboard.json"
    bb = Blackboard()
    await bb.set_recovery_status("probe", "ok")

    pm = PersistenceManager(bb, snap_path, snapshot_interval=0.05)
    await pm.start()
    # Wait longer than one interval
    await asyncio.sleep(0.15)
    await pm.stop()

    assert snap_path.exists()
    data = json.loads(snap_path.read_text())
    assert data["recovery_status"]["probe"] == "ok"


@pytest.mark.asyncio
async def test_atomic_write_uses_tmp_file_then_replace(tmp_path: Path) -> None:
    """_persist_snapshot() writes to a .tmp file then renames it atomically."""
    snap_path = tmp_path / "blackboard.json"
    tmp_snap_path = Path(str(snap_path) + ".tmp")

    bb = Blackboard()
    pm = PersistenceManager(bb, snap_path, snapshot_interval=9999.0)
    await pm.start()
    await pm._persist_snapshot()
    await pm.stop()

    # After replace, .tmp must not remain
    assert not tmp_snap_path.exists()
    assert snap_path.exists()
