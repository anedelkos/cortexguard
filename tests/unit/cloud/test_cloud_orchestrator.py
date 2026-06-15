from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from cortexguard.cloud.orchestrator import CloudOrchestrator
from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.cloud.persistence.repository import InMemoryIncidentRepository
from cortexguard.cloud.planner.mock_client import MockLLMClient
from cortexguard.cloud.validation.capability_adapter import CapabilityAdapter
from cortexguard.cloud.validation.plan_validator import PlanValidator


@pytest.fixture
def orchestrator() -> CloudOrchestrator:
    repo = InMemoryIncidentRepository()
    llm = MockLLMClient()
    validator = PlanValidator(CapabilityAdapter.load_default())
    return CloudOrchestrator(
        repo=repo,
        retrieval_store=None,
        llm_client=llm,
        validator=validator,
        checkpointer=MemorySaver(),
    )


@pytest.mark.asyncio
async def test_resume_get_state_failure_does_not_raise(orchestrator: CloudOrchestrator) -> None:
    """resume() catches get_state exception and returns gracefully."""
    assert orchestrator._graph is not None

    def _raising(_config: object) -> object:
        raise RuntimeError("checkpoint store unavailable")

    original = orchestrator._graph.get_state
    orchestrator._graph.get_state = _raising  # type: ignore[method-assign]
    try:
        await orchestrator.resume(
            thread_id=str(uuid.uuid4()),
            operator_response={"approved": True, "outcome": "resolved"},
        )
    finally:
        orchestrator._graph.get_state = original


@pytest.mark.asyncio
async def test_resume_non_interrupted_thread_returns_early(
    orchestrator: CloudOrchestrator,
) -> None:
    """resume() returns when snap.next is empty."""
    assert orchestrator._graph is not None

    snap = MagicMock()
    snap.next = ()
    orchestrator._graph.get_state = MagicMock(return_value=snap)

    await orchestrator.resume(
        thread_id=str(uuid.uuid4()),
        operator_response={"approved": True},
    )


@pytest.mark.asyncio
async def test_resume_success(orchestrator: CloudOrchestrator) -> None:
    """resume() runs ainvoke and finalises incident on success."""
    assert orchestrator._graph is not None

    snap = MagicMock()
    snap.next = ("pause_for_operator",)
    orchestrator._graph.get_state = MagicMock(return_value=snap)

    final_state = {
        "incident_id": "test-incident",
        "decision": "plan_ready",
        "candidate_plan": None,
        "rationale": "operator approved",
        "confidence": 0.9,
        "validation_result": None,
    }
    orchestrator._graph.ainvoke = AsyncMock(return_value=final_state)

    repo = InMemoryIncidentRepository()
    record = IncidentRecord(
        incident_id="test-incident",
        escalation_id="test-trace",
        trace_id="test-trace",
        device_id="dev-01",
        anomaly_key="OVERHEAT",
        anomaly_type="detected",
        severity="high",
        summary="test",
        raw_packet_json="{}",
        retrieved_incident_ids_json="[]",
        candidate_plan_json=None,
        validation_errors_json="[]",
        decision="pending",
        created_at=datetime.now(UTC),
    )
    await repo.save_incident(record)
    orchestrator._repo = repo

    await orchestrator.resume(
        thread_id="test-thread",
        operator_response={"approved": True, "outcome": "resolved"},
    )


@pytest.mark.asyncio
async def test_execute_detects_interruption(orchestrator: CloudOrchestrator) -> None:
    """_execute returns interrupted=True when graph has pending tasks."""
    assert orchestrator._graph is not None

    snap = MagicMock()
    snap.next = ("pause_for_operator",)
    orchestrator._graph.get_state = MagicMock(return_value=snap)

    final_state = {
        "decision": "needs_human",
        "candidate_plan": None,
        "confidence": 0.0,
        "rationale": "",
    }
    orchestrator._graph.ainvoke = AsyncMock(return_value=final_state)

    packet = MagicMock()
    packet.trace_id = "test-trace"
    state, interrupted = await orchestrator._execute(packet.trace_id, packet)

    assert interrupted is True


@pytest.mark.asyncio
async def test_resolve_stale_checkpoints_no_checkpointer() -> None:
    """resolve_stale_checkpoints returns 0 when graph has no checkpointer."""
    repo = InMemoryIncidentRepository()
    llm = MockLLMClient()
    validator = PlanValidator(CapabilityAdapter.load_default())
    orch = CloudOrchestrator(
        repo=repo,
        retrieval_store=None,
        llm_client=llm,
        validator=validator,
    )
    resolved = await orch.resolve_stale_checkpoints()
    assert resolved == 0


@pytest.mark.asyncio
async def test_resolve_stale_checkpoints_resolves_stale_threads(
    orchestrator: CloudOrchestrator,
) -> None:
    """resolve_stale_checkpoints finds stale interrupted threads and resolves them."""
    assert orchestrator._graph is not None

    mock_checkpointer = MagicMock()
    orchestrator._graph.checkpointer = mock_checkpointer

    old_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()

    class _FakeCp:
        def __init__(self, tid: str, created_at: str):
            self.config = {"configurable": {"thread_id": tid}}
            self.created_at = created_at

    async def fake_alist(_filter: object) -> AsyncMock:
        items = [_FakeCp("thread-stale-1", old_ts), _FakeCp("thread-stale-2", old_ts)]
        for item in items:
            yield item

    mock_checkpointer.alist = fake_alist

    snap = MagicMock()
    snap.next = ("pause_for_operator",)
    snap.created_at = old_ts
    orchestrator._graph.get_state = MagicMock(return_value=snap)

    orchestrator.resume = AsyncMock()  # type: ignore[method-assign]

    resolved = await orchestrator.resolve_stale_checkpoints(max_hours=1)
    assert resolved == 2
    assert orchestrator.resume.await_count == 2


@pytest.mark.asyncio
async def test_finalise_incident_skips_non_dict(orchestrator: CloudOrchestrator) -> None:
    """_finalise_incident returns early when final_state is not a dict."""
    repo = AsyncMock()
    orchestrator._repo = repo

    await orchestrator._finalise_incident("not-a-dict", "no_safe_plan", None)
    repo.get_incident.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalise_incident_skips_no_incident_id(orchestrator: CloudOrchestrator) -> None:
    """_finalise_incident returns early when incident_id is missing."""
    repo = AsyncMock()
    orchestrator._repo = repo

    await orchestrator._finalise_incident({"foo": "bar"}, "no_safe_plan", None)
    repo.get_incident.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_exception_during_workflow(orchestrator: CloudOrchestrator) -> None:
    """_execute handles exceptions from run_planning_workflow gracefully."""
    assert orchestrator._graph is not None
    orchestrator._graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM failed"))

    packet = MagicMock()
    packet.trace_id = "test-trace"
    state, interrupted = await orchestrator._execute(packet.trace_id, packet)

    assert state is None
    assert interrupted is False


@pytest.mark.asyncio
async def test_execute_interrupted_check_fails(orchestrator: CloudOrchestrator) -> None:
    """_execute handles get_state exception during interrupted check."""
    assert orchestrator._graph is not None

    final_state = {
        "decision": "needs_human",
        "candidate_plan": None,
        "confidence": 0.0,
        "rationale": "",
    }
    orchestrator._graph.ainvoke = AsyncMock(return_value=final_state)

    def _raising(_config: object) -> object:
        raise RuntimeError("checkpoint unavailable")

    original = orchestrator._graph.get_state
    orchestrator._graph.get_state = _raising  # type: ignore[method-assign]
    try:
        packet = MagicMock()
        packet.trace_id = "test-trace"
        state, interrupted = await orchestrator._execute(packet.trace_id, packet)
        assert interrupted is False
    finally:
        orchestrator._graph.get_state = original


@pytest.mark.asyncio
async def test_resolve_stale_checkpoints_alist_not_implemented(
    orchestrator: CloudOrchestrator,
) -> None:
    """resolve_stale_checkpoints falls back to raw SQL when alist is not implemented."""
    assert orchestrator._graph is not None

    mock_checkpointer = MagicMock()
    mock_checkpointer.alist.side_effect = NotImplementedError()
    mock_checkpointer.conn = AsyncMock()
    mock_checkpointer.conn.execute = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[("thread-1",), ("thread-2",)])
    cursor.close = AsyncMock()
    mock_checkpointer.conn.execute.return_value = cursor

    orchestrator._graph.checkpointer = mock_checkpointer

    snap = MagicMock()
    snap.next = ()
    orchestrator._graph.get_state = MagicMock(return_value=snap)

    resolved = await orchestrator.resolve_stale_checkpoints(max_hours=1)
    assert resolved == 0  # all skipped because snap.next is empty


@pytest.mark.asyncio
async def test_resolve_stale_checkpoints_alist_not_implemented_no_conn(
    orchestrator: CloudOrchestrator,
) -> None:
    """resolve_stale_checkpoints returns 0 when alist is not implemented and no conn."""
    assert orchestrator._graph is not None

    mock_checkpointer = MagicMock()
    mock_checkpointer.alist.side_effect = NotImplementedError()
    mock_checkpointer.conn = None
    orchestrator._graph.checkpointer = mock_checkpointer

    resolved = await orchestrator.resolve_stale_checkpoints()
    assert resolved == 0


@pytest.mark.asyncio
async def test_run_once_returns_interrupted(orchestrator: CloudOrchestrator) -> None:
    """run_once returns the interrupted flag from _execute."""
    assert orchestrator._graph is not None

    snap = MagicMock()
    snap.next = ("pause_for_operator",)
    orchestrator._graph.get_state = MagicMock(return_value=snap)

    final_state = {
        "decision": "needs_human",
        "candidate_plan": None,
        "confidence": 0.0,
        "rationale": "",
    }
    orchestrator._graph.ainvoke = AsyncMock(return_value=final_state)

    packet = MagicMock()
    packet.trace_id = "test-trace"
    interrupted = await orchestrator.run_once(packet)
    assert interrupted is True
