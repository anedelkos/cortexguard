"""Unit tests for SQSCloudOrchestrator."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from cortexguard.cloud.orchestrator import PlanningResult, SQSCloudOrchestrator
from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.cloud.queue.sqs import SQSQueue
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth


def _make_packet(trace_id: str | None = None) -> MaydayPacket:
    return MaydayPacket(
        trace_id=trace_id or str(uuid.uuid4()),
        device_id="dev-01",
        timestamp=datetime.now(UTC),
        health=SystemHealth(),
    )


def _make_incident(
    trace_id: str,
    decision: str = "pending",
    plan_json: str | None = None,
) -> IncidentRecord:
    return IncidentRecord(
        incident_id=str(uuid.uuid4()),
        escalation_id=trace_id,
        trace_id=trace_id,
        device_id="dev-01",
        anomaly_key="UNKNOWN",
        anomaly_type="detected",
        severity="unknown",
        summary="test",
        raw_packet_json="{}",
        retrieved_incident_ids_json="[]",
        candidate_plan_json=plan_json,
        validation_errors_json="[]",
        decision=decision,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.save_incident = AsyncMock()
    repo.get_incident_by_trace_id = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_sqs() -> MagicMock:
    sqs = MagicMock(spec=SQSQueue)
    sqs.enqueue = AsyncMock()
    return sqs


@pytest.fixture
def orchestrator(mock_repo: AsyncMock, mock_sqs: MagicMock) -> SQSCloudOrchestrator:
    return SQSCloudOrchestrator(repo=mock_repo, sqs_queue=mock_sqs)


@pytest.mark.asyncio
async def test_submit_writes_pending_record(
    orchestrator: SQSCloudOrchestrator,
    mock_repo: AsyncMock,
    mock_sqs: MagicMock,
) -> None:
    packet = _make_packet()
    returned_trace_id = await orchestrator.submit(packet)

    assert returned_trace_id == packet.trace_id
    mock_repo.save_incident.assert_awaited_once()
    saved: IncidentRecord = mock_repo.save_incident.call_args[0][0]
    assert saved.trace_id == packet.trace_id
    assert saved.decision == "pending"
    assert saved.device_id == "dev-01"


@pytest.mark.asyncio
async def test_submit_enqueues_to_sqs(
    orchestrator: SQSCloudOrchestrator,
    mock_sqs: MagicMock,
) -> None:
    packet = _make_packet()
    await orchestrator.submit(packet)

    mock_sqs.enqueue.assert_awaited_once()
    payload: dict[str, Any] = mock_sqs.enqueue.call_args[0][0]
    assert payload["trace_id"] == packet.trace_id
    assert isinstance(payload["packet"], dict)


@pytest.mark.asyncio
async def test_get_result_unknown_trace_id(
    orchestrator: SQSCloudOrchestrator,
    mock_repo: AsyncMock,
) -> None:
    mock_repo.get_incident_by_trace_id.return_value = None
    result = await orchestrator.get_result("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_result_pending(
    orchestrator: SQSCloudOrchestrator,
    mock_repo: AsyncMock,
) -> None:
    trace_id = str(uuid.uuid4())
    mock_repo.get_incident_by_trace_id.return_value = _make_incident(trace_id, decision="pending")
    result = await orchestrator.get_result(trace_id)
    assert result == "pending"


@pytest.mark.asyncio
async def test_get_result_needs_human(
    orchestrator: SQSCloudOrchestrator,
    mock_repo: AsyncMock,
) -> None:
    trace_id = str(uuid.uuid4())
    mock_repo.get_incident_by_trace_id.return_value = _make_incident(
        trace_id, decision="needs_human"
    )
    result = await orchestrator.get_result(trace_id)
    assert isinstance(result, PlanningResult)
    assert result.decision == "needs_human"
    assert result.plan is None


@pytest.mark.asyncio
async def test_get_result_plan_ready(
    orchestrator: SQSCloudOrchestrator,
    mock_repo: AsyncMock,
) -> None:
    from cortexguard.edge.models.goal import GoalContext
    from cortexguard.edge.models.plan import Plan, PlanSource, PlanType

    trace_id = str(uuid.uuid4())
    plan = Plan(
        plan_id=str(uuid.uuid4()),
        context=GoalContext(
            goal_id=str(uuid.uuid4()),
            user_prompt="test",
            intent="Remediate test anomaly",
            priority=1,
        ),
        steps=[],
        plan_type=PlanType.REMEDIATION,
        source=PlanSource.CLOUD_AGENT,
        trace_id=trace_id,
    )
    mock_repo.get_incident_by_trace_id.return_value = _make_incident(
        trace_id, decision="plan_ready", plan_json=plan.model_dump_json()
    )
    result = await orchestrator.get_result(trace_id)
    assert isinstance(result, PlanningResult)
    assert result.decision == "plan_ready"
    assert result.plan is not None
    assert result.plan.plan_id == plan.plan_id


@pytest.mark.asyncio
async def test_get_result_corrupt_plan_json_returns_no_plan(
    orchestrator: SQSCloudOrchestrator,
    mock_repo: AsyncMock,
) -> None:
    trace_id = str(uuid.uuid4())
    mock_repo.get_incident_by_trace_id.return_value = _make_incident(
        trace_id, decision="plan_ready", plan_json='{"not": "a valid plan"}'
    )
    result = await orchestrator.get_result(trace_id)
    assert isinstance(result, PlanningResult)
    assert result.plan is None
