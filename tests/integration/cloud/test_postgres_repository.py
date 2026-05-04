"""Integration tests for PostgresIncidentRepository.

Requires a running Postgres instance. Set CLOUD_DB_URL to a valid DSN, e.g.:

    CLOUD_DB_URL=postgresql://cortexguard:cortexguard@localhost:5432/cortexguard_test \
        pytest tests/integration/cloud/test_postgres_repository.py -v

Skipped automatically when CLOUD_DB_URL is unset.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from cortexguard.cloud.persistence.models import IncidentRecord, OutcomeRecord
from cortexguard.cloud.persistence.postgres_repository import PostgresIncidentRepository

pytestmark = pytest.mark.integration

_DSN = os.getenv("CLOUD_DB_URL", "")


def _skip_if_no_db() -> None:
    if not _DSN:
        pytest.skip("CLOUD_DB_URL not set — skipping Postgres integration tests")


@pytest_asyncio.fixture
async def repo() -> AsyncGenerator[PostgresIncidentRepository, None]:
    _skip_if_no_db()
    r = PostgresIncidentRepository(_DSN)
    await r.initialize()
    yield r
    await r.close()


def _make_incident(*, decision: str = "pending", trace_id: str | None = None) -> IncidentRecord:
    return IncidentRecord(
        incident_id=str(uuid.uuid4()),
        escalation_id=str(uuid.uuid4()),
        trace_id=trace_id or str(uuid.uuid4()),
        device_id="test-device",
        anomaly_key="test_anomaly",
        anomaly_type="detected",
        severity="high",
        summary="Test incident for integration test",
        raw_packet_json="{}",
        retrieved_incident_ids_json="[]",
        candidate_plan_json=None,
        validation_errors_json="[]",
        decision=decision,
        created_at=datetime.now(UTC),
    )


def _make_outcome(escalation_id: str) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_id=str(uuid.uuid4()),
        escalation_id=escalation_id,
        decision_id=str(uuid.uuid4()),
        device_id="test-device",
        status="resolved",
        completed_at=datetime.now(UTC),
        notes="Integration test resolution",
        failure_reason=None,
        linked_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_save_and_get_incident(repo: PostgresIncidentRepository) -> None:
    incident = _make_incident(decision="needs_human")
    await repo.save_incident(incident)

    fetched = await repo.get_incident(incident.incident_id)
    assert fetched is not None
    assert fetched.incident_id == incident.incident_id
    assert fetched.decision == "needs_human"
    assert fetched.anomaly_key == "test_anomaly"


@pytest.mark.asyncio
async def test_upsert_incident(repo: PostgresIncidentRepository) -> None:
    incident = _make_incident(decision="pending")
    await repo.save_incident(incident)

    updated = IncidentRecord(**{**incident.__dict__, "decision": "plan_ready"})
    await repo.save_incident(updated)

    fetched = await repo.get_incident(incident.incident_id)
    assert fetched is not None
    assert fetched.decision == "plan_ready"


@pytest.mark.asyncio
async def test_get_incident_by_trace_id(repo: PostgresIncidentRepository) -> None:
    trace_id = str(uuid.uuid4())
    incident = _make_incident(trace_id=trace_id)
    await repo.save_incident(incident)

    fetched = await repo.get_incident_by_trace_id(trace_id)
    assert fetched is not None
    assert fetched.trace_id == trace_id


@pytest.mark.asyncio
async def test_get_incident_not_found(repo: PostgresIncidentRepository) -> None:
    result = await repo.get_incident(str(uuid.uuid4()))
    assert result is None


@pytest.mark.asyncio
async def test_save_and_get_outcome(repo: PostgresIncidentRepository) -> None:
    incident = _make_incident()
    await repo.save_incident(incident)

    outcome = _make_outcome(incident.escalation_id)
    await repo.save_outcome(outcome)

    fetched = await repo.get_outcome_by_escalation(incident.escalation_id)
    assert fetched is not None
    assert fetched.status == "resolved"


@pytest.mark.asyncio
async def test_list_recent_incidents(repo: PostgresIncidentRepository) -> None:
    now = datetime.now(UTC)
    # Explicit 1-second spacing so ORDER BY created_at DESC is deterministic.
    for offset in range(3):
        inc = _make_incident()
        inc = IncidentRecord(**{**inc.__dict__, "created_at": now - timedelta(seconds=2 - offset)})
        await repo.save_incident(inc)

    results = await repo.list_recent_incidents(limit=3)
    assert len(results) >= 3
    for i in range(len(results) - 1):
        assert results[i].created_at >= results[i + 1].created_at


@pytest.mark.asyncio
async def test_list_recent_outcomes(repo: PostgresIncidentRepository) -> None:
    incident = _make_incident()
    await repo.save_incident(incident)
    await repo.save_outcome(_make_outcome(incident.escalation_id))

    results = await repo.list_recent_outcomes(limit=10)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_update_operator_resolution(repo: PostgresIncidentRepository) -> None:
    incident = _make_incident()
    await repo.save_incident(incident)

    updated = await repo.update_operator_resolution(incident.incident_id, '{"outcome": "resolved"}')
    assert updated is True

    fetched = await repo.get_incident(incident.incident_id)
    assert fetched is not None
    assert fetched.operator_resolution_json == '{"outcome": "resolved"}'


@pytest.mark.asyncio
async def test_update_operator_resolution_not_found(repo: PostgresIncidentRepository) -> None:
    result = await repo.update_operator_resolution(str(uuid.uuid4()), '{"outcome": "resolved"}')
    assert result is False
