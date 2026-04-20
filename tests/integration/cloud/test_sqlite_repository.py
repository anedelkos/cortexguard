from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from cortexguard.cloud.persistence.repository import SQLiteIncidentRepository
from tests.unit.cloud.factories import make_incident as _make_incident
from tests.unit.cloud.factories import make_outcome as _make_outcome


@pytest_asyncio.fixture
async def repo(tmp_path: Path) -> SQLiteIncidentRepository:
    r = SQLiteIncidentRepository(tmp_path / "test.db")
    await r.initialize()
    return r


@pytest.mark.integration
class TestSQLiteIncidentRepository:
    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, tmp_path: Path) -> None:
        import aiosqlite

        db_path = tmp_path / "init_test.db"
        r = SQLiteIncidentRepository(db_path)
        await r.initialize()
        async with aiosqlite.connect(str(db_path)) as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cursor:
                tables = {row[0] for row in await cursor.fetchall()}
        assert "incidents" in tables
        assert "outcomes" in tables

    @pytest.mark.asyncio
    async def test_incident_round_trip(self, repo: SQLiteIncidentRepository) -> None:
        record = _make_incident()
        await repo.save_incident(record)
        result = await repo.get_incident(record.incident_id)
        assert result is not None
        assert result.incident_id == record.incident_id
        assert result.anomaly_key == record.anomaly_key
        assert result.candidate_plan_json is None
        assert result.created_at.isoformat() == record.created_at.isoformat()

    @pytest.mark.asyncio
    async def test_outcome_round_trip(self, repo: SQLiteIncidentRepository) -> None:
        escalation_id = str(uuid.uuid4())
        record = _make_outcome(escalation_id=escalation_id)
        await repo.save_outcome(record)
        result = await repo.get_outcome_by_escalation(escalation_id)
        assert result is not None
        assert result.outcome_id == record.outcome_id
        assert result.status == record.status
        assert result.notes is None
        assert result.completed_at.isoformat() == record.completed_at.isoformat()
        assert result.linked_at.isoformat() == record.linked_at.isoformat()

    @pytest.mark.asyncio
    async def test_get_incident_returns_none_for_unknown(
        self, repo: SQLiteIncidentRepository
    ) -> None:
        result = await repo.get_incident("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_recent_incidents_limit(self, repo: SQLiteIncidentRepository) -> None:
        base_time = datetime(2025, 1, 1, tzinfo=UTC)
        records = [_make_incident(created_at=base_time + timedelta(seconds=i)) for i in range(3)]
        for r in records:
            await repo.save_incident(r)
        result = await repo.list_recent_incidents(limit=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_recent_incidents_newest_first(self, repo: SQLiteIncidentRepository) -> None:
        base_time = datetime(2025, 1, 1, tzinfo=UTC)
        records = [_make_incident(created_at=base_time + timedelta(seconds=i)) for i in range(3)]
        for r in records:
            await repo.save_incident(r)
        result = await repo.list_recent_incidents(limit=3)
        assert len(result) == 3
        assert result[0].created_at >= result[1].created_at >= result[2].created_at
