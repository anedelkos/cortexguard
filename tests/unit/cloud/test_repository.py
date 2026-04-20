from __future__ import annotations

import uuid

import pytest

from cortexguard.cloud.persistence.repository import InMemoryIncidentRepository
from tests.unit.cloud.factories import make_incident as _make_incident
from tests.unit.cloud.factories import make_outcome as _make_outcome


class TestInMemoryIncidentRepository:
    @pytest.mark.asyncio
    async def test_save_and_get_incident(self) -> None:
        repo = InMemoryIncidentRepository()
        record = _make_incident()
        await repo.save_incident(record)
        result = await repo.get_incident(record.incident_id)
        assert result == record

    @pytest.mark.asyncio
    async def test_get_incident_returns_none_for_unknown_id(self) -> None:
        repo = InMemoryIncidentRepository()
        result = await repo.get_incident("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_and_get_outcome_by_escalation(self) -> None:
        repo = InMemoryIncidentRepository()
        escalation_id = str(uuid.uuid4())
        record = _make_outcome(escalation_id=escalation_id)
        await repo.save_outcome(record)
        result = await repo.get_outcome_by_escalation(escalation_id)
        assert result == record

    @pytest.mark.asyncio
    async def test_get_outcome_returns_none_for_unknown_escalation(self) -> None:
        repo = InMemoryIncidentRepository()
        result = await repo.get_outcome_by_escalation("nonexistent-escalation")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_recent_incidents_respects_limit(self) -> None:
        repo = InMemoryIncidentRepository()
        records = [_make_incident() for _ in range(5)]
        for r in records:
            await repo.save_incident(r)
        result = await repo.list_recent_incidents(limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_recent_incidents_returns_newest_first(self) -> None:
        repo = InMemoryIncidentRepository()
        records = [_make_incident() for _ in range(3)]
        for r in records:
            await repo.save_incident(r)
        result = await repo.list_recent_incidents(limit=2)
        assert result[0].incident_id == records[2].incident_id
        assert result[1].incident_id == records[1].incident_id

    @pytest.mark.asyncio
    async def test_list_recent_incidents_returns_all_when_limit_exceeds_count(self) -> None:
        repo = InMemoryIncidentRepository()
        records = [_make_incident() for _ in range(2)]
        for r in records:
            await repo.save_incident(r)
        result = await repo.list_recent_incidents(limit=10)
        assert len(result) == 2
