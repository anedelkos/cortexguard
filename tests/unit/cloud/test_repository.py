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
    async def test_list_recent_outcomes_respects_limit_and_order(self) -> None:
        repo = InMemoryIncidentRepository()
        first = _make_outcome(notes="first")
        second = _make_outcome(notes="second")
        third = _make_outcome(notes="third")
        await repo.save_outcome(first)
        await repo.save_outcome(second)
        await repo.save_outcome(third)

        result = await repo.list_recent_outcomes(limit=2)
        assert len(result) == 2
        assert result[0].outcome_id == third.outcome_id
        assert result[1].outcome_id == second.outcome_id

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

    @pytest.mark.asyncio
    async def test_update_operator_resolution_persists_field(self) -> None:
        """update_operator_resolution should store JSON and return True for a known incident."""
        repo = InMemoryIncidentRepository()
        record = _make_incident()
        await repo.save_incident(record)
        resolution_json = '{"outcome": "resolved"}'
        result = await repo.update_operator_resolution(record.incident_id, resolution_json)
        assert result is True
        updated = await repo.get_incident(record.incident_id)
        assert updated is not None
        assert updated.operator_resolution_json == resolution_json

    @pytest.mark.asyncio
    async def test_update_operator_resolution_returns_false_for_unknown_id(self) -> None:
        """update_operator_resolution should return False when the incident does not exist."""
        repo = InMemoryIncidentRepository()
        result = await repo.update_operator_resolution("does-not-exist", '{"outcome": "resolved"}')
        assert result is False

    @pytest.mark.asyncio
    async def test_get_incident_by_trace_id_returns_record(self) -> None:
        """get_incident_by_trace_id should find a record by its trace_id column."""
        repo = InMemoryIncidentRepository()
        record = _make_incident()
        await repo.save_incident(record)
        result = await repo.get_incident_by_trace_id(record.trace_id)
        assert result is not None
        assert result.incident_id == record.incident_id

    @pytest.mark.asyncio
    async def test_get_incident_by_trace_id_returns_none_for_unknown(self) -> None:
        """get_incident_by_trace_id should return None when trace_id is not present."""
        repo = InMemoryIncidentRepository()
        result = await repo.get_incident_by_trace_id("nonexistent-trace-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieved_incidents_json_round_trips(self) -> None:
        """retrieved_incidents_json should survive a save/load cycle unchanged."""
        import json

        repo = InMemoryIncidentRepository()
        payload = json.dumps([{"incident_id": "abc", "similarity_score": 0.75}])
        record = _make_incident(retrieved_incidents_json=payload)
        await repo.save_incident(record)
        loaded = await repo.get_incident(record.incident_id)
        assert loaded is not None
        assert loaded.retrieved_incidents_json == payload
        parsed = json.loads(loaded.retrieved_incidents_json)
        assert parsed[0]["incident_id"] == "abc"
        assert parsed[0]["similarity_score"] == pytest.approx(0.75)
