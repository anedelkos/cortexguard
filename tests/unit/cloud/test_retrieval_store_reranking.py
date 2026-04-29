"""Tests for learning-to-rank re-ranking in RetrievalStore."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.cloud.retrieval.embedder import MockEmbedder
from cortexguard.cloud.retrieval.store import RetrievalStore
from cortexguard.cloud.retrieval.vector_store import SearchResult
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth
from tests.unit.cloud.factories import make_incident


def _make_packet(anomaly_key: str = "OVERHEAT") -> MaydayPacket:
    return MaydayPacket(
        trace_id=str(uuid.uuid4()),
        device_id="device-01",
        timestamp=datetime.now(UTC),
        health=SystemHealth(),
        anomalies=[
            AnomalyEvent(
                id=str(uuid.uuid4()),
                key=anomaly_key,
                timestamp=datetime.now(UTC),
                severity=AnomalySeverity.HIGH,
                score=0.9,
                contributing_detectors=["HardLimitDetector"],
            )
        ],
    )


@pytest.mark.asyncio
async def test_reranker_reorders_results() -> None:
    low_sim_id = str(uuid.uuid4())
    high_sim_id = str(uuid.uuid4())

    # low_sim record: similarity=0.5 but has resolved outcome → composite = 0.5 + 0.2 = 0.7
    resolved_record = make_incident(
        incident_id=low_sim_id,
        decision="needs_human",
        operator_resolution_json=json.dumps({"outcome": "resolved"}),
    )
    # high_sim record: similarity=0.6, no resolution → composite = 0.6 - 0.1 = 0.5
    penalised_record = make_incident(
        incident_id=high_sim_id,
        decision="needs_human",
        operator_resolution_json=None,
    )

    records: dict[str, IncidentRecord] = {
        low_sim_id: resolved_record,
        high_sim_id: penalised_record,
    }
    mock_repo = AsyncMock()
    mock_repo.get_incident = AsyncMock(side_effect=lambda iid: records.get(iid))

    mock_vector_store = AsyncMock()
    mock_vector_store.search = AsyncMock(
        return_value=[
            SearchResult(id=high_sim_id, score=0.6, payload={}),
            SearchResult(id=low_sim_id, score=0.5, payload={}),
        ]
    )
    mock_vector_store.upsert = AsyncMock(return_value=None)

    store = RetrievalStore(
        embedder=MockEmbedder(),
        vector_store=mock_vector_store,
        repo=mock_repo,
        outcome_boost=0.2,
        failure_penalty=0.1,
    )

    packet = _make_packet()
    results = await store.retrieve_similar(packet)

    assert len(results) == 2
    top_id, top_score, top_record = results[0]
    assert top_id == low_sim_id, "resolved record should rank first despite lower raw similarity"
    assert top_score == pytest.approx(0.7)
    assert top_record is resolved_record
    second_id, second_score, second_record = results[1]
    assert second_id == high_sim_id
    assert second_score == pytest.approx(0.5)
    assert second_record is penalised_record


@pytest.mark.asyncio
async def test_reranker_falls_back_gracefully_on_missing_record() -> None:
    known_id = str(uuid.uuid4())
    missing_id = str(uuid.uuid4())

    known_record = make_incident(incident_id=known_id)

    records: dict[str, IncidentRecord] = {known_id: known_record}
    mock_repo = AsyncMock()
    mock_repo.get_incident = AsyncMock(side_effect=lambda iid: records.get(iid))

    mock_vector_store = AsyncMock()
    mock_vector_store.search = AsyncMock(
        return_value=[
            SearchResult(id=known_id, score=0.7, payload={}),
            SearchResult(id=missing_id, score=0.9, payload={}),
        ]
    )
    mock_vector_store.upsert = AsyncMock(return_value=None)

    store = RetrievalStore(
        embedder=MockEmbedder(),
        vector_store=mock_vector_store,
        repo=mock_repo,
        outcome_boost=0.2,
        failure_penalty=0.1,
    )

    packet = _make_packet()
    results = await store.retrieve_similar(packet)

    assert len(results) == 2
    ids = [rid for rid, _, _ in results]
    assert known_id in ids
    assert missing_id in ids
    missing_result = next((rid, score, rec) for rid, score, rec in results if rid == missing_id)
    assert missing_result[1] == pytest.approx(0.9)
    assert missing_result[2] is None
