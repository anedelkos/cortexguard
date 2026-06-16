"""Tests for Retrieval Explainability — similarity scores surfaced alongside incident IDs."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from cortexguard.cloud.graph.nodes import make_retrieve_similar_incidents_node
from cortexguard.cloud.graph.state import CloudPlanningState
from cortexguard.cloud.persistence.repository import InMemoryIncidentRepository
from cortexguard.cloud.retrieval.embedder import MockEmbedder
from cortexguard.cloud.retrieval.store import RetrievalStore
from cortexguard.cloud.retrieval.vector_store import InMemoryVectorStore
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth
from tests.unit.cloud.factories import make_incident


def _make_packet(anomaly_key: str = "S1.1_MISGRASP") -> MaydayPacket:
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
                contributing_detectors=["LogicalRuleDetector"],
            )
        ],
    )


def _make_state(packet: MaydayPacket, incident_id: str | None = None) -> CloudPlanningState:
    return CloudPlanningState(
        request=packet,
        incident_id=incident_id,
        retrieved_incidents=[],
        retrieved_incident_records=[],
        candidate_plan=None,
        validation_result=None,
        decision=None,
        rationale=None,
        confidence=None,
        needs_human_review=False,
        errors=[],
    )


@pytest.mark.asyncio
async def test_retrieval_store_returns_scores() -> None:
    store = RetrievalStore(embedder=MockEmbedder(), vector_store=InMemoryVectorStore())
    record = make_incident(anomaly_key="S1.1_MISGRASP")
    await store.index_incident(record)

    packet = _make_packet("S1.1_MISGRASP")
    results = await store.retrieve_similar(packet)

    assert len(results) == 1
    incident_id, score, record_out = results[0]
    assert incident_id == record.incident_id
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_retrieve_node_populates_retrieved_incidents() -> None:
    repo = InMemoryIncidentRepository()
    record = make_incident(anomaly_key="S1.1_MISGRASP")
    await repo.save_incident(record)

    store = RetrievalStore(
        embedder=MockEmbedder(),
        vector_store=InMemoryVectorStore(),
        repo=repo,
    )
    await store.index_incident(record)

    node = make_retrieve_similar_incidents_node(retrieval_store=store, repo=repo)
    packet = _make_packet("S1.1_MISGRASP")
    state = _make_state(packet)
    result = await node(state)

    assert len(result["retrieved_incidents"]) > 0
    entry = result["retrieved_incidents"][0]
    assert "incident_id" in entry
    assert "similarity_score" in entry
    assert entry["incident_id"] == record.incident_id
    assert isinstance(entry["similarity_score"], float)


@pytest.mark.asyncio
async def test_incident_record_persists_retrieved_incidents_json() -> None:
    repo = InMemoryIncidentRepository()
    persisted = make_incident(anomaly_key="S1.1_MISGRASP")
    await repo.save_incident(persisted)

    retrieved = make_incident(anomaly_key="S1.1_MISGRASP")
    await repo.save_incident(retrieved)

    store = RetrievalStore(embedder=MockEmbedder(), vector_store=InMemoryVectorStore())
    await store.index_incident(retrieved)

    node = make_retrieve_similar_incidents_node(retrieval_store=store, repo=repo)
    packet = _make_packet("S1.1_MISGRASP")
    state = _make_state(packet, incident_id=persisted.incident_id)
    await node(state)

    saved = await repo.get_incident(persisted.incident_id)
    assert saved is not None
    assert saved.retrieved_incidents_json is not None
    parsed: list[dict[str, Any]] = json.loads(saved.retrieved_incidents_json)
    assert len(parsed) > 0
    assert "incident_id" in parsed[0]
    assert "similarity_score" in parsed[0]


@pytest.mark.asyncio
async def test_mcp_latest_planner_decision_returns_scores() -> None:
    from pydantic import AnyUrl

    from cortexguard.cloud.mcp_server import ResourceHandler

    sample_retrieved = json.dumps([{"incident_id": "inc-abc", "similarity_score": 0.87}])
    incident = make_incident(retrieved_incidents_json=sample_retrieved)
    mock_repo = AsyncMock()
    mock_repo.list_recent_incidents = AsyncMock(return_value=[incident])

    handler = ResourceHandler(repo=mock_repo, capability_registry_json='{"capabilities": {}}')
    results = await handler.handle(AnyUrl("cortexguard://latest_planner_decision"))

    data = json.loads(results[0].content)
    assert "retrieved_incidents" in data
    ri = data["retrieved_incidents"]
    assert isinstance(ri, list)
    assert len(ri) == 1
    assert ri[0]["incident_id"] == "inc-abc"
    assert ri[0]["similarity_score"] == pytest.approx(0.87)


@pytest.mark.asyncio
async def test_retrieval_similarity_metric_observed() -> None:
    import cortexguard.cloud.runtime as runtime

    repo = InMemoryIncidentRepository()
    record = make_incident(anomaly_key="OVERHEAT")
    await repo.save_incident(record)

    store = RetrievalStore(embedder=MockEmbedder(), vector_store=InMemoryVectorStore())
    await store.index_incident(record)

    node = make_retrieve_similar_incidents_node(retrieval_store=store, repo=repo)
    packet = _make_packet("OVERHEAT")
    state = _make_state(packet)

    before = _read_histogram_count(
        runtime.cloud_retrieval_similarity_score, {"anomaly_key": "OVERHEAT"}
    )
    await node(state)
    after = _read_histogram_count(
        runtime.cloud_retrieval_similarity_score, {"anomaly_key": "OVERHEAT"}
    )

    assert after > before


def _read_histogram_count(metric: Any, labels: dict[str, str]) -> float:
    """Return the total observation count for the given label set."""
    try:
        for family in metric.labels(**labels).collect():
            for sample in family.samples:
                if sample.name.endswith("_count") and not sample.labels:
                    return float(sample.value)
    except Exception:
        pass
    return 0.0
