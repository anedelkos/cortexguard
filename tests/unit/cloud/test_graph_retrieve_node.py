from __future__ import annotations

import uuid
from datetime import UTC, datetime

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


def _make_packet() -> MaydayPacket:
    return MaydayPacket(
        trace_id=str(uuid.uuid4()),
        device_id="robot-arm-01",
        timestamp=datetime.now(UTC),
        health=SystemHealth(),
        anomalies=[
            AnomalyEvent(
                id=str(uuid.uuid4()),
                key="S1.1_MISGRASP",
                timestamp=datetime.now(UTC),
                severity=AnomalySeverity.HIGH,
                score=0.9,
                contributing_detectors=["LogicalRuleDetector"],
            )
        ],
    )


def _make_state(packet: MaydayPacket) -> CloudPlanningState:
    return CloudPlanningState(
        request=packet,
        incident_id=None,
        retrieved_incidents=[],
        candidate_plan=None,
        validation_result=None,
        decision=None,
        rationale=None,
        confidence=None,
        needs_human_review=False,
        errors=[],
    )


@pytest.mark.asyncio
async def test_retrieve_node_populates_retrieved_incidents() -> None:
    repo = InMemoryIncidentRepository()
    record = make_incident(anomaly_key="S1.1_MISGRASP")
    await repo.save_incident(record)

    store = RetrievalStore(embedder=MockEmbedder(), vector_store=InMemoryVectorStore())
    await store.index_incident(record)

    node = make_retrieve_similar_incidents_node(retrieval_store=store, repo=repo)
    packet = _make_packet()
    state = _make_state(packet)
    result = await node(state)

    assert len(result["retrieved_incidents"]) > 0
    assert result["retrieved_incidents"][0].incident_id == record.incident_id


@pytest.mark.asyncio
async def test_retrieve_node_none_store_returns_empty() -> None:
    node = make_retrieve_similar_incidents_node(retrieval_store=None)
    packet = _make_packet()
    state = _make_state(packet)
    result = await node(state)
    assert result["retrieved_incidents"] == []
