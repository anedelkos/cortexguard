from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.cloud.persistence.repository import InMemoryIncidentRepository
from cortexguard.cloud.retrieval.embedder import MockEmbedder
from cortexguard.cloud.retrieval.seeder import SeedLoader
from cortexguard.cloud.retrieval.store import RetrievalStore
from cortexguard.cloud.retrieval.vector_store import InMemoryVectorStore
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth


def _make_packet(anomaly_key: str = "S1.1_MISGRASP") -> MaydayPacket:
    return MaydayPacket(
        trace_id=str(uuid.uuid4()),
        device_id="robot-arm-01",
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


def _make_incident(
    anomaly_key: str = "S1.1_MISGRASP", summary: str = "test incident"
) -> IncidentRecord:
    return IncidentRecord(
        incident_id=str(uuid.uuid4()),
        escalation_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        device_id="robot-arm-01",
        anomaly_key=anomaly_key,
        anomaly_type="repeated_failure",
        severity="high",
        summary=summary,
        raw_packet_json="{}",
        retrieved_incident_ids_json="[]",
        candidate_plan_json=None,
        validation_errors_json="[]",
        decision="plan_ready",
        created_at=datetime.now(UTC),
    )


def _make_store() -> RetrievalStore:
    return RetrievalStore(embedder=MockEmbedder(), vector_store=InMemoryVectorStore())


class TestRetrievalStore:
    @pytest.mark.asyncio
    async def test_index_and_retrieve_roundtrip(self) -> None:
        store = _make_store()
        record = _make_incident()
        await store.index_incident(record)
        packet = _make_packet(record.anomaly_key)
        results = await store.retrieve_similar(packet)
        assert len(results) == 1
        assert results[0].id == record.incident_id

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self) -> None:
        store = _make_store()
        for i in range(3):
            await store.index_incident(_make_incident(summary=f"incident {i}"))
        packet = _make_packet()
        results = await store.retrieve_similar(packet, top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_metadata_filter_excludes_non_matching(self) -> None:
        store = _make_store()
        record_a = _make_incident(anomaly_key="S1.1_MISGRASP")
        record_b = _make_incident(anomaly_key="S1.2_VISION_OCCLUSION")
        await store.index_incident(record_a)
        await store.index_incident(record_b)
        packet = _make_packet("S1.1_MISGRASP")
        results = await store.retrieve_similar(packet, top_k=5)
        assert all(r.payload["anomaly_key"] == "S1.1_MISGRASP" for r in results)
        assert len(results) == 1


class TestSeedLoader:
    @pytest.mark.asyncio
    async def test_inserts_records_into_empty_store(self) -> None:
        from pathlib import Path

        store = _make_store()
        repo = InMemoryIncidentRepository()
        loader = SeedLoader()
        expected = len(
            list((Path(__file__).parents[3] / "src/cortexguard/cloud/data/seeds").glob("*.json"))
        )
        count = await loader.seed_if_empty(store, repo)
        assert count == expected
        assert count > 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_already_seeded(self) -> None:
        store = _make_store()
        repo = InMemoryIncidentRepository()
        loader = SeedLoader()
        await loader.seed_if_empty(store, repo)
        count = await loader.seed_if_empty(store, repo)
        assert count == 0
