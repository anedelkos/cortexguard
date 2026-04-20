"""High-level retrieval service combining embeddings and vector search."""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Protocol

from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.cloud.retrieval.embedder import EmbedderProtocol
from cortexguard.cloud.retrieval.vector_store import SearchResult, VectorStoreProtocol
from cortexguard.edge.models.mayday_packet import MaydayPacket

_SUCCESS_DECISIONS = {"plan_ready"}
_OUTCOME_BOOST = 1.1


class RetrievalStoreProtocol(Protocol):
    async def retrieve_similar(
        self, packet: MaydayPacket, top_k: int = 5
    ) -> list[SearchResult]: ...


class RetrievalStore:
    def __init__(self, embedder: EmbedderProtocol, vector_store: VectorStoreProtocol) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def index_incident(self, record: IncidentRecord) -> None:
        vector = self._embedder.embed(record.summary)
        payload = {
            "anomaly_key": record.anomaly_key,
            "severity": record.severity,
            "decision": record.decision,
        }
        await self._vector_store.upsert(id=record.incident_id, vector=vector, payload=payload)

    async def retrieve_similar(self, packet: MaydayPacket, top_k: int = 5) -> list[SearchResult]:
        anomaly_keys = [a.key for a in packet.anomalies]
        summary = (
            f"device={packet.device_id} anomalies={','.join(anomaly_keys)} "
            f"plan={packet.current_plan_id or 'none'}"
        )
        vector = self._embedder.embed(summary)
        filter_dict: dict[str, object] | None = None
        if len(anomaly_keys) == 1:
            filter_dict = {"anomaly_key": anomaly_keys[0]}
        raw = await self._vector_store.search(vector=vector, top_k=top_k, filter=filter_dict)
        boosted = sorted(
            (
                (
                    _dc_replace(r, score=r.score * _OUTCOME_BOOST)
                    if str(r.payload.get("decision", "")) in _SUCCESS_DECISIONS
                    else r
                )
                for r in raw
            ),
            key=lambda r: r.score,
            reverse=True,
        )
        return boosted[:top_k]
