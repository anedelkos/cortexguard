"""High-level retrieval service combining embeddings and vector search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.cloud.retrieval.embedder import EmbedderProtocol
from cortexguard.cloud.retrieval.vector_store import VectorStoreProtocol
from cortexguard.edge.models.mayday_packet import MaydayPacket

if TYPE_CHECKING:
    from cortexguard.cloud.persistence.repository import IncidentRepositoryProtocol


class RetrievalStoreProtocol(Protocol):
    async def retrieve_similar(
        self, packet: MaydayPacket, top_k: int = 5
    ) -> list[tuple[str, float, IncidentRecord | None]]: ...


class RetrievalStore:
    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        repo: IncidentRepositoryProtocol | None = None,
        outcome_boost: float = 0.2,
        failure_penalty: float = 0.1,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._repo = repo
        self._outcome_boost = outcome_boost
        self._failure_penalty = failure_penalty

    async def index_incident(self, record: IncidentRecord) -> None:
        vector = self._embedder.embed(record.summary)
        payload = {
            "anomaly_key": record.anomaly_key,
            "severity": record.severity,
            "decision": record.decision,
        }
        await self._vector_store.upsert(id=record.incident_id, vector=vector, payload=payload)

    async def retrieve_similar(
        self, packet: MaydayPacket, top_k: int = 5
    ) -> list[tuple[str, float, IncidentRecord | None]]:
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

        if self._repo is not None:
            from cortexguard.cloud.retrieval.ranker import score_incident

            triples: list[tuple[str, float, IncidentRecord | None]] = []
            for r in raw:
                record = await self._repo.get_incident(r.id)
                if record is None:
                    triples.append((r.id, r.score, None))
                else:
                    composite = score_incident(
                        record,
                        r.score,
                        self._outcome_boost,
                        self._failure_penalty,
                    )
                    triples.append((r.id, composite, record))
            triples.sort(key=lambda t: t[1], reverse=True)
            return triples[:top_k]

        no_repo_results: list[tuple[str, float, IncidentRecord | None]] = [
            (r.id, r.score, None) for r in sorted(raw, key=lambda r: r.score, reverse=True)
        ]
        return no_repo_results[:top_k]
