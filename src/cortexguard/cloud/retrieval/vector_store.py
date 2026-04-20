"""Vector-store backends used by the cloud retrieval layer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict[str, Any]


@runtime_checkable
class VectorStoreProtocol(Protocol):
    async def upsert(self, id: str, vector: list[float], payload: dict[str, Any]) -> None: ...

    async def search(
        self, vector: list[float], top_k: int, filter: dict[str, Any] | None
    ) -> list[SearchResult]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


@dataclass
class _Entry:
    id: str
    vector: list[float]
    payload: dict[str, Any]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._entries: list[_Entry] = []

    async def upsert(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        for entry in self._entries:
            if entry.id == id:
                entry.vector = vector
                entry.payload = payload
                return
        self._entries.append(_Entry(id=id, vector=vector, payload=payload))

    async def search(
        self, vector: list[float], top_k: int, filter: dict[str, Any] | None
    ) -> list[SearchResult]:
        candidates: list[_Entry] = []
        for entry in self._entries:
            if filter:
                if not all(entry.payload.get(k) == v for k, v in filter.items()):
                    continue
            candidates.append(entry)
        scored = sorted(
            candidates,
            key=lambda e: _cosine(vector, e.vector),
            reverse=True,
        )
        return [
            SearchResult(id=e.id, score=_cosine(vector, e.vector), payload=e.payload)
            for e in scored[:top_k]
        ]


class QdrantVectorStore:
    COLLECTION_NAME = "incidents"
    VECTOR_SIZE = 384

    def __init__(self, url: str, collection_name: str = COLLECTION_NAME) -> None:
        from qdrant_client import AsyncQdrantClient

        self._client = AsyncQdrantClient(url=url)
        self._collection_name = collection_name

    async def initialize(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        collections = await self._client.get_collections()
        names = {c.name for c in collections.collections}
        if self._collection_name not in names:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
            )

    async def upsert(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        from qdrant_client.models import PointStruct

        await self._client.upsert(
            collection_name=self._collection_name,
            points=[PointStruct(id=id, vector=vector, payload=payload)],
        )

    async def search(
        self, vector: list[float], top_k: int, filter: dict[str, Any] | None
    ) -> list[SearchResult]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qdrant_filter: Filter | None = None
        if filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter.items()
            ]
            qdrant_filter = Filter(must=conditions)  # type: ignore[arg-type]

        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [
            SearchResult(id=str(h.id), score=float(h.score), payload=dict(h.payload or {}))
            for h in response.points
        ]
