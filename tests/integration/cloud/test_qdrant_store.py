from __future__ import annotations

import uuid

import pytest

from cortexguard.cloud.retrieval.vector_store import QdrantVectorStore


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_upsert_and_search() -> None:
    try:
        import httpx

        r = httpx.get("http://localhost:6333/healthz", timeout=2.0)
        if r.status_code != 200:
            pytest.skip("Qdrant not reachable")
    except Exception:
        pytest.skip("Qdrant not reachable")

    collection = f"test-{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(url="http://localhost:6333", collection_name=collection)
    await store.initialize()

    vector = [1.0] + [0.0] * 383
    await store.upsert(id="test-id-1", vector=vector, payload={"anomaly_key": "TEST"})

    results = await store.search(vector=vector, top_k=1, filter=None)
    assert len(results) == 1
    assert results[0].id == "test-id-1"
    assert results[0].score > 0.9
