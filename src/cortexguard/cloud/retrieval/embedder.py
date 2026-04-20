"""Embedding backends used to index and search cloud incident episodes."""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedderProtocol(Protocol):
    def embed(self, text: str) -> list[float]: ...


class MockEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.0] * 384


class MiniLMEmbedder:
    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, text: str) -> list[float]:
        result = next(iter(self._model.embed([text])))  # type: ignore[union-attr]
        return [float(x) for x in result]


def get_embedder(backend: str | None = None) -> EmbedderProtocol:
    selected = backend or os.getenv("CLOUD_EMBEDDER_BACKEND", "mock")
    if selected == "miniLM":
        return MiniLMEmbedder()
    return MockEmbedder()
