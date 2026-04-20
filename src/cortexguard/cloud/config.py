"""Environment-backed configuration for the cloud planner service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class CloudConfig:
    llm_backend: str = field(default_factory=lambda: os.getenv("CLOUD_LLM_BACKEND", "mock"))
    embedder_backend: str = field(
        default_factory=lambda: os.getenv("CLOUD_EMBEDDER_BACKEND", "mock")
    )
    vector_store_backend: str = field(
        default_factory=lambda: os.getenv("CLOUD_VECTOR_STORE_BACKEND", "in_memory")
    )
    incident_store: str = field(default_factory=lambda: os.getenv("CLOUD_INCIDENT_STORE", "sqlite"))
    db_path: str = field(default_factory=lambda: os.getenv("CLOUD_DB_PATH", "cortexguard_cloud.db"))
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("CLOUD_QDRANT_URL", "http://localhost:6333")
    )
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.getenv("CLOUD_ANTHROPIC_API_KEY")
    )
    min_confidence: float = field(
        default_factory=lambda: float(os.getenv("CLOUD_MIN_CONFIDENCE", "0.5"))
    )
