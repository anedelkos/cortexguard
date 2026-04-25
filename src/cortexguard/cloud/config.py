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
    cloud_mayday_rate_limit: str = field(
        default_factory=lambda: os.getenv("CLOUD_MAYDAY_RATE_LIMIT", "10/minute")
    )
    cloud_result_rate_limit: str = field(
        default_factory=lambda: os.getenv("CLOUD_RESULT_RATE_LIMIT", "60/minute")
    )
    cloud_outcome_rate_limit: str = field(
        default_factory=lambda: os.getenv("CLOUD_OUTCOME_RATE_LIMIT", "30/minute")
    )
    cloud_llm_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("CLOUD_LLM_TIMEOUT_S", "20.0"))
    )
    cloud_llm_max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("CLOUD_LLM_MAX_CONCURRENCY", "4"))
    )
    cloud_llm_max_retries: int = field(
        default_factory=lambda: int(os.getenv("CLOUD_LLM_MAX_RETRIES", "2"))
    )
    cloud_llm_base_backoff_ms: int = field(
        default_factory=lambda: int(os.getenv("CLOUD_LLM_BASE_BACKOFF_MS", "500"))
    )
