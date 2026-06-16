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
    cloud_retrieval_outcome_boost: float = field(
        default_factory=lambda: float(os.getenv("CLOUD_RETRIEVAL_OUTCOME_BOOST", "0.2"))
    )
    cloud_retrieval_failure_penalty: float = field(
        default_factory=lambda: float(os.getenv("CLOUD_RETRIEVAL_FAILURE_PENALTY", "0.1"))
    )
    # Postgres DSN, used when incident_store == "postgres"
    db_url: str | None = field(default_factory=lambda: os.getenv("CLOUD_DB_URL"))
    # Shared-secret auth, unset disables auth (local dev only)
    api_key: str | None = field(default_factory=lambda: os.getenv("CLOUD_API_KEY"))
    # SQS: unset keeps in-process async mode; set enables worker-based async mode
    sqs_queue_url: str | None = field(default_factory=lambda: os.getenv("CLOUD_SQS_QUEUE_URL"))
    sqs_region: str = field(default_factory=lambda: os.getenv("CLOUD_SQS_REGION", "us-east-1"))
    # LangGraph checkpointer backend: "memory" (dev), "sqlite" (single-process), "postgres" (production)
    checkpoint_store: str = field(
        default_factory=lambda: os.getenv("CLOUD_CHECKPOINT_STORE", "memory")
    )
    # Absolute path for sqlite checkpoint file; defaults to "checkpoints.db" (CWD).
    # In production ECS, set CLOUD_CHECKPOINT_DB_PATH to a persistent volume path.
    checkpoint_db_path: str | None = field(
        default_factory=lambda: os.getenv("CLOUD_CHECKPOINT_DB_PATH")
    )
