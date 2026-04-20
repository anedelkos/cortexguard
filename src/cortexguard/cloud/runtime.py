"""FastAPI application assembly for the CortexGuard cloud planner service."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import Counter, Histogram

from cortexguard.cloud.api.health import get_health_router
from cortexguard.cloud.api.mayday import get_mayday_router
from cortexguard.cloud.api.outcomes import get_outcomes_router
from cortexguard.cloud.config import CloudConfig
from cortexguard.cloud.orchestrator import CloudOrchestrator
from cortexguard.cloud.persistence.repository import (
    IncidentRepositoryProtocol,
    InMemoryIncidentRepository,
    SQLiteIncidentRepository,
)
from cortexguard.cloud.planner.factory import get_llm_client
from cortexguard.cloud.planner.llm_client import LLMClientProtocol
from cortexguard.cloud.retrieval.embedder import get_embedder
from cortexguard.cloud.retrieval.seeder import SeedLoader
from cortexguard.cloud.retrieval.store import RetrievalStore
from cortexguard.cloud.retrieval.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStoreProtocol,
)
from cortexguard.cloud.validation.capability_adapter import CapabilityAdapter
from cortexguard.cloud.validation.plan_validator import PlanValidator
from cortexguard.common.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _setup_cloud_tracing() -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "cortexguard-cloud"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    logger.info("OTEL tracing configured: endpoint=%s", endpoint)


cloud_needs_human_total = Counter(
    "cloud_needs_human_total",
    "Number of cloud planner decisions that require human intervention",
)
cloud_planning_requests_total = Counter(
    "cloud_planning_requests_total",
    "Total cloud planning requests received",
)
cloud_planning_duration_seconds = Histogram(
    "cloud_planning_duration_seconds",
    "End-to-end planning workflow duration in seconds",
)
cloud_llm_duration_seconds = Histogram(
    "cloud_llm_duration_seconds",
    "LLM plan generation duration in seconds",
)
cloud_retrieval_duration_seconds = Histogram(
    "cloud_retrieval_duration_seconds",
    "Incident retrieval duration in seconds",
)
cloud_validation_failures_total = Counter(
    "cloud_validation_failures_total",
    "Number of candidate plans that failed validation",
)
cloud_decisions_total = Counter(
    "cloud_decisions_total",
    "Cloud planning decisions by type",
    ["decision"],
)
cloud_outcome_status_total = Counter(
    "cloud_outcome_status_total",
    "Execution outcome status totals",
    ["status"],
)


def create_cloud_app(
    config: CloudConfig,
    readiness_checks: list[Callable[[], Awaitable[None]]] | None = None,
) -> FastAPI:
    # --- Incident store ---
    _sqlite_repo: SQLiteIncidentRepository | None = None
    repo: IncidentRepositoryProtocol
    if config.incident_store == "sqlite":
        _sqlite_repo = SQLiteIncidentRepository(config.db_path)
        repo = _sqlite_repo
    else:
        repo = InMemoryIncidentRepository()

    # --- Embedder + vector store ---
    embedder = get_embedder(config.embedder_backend)
    _qdrant_store: QdrantVectorStore | None = None
    vector_store: VectorStoreProtocol
    if config.vector_store_backend == "qdrant":
        _qdrant_store = QdrantVectorStore(config.qdrant_url)
        vector_store = _qdrant_store
    else:
        vector_store = InMemoryVectorStore()

    retrieval_store = RetrievalStore(embedder, vector_store)

    # --- LLM client ---
    llm_client: LLMClientProtocol | None = get_llm_client(
        config.llm_backend, api_key=config.anthropic_api_key
    )

    validator = PlanValidator(
        CapabilityAdapter.load_default(), min_confidence=config.min_confidence
    )
    orchestrator = CloudOrchestrator(
        repo=repo,
        retrieval_store=retrieval_store,
        llm_client=llm_client,
        validator=validator,
    )

    async def _check_db() -> None:
        await repo.list_recent_incidents(limit=1)

    async def _check_vector_store() -> None:
        if _qdrant_store is not None:
            await _qdrant_store._client.get_collections()  # type: ignore[attr-defined]

    default_checks: list[Callable[[], Awaitable[None]]] = [_check_db, _check_vector_store]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        logger.info("CortexGuard cloud API starting up (config=%s)", config)
        _setup_cloud_tracing()
        if _sqlite_repo is not None:
            await _sqlite_repo.initialize()
        if _qdrant_store is not None:
            await _qdrant_store.initialize()
        await SeedLoader().seed_if_empty(retrieval_store, repo)
        yield
        logger.info("CortexGuard cloud API shutting down")

    app = FastAPI(
        title="CortexGuard Cloud API",
        description="Cloud-tier deliberative planner for CortexGuard edge agents.",
        lifespan=lifespan,
    )

    from fastapi import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    app.include_router(
        get_health_router(readiness_checks=readiness_checks or default_checks), prefix=""
    )
    app.include_router(get_mayday_router(orchestrator=orchestrator), prefix="/api/v1")
    app.include_router(get_outcomes_router(repo=repo), prefix="/api/v1")

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_cloud_app(CloudConfig())
