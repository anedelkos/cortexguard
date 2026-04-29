"""FastAPI application assembly for the CortexGuard cloud planner service."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

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

cloud_http_requests_total = Counter(
    "cloud_http_requests_total",
    "Total HTTP requests to the cloud planner API",
    ["method", "status_code", "route"],
)
cloud_http_request_duration_ms = Histogram(
    "cloud_http_request_duration_ms",
    "HTTP request duration in milliseconds for the cloud planner API",
    ["route"],
)
cloud_rate_limited_total = Counter(
    "cloud_rate_limited_total",
    "Total rate-limited requests to the cloud planner API",
    ["route"],
)

cloud_llm_requests_total = Counter(
    "cloud_llm_requests_total",
    "Total outbound LLM requests by provider and outcome",
    ["provider", "outcome"],
)
cloud_llm_retries_total = Counter(
    "cloud_llm_retries_total",
    "Total outbound LLM retry attempts by provider",
    ["provider"],
)
cloud_llm_inflight = Gauge(
    "cloud_llm_inflight",
    "Current number of in-flight LLM requests by provider",
    ["provider"],
)

cloud_mcp_tool_calls_total = Counter(
    "cloud_mcp_tool_calls_total",
    "MCP tool invocations",
    ["tool"],
)
cloud_operator_resolutions_total = Counter(
    "cloud_operator_resolutions_total",
    "Operator resolutions recorded via MCP",
    ["outcome"],
)

cloud_retrieval_similarity_score = Histogram(
    "cloud_retrieval_similarity_score",
    "Top-1 RAG similarity score per planning run",
    ["anomaly_key"],
)


def create_cloud_app(
    config: CloudConfig,
    readiness_checks: list[Callable[[], Awaitable[None]]] | None = None,
    llm_client: LLMClientProtocol | None = None,
) -> FastAPI:
    """Assemble the FastAPI cloud planner application with all subsystems wired."""
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

    retrieval_store = RetrievalStore(
        embedder,
        vector_store,
        repo,
        outcome_boost=config.cloud_retrieval_outcome_boost,
        failure_penalty=config.cloud_retrieval_failure_penalty,
    )

    # --- LLM client ---
    if llm_client is None:
        llm_client = get_llm_client(config.llm_backend, api_key=config.anthropic_api_key)

    if llm_client is not None:
        from cortexguard.cloud.planner.throttler import LLMThrottler

        llm_client = LLMThrottler(llm_client, config)

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

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
        route = request.url.path
        cloud_rate_limited_total.labels(route=route).inc()
        return Response(
            content='{"error": "Too many requests"}',
            status_code=429,
            media_type="application/json",
        )

    @app.middleware("http")
    async def _track_request_metrics(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        route = request.url.path
        cloud_http_requests_total.labels(
            method=request.method, status_code=str(response.status_code), route=route
        ).inc()
        cloud_http_request_duration_ms.labels(route=route).observe(duration_ms)
        return response

    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    app.include_router(
        get_health_router(readiness_checks=readiness_checks or default_checks), prefix=""
    )
    app.include_router(
        get_mayday_router(
            orchestrator=orchestrator,
            mayday_rate_limit=config.cloud_mayday_rate_limit,
            result_rate_limit=config.cloud_result_rate_limit,
            _limiter=limiter,
        ),
        prefix="/api/v1",
    )
    app.include_router(
        get_outcomes_router(
            repo=repo,
            outcome_rate_limit=config.cloud_outcome_rate_limit,
            _limiter=limiter,
        ),
        prefix="/api/v1",
    )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/internal/mcp-event", include_in_schema=False)
    async def mcp_event(request: Request) -> Response:
        try:
            body = await request.json()
            tool = body.get("tool")
            outcome = body.get("outcome")
            if tool:
                cloud_mcp_tool_calls_total.labels(tool=tool).inc()
            if outcome:
                cloud_operator_resolutions_total.labels(outcome=outcome).inc()
        except Exception:  # nosec B110 — best-effort metric update; never fail the caller
            pass
        return Response(content='{"ok":true}', media_type="application/json")

    return app


app = create_cloud_app(CloudConfig())
