"""MCP server exposing CortexGuard incident history and planning tools to operators."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import mcp.types as types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl

from cortexguard.cloud.config import CloudConfig
from cortexguard.cloud.graph.workflow import create_checkpointer
from cortexguard.cloud.orchestrator import CloudOrchestrator, PlanningResult
from cortexguard.cloud.persistence.repository import (
    IncidentRepositoryProtocol,
    InMemoryIncidentRepository,
    SQLiteIncidentRepository,
)
from cortexguard.cloud.planner.factory import get_llm_client
from cortexguard.cloud.planner.throttler import LLMThrottler
from cortexguard.cloud.queue.sqs import SQSQueue
from cortexguard.cloud.retrieval.embedder import get_embedder
from cortexguard.cloud.retrieval.store import RetrievalStore
from cortexguard.cloud.retrieval.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStoreProtocol,
)
from cortexguard.cloud.validation.capability_adapter import CapabilityAdapter
from cortexguard.cloud.validation.plan_validator import PlanValidator
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth
from cortexguard.edge.models.plan import Plan

logger = logging.getLogger(__name__)

_VALID_OUTCOMES = frozenset({"resolved", "escalated_further", "hardware_replaced", "aborted"})


# ---------------------------------------------------------------------------
# Metrics reporter
# ---------------------------------------------------------------------------


class MetricsReporter:
    """Fire-and-forget metrics reporting to the cloud API process."""

    def __init__(self, api_url: str = "http://localhost:8001/internal/mcp-event") -> None:
        self._api_url = api_url

    def record_tool_call(self, tool_name: str) -> None:
        self._post({"tool": tool_name})

    def record_resolution(self, outcome: str) -> None:
        self._post({"outcome": outcome})

    def _post(self, payload: dict[str, str]) -> None:
        async def _send() -> None:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=2.0) as client:
                    await client.post(self._api_url, json=payload)
            except Exception:
                logger.exception("Failed to send MCP metrics to %s", self._api_url)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_send())
        except RuntimeError:
            logger.debug("No running event loop, dropping metrics payload")


# ---------------------------------------------------------------------------
# Handler protocol & registry
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolHandler(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def input_schema(self) -> dict[str, Any]: ...

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, handler: ToolHandler) -> None:
        self._handlers[handler.name] = handler

    def list_tools(self) -> list[types.Tool]:
        return [
            types.Tool(
                name=h.name,
                description=h.description,
                inputSchema=h.input_schema,
            )
            for h in self._handlers.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        return await handler.handle(arguments)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass
class _PlanPollResult:
    decision: str
    plan: dict[str, Any] | None
    confidence: float
    rationale: str
    incident_id: str


async def _poll_result(
    orchestrator: CloudOrchestrator, trace_id: str, timeout_s: float
) -> PlanningResult | None:
    """Poll orchestrator.get_result until the plan is ready or *timeout_s* elapses."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        result = await orchestrator.get_result(trace_id)
        if isinstance(result, PlanningResult):
            return result
        await asyncio.sleep(0.5)
    return None


async def _poll_and_collect(
    orchestrator: CloudOrchestrator,
    repo: IncidentRepositoryProtocol,
    submitted_trace_id: str,
    timeout_s: float,
    on_incident: Callable[[Any], None] | None = None,
) -> _PlanPollResult | None:
    """Poll for a plan result and collect incident metadata.

    Returns None on timeout.  When an incident is found *on_incident* is called
    so the caller can mutate the record (e.g. set ``source`` or ``parent_incident_id``)
    before it is saved.
    """
    plan_result = await _poll_result(orchestrator, submitted_trace_id, timeout_s)
    if plan_result is None:
        return None

    incident = await repo.get_incident_by_trace_id(submitted_trace_id)
    incident_id = incident.incident_id if incident is not None else submitted_trace_id
    confidence = incident.confidence or 0.0 if incident is not None else 0.0
    rationale = incident.rationale or "" if incident is not None else ""

    if incident is not None:
        if on_incident is not None:
            on_incident(incident)
        await repo.save_incident(incident)

    plan_dict = plan_result.plan.model_dump(mode="json") if plan_result.plan is not None else None
    return _PlanPollResult(
        decision=plan_result.decision,
        plan=plan_dict,
        confidence=confidence,
        rationale=rationale,
        incident_id=incident_id,
    )


# ---------------------------------------------------------------------------
# Tool handler implementations
# ---------------------------------------------------------------------------


class GetLatestIncidentHandler:
    def __init__(self, repo: IncidentRepositoryProtocol, metrics: MetricsReporter) -> None:
        self._repo = repo
        self._metrics = metrics

    @property
    def name(self) -> str:
        return "get_latest_incident"

    @property
    def description(self) -> str:
        return (
            "Get the most recent CortexGuard planning incident, use this first "
            "when an alert fires to see the decision, plan, rationale, and retrieved "
            "similar past incidents with similarity scores."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._metrics.record_tool_call(self.name)
        records = await self._repo.list_recent_incidents(limit=1)
        if not records:
            return {"error": "No incidents found."}
        r = records[0]
        plan_dict: dict[str, Any] | None = None
        if r.candidate_plan_json:
            try:
                plan_dict = json.loads(r.candidate_plan_json)
            except json.JSONDecodeError:
                plan_dict = None
        retrieved_incidents: list[dict[str, Any]] = []
        if r.retrieved_incidents_json:
            try:
                retrieved_incidents = json.loads(r.retrieved_incidents_json)
            except json.JSONDecodeError:
                retrieved_incidents = []
        return {
            "incident_id": r.incident_id,
            "device_id": r.device_id,
            "anomaly_key": r.anomaly_key,
            "severity": r.severity,
            "decision": r.decision,
            "rationale": r.rationale,
            "confidence": r.confidence,
            "plan": plan_dict,
            "retrieved_incidents": retrieved_incidents,
            "created_at": r.created_at.isoformat(),
        }


class ValidatePlanHandler:
    def __init__(self, validator: PlanValidator, metrics: MetricsReporter) -> None:
        self._validator = validator
        self._metrics = metrics

    @property
    def name(self) -> str:
        return "validate_plan"

    @property
    def description(self) -> str:
        return "Validate a plan against the capability registry."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "properties": {"steps": {"type": "array"}},
                    "required": ["steps"],
                }
            },
            "required": ["plan"],
        }

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._metrics.record_tool_call(self.name)
        try:
            plan = Plan.model_validate(arguments["plan"])
            result = self._validator.validate(plan, confidence=1.0, needs_human_review=False)
            return {
                "valid": result.passed,
                "errors": result.errors,
                "warnings": [],
            }
        except Exception as exc:
            return {
                "valid": False,
                "errors": [str(exc)],
                "warnings": [],
            }


class CreateRemediationPlanHandler:
    def __init__(
        self,
        orchestrator: CloudOrchestrator,
        repo: IncidentRepositoryProtocol,
        config: CloudConfig,
        metrics: MetricsReporter,
    ) -> None:
        self._orchestrator = orchestrator
        self._repo = repo
        self._config = config
        self._metrics = metrics

    @property
    def name(self) -> str:
        return "create_remediation_plan"

    @property
    def description(self) -> str:
        return "Generate a new remediation plan for a given anomaly via the cloud planner."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "anomaly_key": {"type": "string"},
                "severity": {"type": "string"},
                "device_id": {"type": "string"},
                "summary": {
                    "type": "string",
                    "description": "Optional human-supplied context about the incident.",
                },
                "state_summary": {
                    "type": "string",
                    "description": "Optional human-supplied summary of the system state.",
                },
                "proposed_local_attempts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["anomaly_key", "severity", "device_id"],
        }

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._metrics.record_tool_call(self.name)
        anomaly_key: str = arguments["anomaly_key"]
        severity_str: str = arguments["severity"]
        device_id: str = arguments["device_id"]
        summary: str | None = arguments.get("summary")
        state_summary: str | None = arguments.get("state_summary")
        proposed_local_attempts: list[str] = arguments.get("proposed_local_attempts", [])

        operator_context: list[str] = []
        if summary:
            operator_context.append(f"OPERATOR_SUMMARY: {summary}")
        if state_summary:
            operator_context.append(f"STATE_SUMMARY: {state_summary}")
        reasoning_trace: list[dict[str, object]] = [
            {"text": s} for s in operator_context + list(proposed_local_attempts)
        ]

        try:
            severity = AnomalySeverity[severity_str.upper()]
        except KeyError:
            severity = AnomalySeverity.MEDIUM

        trace_id = str(uuid.uuid4())
        packet = MaydayPacket(
            trace_id=trace_id,
            device_id=device_id,
            timestamp=datetime.now(UTC),
            health=SystemHealth(),
            anomalies=[
                AnomalyEvent(
                    id=str(uuid.uuid4()),
                    key=anomaly_key,
                    severity=severity,
                    timestamp=datetime.now(UTC),
                    metadata={},
                    score=1.0,
                    contributing_detectors=[],
                )
            ],
            reasoning_trace=reasoning_trace,
        )

        submitted_trace_id = await self._orchestrator.submit(packet)
        timeout_s = self._config.cloud_llm_timeout_s + 10.0

        def _tag_source(incident: Any) -> None:
            incident.source = "mcp"

        result = await _poll_and_collect(
            self._orchestrator,
            self._repo,
            submitted_trace_id,
            timeout_s,
            on_incident=_tag_source,
        )

        if result is None:
            return {
                "decision": "timeout",
                "plan": None,
                "confidence": 0.0,
                "rationale": "Planning timed out.",
                "incident_id": submitted_trace_id,
            }

        return {
            "decision": result.decision,
            "plan": result.plan,
            "confidence": result.confidence,
            "rationale": result.rationale,
            "incident_id": result.incident_id,
        }


class ProposeAlternativePlanHandler:
    def __init__(
        self,
        orchestrator: CloudOrchestrator,
        repo: IncidentRepositoryProtocol,
        config: CloudConfig,
        metrics: MetricsReporter,
    ) -> None:
        self._orchestrator = orchestrator
        self._repo = repo
        self._config = config
        self._metrics = metrics

    @property
    def name(self) -> str:
        return "propose_alternative_plan"

    @property
    def description(self) -> str:
        return "Generate an alternative remediation plan avoiding a specific step or approach."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "incident_id": {"type": "string"},
                "avoid": {"type": "string"},
            },
            "required": ["incident_id", "avoid"],
        }

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._metrics.record_tool_call(self.name)
        incident_id: str = arguments["incident_id"]
        avoid: str = arguments["avoid"]

        original = await self._repo.get_incident(incident_id)
        if original is None:
            return {
                "decision": "error",
                "plan": None,
                "confidence": 0.0,
                "rationale": f"Incident {incident_id!r} not found.",
                "incident_id": incident_id,
            }

        try:
            packet = MaydayPacket.model_validate_json(original.raw_packet_json)
        except Exception as exc:
            return {
                "decision": "error",
                "plan": None,
                "confidence": 0.0,
                "rationale": f"Failed to deserialize original packet: {exc}",
                "incident_id": incident_id,
            }

        normalized: list[dict[str, object]] = []
        for t in list(packet.reasoning_trace):
            if isinstance(t, dict):
                normalized.append(t)
            else:
                normalized.append({"text": str(t)})
        normalized.append({"text": f"CONSTRAINT: {avoid}"})
        packet.reasoning_trace = normalized
        new_trace_id = str(uuid.uuid4())
        packet.trace_id = new_trace_id

        submitted_trace_id = await self._orchestrator.submit(packet)
        timeout_s = self._config.cloud_llm_timeout_s + 10.0

        original_id = incident_id

        def _tag_alternative(incident: Any) -> None:
            incident.parent_incident_id = original_id
            incident.source = "mcp"

        result = await _poll_and_collect(
            self._orchestrator,
            self._repo,
            submitted_trace_id,
            timeout_s,
            on_incident=_tag_alternative,
        )

        if result is None:
            return {
                "decision": "timeout",
                "plan": None,
                "confidence": 0.0,
                "rationale": "Alternative planning timed out.",
                "incident_id": submitted_trace_id,
            }

        return {
            "decision": result.decision,
            "plan": result.plan,
            "confidence": result.confidence,
            "rationale": result.rationale,
            "incident_id": result.incident_id,
        }


class RecordOperatorResolutionHandler:
    def __init__(
        self,
        repo: IncidentRepositoryProtocol,
        retrieval_store: RetrievalStore | None,
        sqs_queue: SQSQueue | None,
        metrics: MetricsReporter,
    ) -> None:
        self._repo = repo
        self._retrieval_store = retrieval_store
        self._sqs_queue = sqs_queue
        self._metrics = metrics

    @property
    def name(self) -> str:
        return "record_operator_resolution"

    @property
    def description(self) -> str:
        return "Record how an operator resolved an incident after the fact."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "incident_id": {"type": "string"},
                "actions_taken": {"type": "string"},
                "outcome": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["incident_id", "actions_taken", "outcome"],
        }

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._metrics.record_tool_call(self.name)
        incident_id: str = arguments["incident_id"]
        actions_taken: str = arguments["actions_taken"]
        outcome: str = arguments["outcome"]
        notes: str | None = arguments.get("notes")

        if outcome not in _VALID_OUTCOMES:
            return {
                "error": (
                    f"Invalid outcome {outcome!r}. "
                    f"Must be one of: {', '.join(sorted(_VALID_OUTCOMES))}"
                )
            }

        resolution: dict[str, Any] = {
            "actions_taken": actions_taken,
            "outcome": outcome,
            "notes": notes,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        updated = await self._repo.update_operator_resolution(incident_id, json.dumps(resolution))
        if not updated:
            return {
                "incident_id": incident_id,
                "recorded": False,
                "error": f"Incident {incident_id!r} not found.",
            }

        embedding_updated = False
        record = await self._repo.get_incident(incident_id)
        if record is not None:
            enriched = (
                f"device={record.device_id} anomaly={record.anomaly_key} "
                f"severity={record.severity} decision={record.decision} "
                f"actions_taken={actions_taken} outcome={outcome}"
            )
            record.summary = enriched
            if self._retrieval_store is not None:
                try:
                    await self._retrieval_store.index_incident(record)
                    embedding_updated = True
                except Exception:
                    logger.exception("Failed to re-embed incident %s", incident_id)

        self._metrics.record_resolution(outcome)

        if self._sqs_queue is not None and record is not None:
            try:
                await self._sqs_queue.enqueue(
                    {
                        "action": "resume",
                        "incident_id": incident_id,
                        "trace_id": record.trace_id,
                        "operator_response": {
                            "approved": outcome in ("resolved", "escalated_further"),
                            "action": "reject" if outcome == "aborted" else "approve",
                            "outcome": outcome,
                            "notes": notes,
                        },
                    }
                )
                logger.info("Pushed resume SQS message for incident %s", incident_id)
            except Exception:
                logger.exception("Failed to push resume SQS message for incident %s", incident_id)

        return {
            "incident_id": incident_id,
            "recorded": True,
            "embedding_updated": embedding_updated,
        }


# ---------------------------------------------------------------------------
# Resource handler
# ---------------------------------------------------------------------------


def _build_capability_registry_json() -> str:
    adapter = CapabilityAdapter.load_default()
    registry_dict: dict[str, Any] = {
        "capabilities": {
            name: {
                "description": cap.description,
                "risk_level": cap.risk_level.value,
            }
            for name, cap in adapter.capabilities().items()
        }
    }
    return json.dumps(registry_dict)


class ResourceHandler:
    """MCP resource handler wrapping incident and capability registry data."""

    def __init__(
        self,
        repo: IncidentRepositoryProtocol,
        capability_registry_json: str,
    ) -> None:
        self._repo = repo
        self._capability_registry_json = capability_registry_json

    async def handle(self, uri: AnyUrl) -> list[ReadResourceContents]:
        uri_str = str(uri)

        if uri_str == "cortexguard://capability_registry":
            return [
                ReadResourceContents(
                    content=self._capability_registry_json, mime_type="application/json"
                )
            ]

        if uri_str == "cortexguard://recent_incident_summaries":
            records = await self._repo.list_recent_incidents(limit=20)
            summaries = [
                {
                    "incident_id": r.incident_id,
                    "timestamp": r.created_at.isoformat(),
                    "device_id": r.device_id,
                    "anomaly_key": r.anomaly_key,
                    "severity": r.severity,
                    "decision": r.decision,
                    "confidence": r.confidence,
                    "outcome": r.operator_resolution_json,
                    "source": r.source,
                }
                for r in records
            ]
            return [
                ReadResourceContents(content=json.dumps(summaries), mime_type="application/json")
            ]

        if uri_str == "cortexguard://latest_planner_decision":
            records = await self._repo.list_recent_incidents(limit=1)
            if not records:
                return [ReadResourceContents(content=json.dumps({}), mime_type="application/json")]
            r = records[0]
            plan_dict: dict[str, Any] | None = None
            if r.candidate_plan_json:
                try:
                    plan_dict = json.loads(r.candidate_plan_json)
                except json.JSONDecodeError:
                    plan_dict = None
            retrieved_incidents: list[dict[str, Any]] = []
            if r.retrieved_incidents_json:
                try:
                    retrieved_incidents = json.loads(r.retrieved_incidents_json)
                except json.JSONDecodeError:
                    retrieved_incidents = []
            detail: dict[str, Any] = {
                "incident_id": r.incident_id,
                "escalation_id": r.escalation_id,
                "trace_id": r.trace_id,
                "device_id": r.device_id,
                "anomaly_key": r.anomaly_key,
                "anomaly_type": r.anomaly_type,
                "severity": r.severity,
                "summary": r.summary,
                "decision": r.decision,
                "rationale": r.rationale,
                "confidence": r.confidence,
                "plan": plan_dict,
                "created_at": r.created_at.isoformat(),
                "source": r.source,
                "parent_incident_id": r.parent_incident_id,
                "operator_resolution_json": r.operator_resolution_json,
                "retrieved_incidents": retrieved_incidents,
            }
            return [ReadResourceContents(content=json.dumps(detail), mime_type="application/json")]

        raise ValueError(f"Unknown resource URI: {uri_str}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@dataclass
class CloudDependencies:
    """Wired dependencies for the cloud MCP server."""

    repo: IncidentRepositoryProtocol
    orchestrator: CloudOrchestrator | None
    validator: PlanValidator | None
    retrieval_store: RetrievalStore | None
    config: CloudConfig
    sqs_queue: SQSQueue | None
    metrics: MetricsReporter
    capability_registry_json: str


def create_mcp_server(deps: CloudDependencies) -> Server:
    """Assemble the MCP server with all dependencies wired."""
    server = Server("cortexguard")
    registry = ToolRegistry()

    registry.register(GetLatestIncidentHandler(repo=deps.repo, metrics=deps.metrics))

    if deps.validator is not None:
        registry.register(ValidatePlanHandler(validator=deps.validator, metrics=deps.metrics))

    if deps.orchestrator is not None:
        registry.register(
            CreateRemediationPlanHandler(
                orchestrator=deps.orchestrator,
                repo=deps.repo,
                config=deps.config,
                metrics=deps.metrics,
            )
        )

    if deps.orchestrator is not None:
        registry.register(
            ProposeAlternativePlanHandler(
                orchestrator=deps.orchestrator,
                repo=deps.repo,
                config=deps.config,
                metrics=deps.metrics,
            )
        )

    registry.register(
        RecordOperatorResolutionHandler(
            repo=deps.repo,
            retrieval_store=deps.retrieval_store,
            sqs_queue=deps.sqs_queue,
            metrics=deps.metrics,
        )
    )

    resource_handler = ResourceHandler(
        repo=deps.repo, capability_registry_json=deps.capability_registry_json
    )

    @server.list_resources()  # type: ignore[no-untyped-call]
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri="cortexguard://capability_registry",  # type: ignore[arg-type]
                name="Capability Registry",
                description="Full registry of edge capabilities available for planning.",
                mimeType="application/json",
            ),
            types.Resource(
                uri="cortexguard://recent_incident_summaries",  # type: ignore[arg-type]
                name="Recent Incident Summaries",
                description="The 20 most recent planning incident summaries.",
                mimeType="application/json",
            ),
            types.Resource(
                uri="cortexguard://latest_planner_decision",  # type: ignore[arg-type]
                name="Latest Planner Decision",
                description="Full detail of the most recent cloud planning incident.",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()  # type: ignore[no-untyped-call]
    async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        return await resource_handler.handle(uri)

    @server.list_tools()  # type: ignore[no-untyped-call]
    async def list_tools() -> list[types.Tool]:
        return registry.list_tools()

    @server.call_tool()  # type: ignore[no-untyped-call]
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await registry.call_tool(name, arguments)

    return server


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


async def build_cloud_dependencies(config: CloudConfig) -> CloudDependencies:
    """Build all cloud dependencies from config."""
    repo: IncidentRepositoryProtocol
    if config.incident_store == "sqlite":
        sqlite_repo = SQLiteIncidentRepository(config.db_path)
        await sqlite_repo.initialize()
        repo = sqlite_repo
    else:
        repo = InMemoryIncidentRepository()

    embedder = get_embedder(config.embedder_backend)

    vector_store: VectorStoreProtocol
    if config.vector_store_backend == "qdrant":
        qdrant_store = QdrantVectorStore(config.qdrant_url)
        await qdrant_store.initialize()
        vector_store = qdrant_store
    else:
        vector_store = InMemoryVectorStore()

    retrieval_store = RetrievalStore(
        embedder,
        vector_store,
        repo,
        outcome_boost=config.cloud_retrieval_outcome_boost,
        failure_penalty=config.cloud_retrieval_failure_penalty,
    )

    llm_client = get_llm_client(config.llm_backend, api_key=config.anthropic_api_key)
    throttled_llm = LLMThrottler(llm_client, config)

    adapter = CapabilityAdapter.load_default()
    validator = PlanValidator(adapter, min_confidence=config.min_confidence)

    orchestrator: CloudOrchestrator | None = None
    try:
        checkpointer = await create_checkpointer(config.checkpoint_store, config.checkpoint_db_path)
        orchestrator = CloudOrchestrator(
            repo=repo,
            retrieval_store=retrieval_store,
            llm_client=throttled_llm,
            validator=validator,
            checkpointer=checkpointer,
        )
    except Exception:
        logger.warning("Failed to create CloudOrchestrator, planning tools will be unavailable")

    sqs_queue = SQSQueue(config.sqs_queue_url, config.sqs_region) if config.sqs_queue_url else None
    metrics = MetricsReporter()
    capability_registry_json = _build_capability_registry_json()

    return CloudDependencies(
        repo=repo,
        orchestrator=orchestrator,
        validator=validator,
        retrieval_store=retrieval_store,
        config=config,
        sqs_queue=sqs_queue,
        metrics=metrics,
        capability_registry_json=capability_registry_json,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Initialise all dependencies and start the MCP stdio server."""
    config = CloudConfig()
    deps = await build_cloud_dependencies(config)
    server = create_mcp_server(deps)

    logger.info("CortexGuard MCP server starting")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
