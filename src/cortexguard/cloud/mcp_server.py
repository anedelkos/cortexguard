"""MCP server exposing CortexGuard incident history and planning tools to operators."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast

import mcp.types as types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl

from cortexguard.cloud.config import CloudConfig
from cortexguard.cloud.orchestrator import CloudOrchestrator, PlanningResult
from cortexguard.cloud.persistence.repository import (
    IncidentRepositoryProtocol,
    InMemoryIncidentRepository,
    SQLiteIncidentRepository,
)
from cortexguard.cloud.planner.explain_client import ExplainClientProtocol
from cortexguard.cloud.planner.factory import get_explain_client, get_llm_client
from cortexguard.cloud.planner.throttler import LLMThrottler
from cortexguard.cloud.retrieval.embedder import get_embedder
from cortexguard.cloud.retrieval.store import RetrievalStore
from cortexguard.cloud.retrieval.vector_store import InMemoryVectorStore, QdrantVectorStore
from cortexguard.cloud.validation.capability_adapter import CapabilityAdapter
from cortexguard.cloud.validation.plan_validator import PlanValidator
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth
from cortexguard.edge.models.plan import Plan

logger = logging.getLogger(__name__)

_VALID_OUTCOMES = frozenset({"resolved", "escalated_further", "hardware_replaced", "aborted"})

server: Server = Server("cortexguard")

# These are populated in main() before stdio_server starts.
# Module-level assignments enable unittest.mock.patch.object in tests.
_repo: IncidentRepositoryProtocol = InMemoryIncidentRepository()
_orchestrator: CloudOrchestrator | None = None
_explain_client: ExplainClientProtocol | None = None
_validator: PlanValidator | None = None
_retrieval_store: RetrievalStore | None = None
_config: CloudConfig = CloudConfig()


def _post_mcp_event(payload: dict[str, str]) -> None:
    """Fire-and-forget POST to the API process to increment MCP metrics."""
    import asyncio

    async def _send() -> None:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.post("http://localhost:8001/internal/mcp-event", json=payload)
        except Exception:  # nosec B110 — fire-and-forget; metric loss is acceptable
            pass

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send())
    except RuntimeError:
        pass


def _get_mcp_counter(label: str) -> None:
    _post_mcp_event({"tool": label})


def _get_resolution_counter(outcome: str) -> None:
    _post_mcp_event({"outcome": outcome})


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


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@server.list_resources()  # type: ignore[no-untyped-call]
async def list_resources() -> list[types.Resource]:
    """Return the three static CortexGuard resources."""
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
    """Dispatch URI reads to the appropriate handler."""
    uri_str = str(uri)

    if uri_str == "cortexguard://capability_registry":
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
        return [
            ReadResourceContents(content=json.dumps(registry_dict), mime_type="application/json")
        ]

    if uri_str == "cortexguard://recent_incident_summaries":
        records = await _repo.list_recent_incidents(limit=20)
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
        return [ReadResourceContents(content=json.dumps(summaries), mime_type="application/json")]

    if uri_str == "cortexguard://latest_planner_decision":
        records = await _repo.list_recent_incidents(limit=1)
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
# Tools
# ---------------------------------------------------------------------------


@server.list_tools()  # type: ignore[no-untyped-call]
async def list_tools() -> list[types.Tool]:
    """Return all available MCP tools."""
    return [
        types.Tool(
            name="get_latest_incident",
            description=(
                "Get the most recent CortexGuard planning incident — use this first "
                "when an alert fires to see the decision, plan, rationale, and retrieved "
                "similar past incidents with similarity scores."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="validate_plan",
            description="Validate a plan against the capability registry.",
            inputSchema={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "object",
                        "properties": {"steps": {"type": "array"}},
                        "required": ["steps"],
                    }
                },
                "required": ["plan"],
            },
        ),
        types.Tool(
            name="create_remediation_plan",
            description="Generate a new remediation plan for a given anomaly via the cloud planner.",
            inputSchema={
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
            },
        ),
        types.Tool(
            name="explain_plan",
            description=(
                "Explain a plan in plain English for a hardware operator. "
                "At least one of 'incident_id' or 'plan' must be provided."
            ),
            inputSchema={
                "type": "object",
                "description": "At least one of incident_id or plan must be provided.",
                "properties": {
                    "incident_id": {"type": "string"},
                    "plan": {"type": "object"},
                    "rationale": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="propose_alternative_plan",
            description="Generate an alternative remediation plan avoiding a specific step or approach.",
            inputSchema={
                "type": "object",
                "properties": {
                    "incident_id": {"type": "string"},
                    "avoid": {"type": "string"},
                },
                "required": ["incident_id", "avoid"],
            },
        ),
        types.Tool(
            name="record_operator_resolution",
            description="Record how an operator resolved an incident after the fact.",
            inputSchema={
                "type": "object",
                "properties": {
                    "incident_id": {"type": "string"},
                    "actions_taken": {"type": "string"},
                    "outcome": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["incident_id", "actions_taken", "outcome"],
            },
        ),
    ]


@server.call_tool()  # type: ignore[no-untyped-call]
async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch MCP tool calls to the appropriate handler."""
    match name:
        case "get_latest_incident":
            return await _handle_get_latest_incident()
        case "validate_plan":
            return await _handle_validate_plan(arguments)
        case "create_remediation_plan":
            return await _handle_create_remediation_plan(arguments)
        case "explain_plan":
            return await _handle_explain_plan(arguments)
        case "propose_alternative_plan":
            return await _handle_propose_alternative_plan(arguments)
        case "record_operator_resolution":
            return await _handle_record_operator_resolution(arguments)
        case _:
            return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def _handle_get_latest_incident() -> dict[str, Any]:
    """Return full detail of the most recent planning incident."""
    _get_mcp_counter("get_latest_incident")
    records = await _repo.list_recent_incidents(limit=1)
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


async def _handle_validate_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate a plan dict against the capability registry."""
    _get_mcp_counter("validate_plan")
    if _validator is None:
        raise RuntimeError("_validator is not initialised")
    try:
        plan = Plan.model_validate(arguments["plan"])
        result = _validator.validate(plan, confidence=1.0, needs_human_review=False)
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


async def _handle_create_remediation_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    """Generate a remediation plan via the cloud orchestrator."""
    _get_mcp_counter("create_remediation_plan")
    if _orchestrator is None:
        raise RuntimeError("_orchestrator is not initialised")
    anomaly_key: str = arguments["anomaly_key"]
    severity_str: str = arguments["severity"]
    device_id: str = arguments["device_id"]
    summary: str | None = arguments.get("summary")
    state_summary: str | None = arguments.get("state_summary")
    proposed_local_attempts: list[str] = arguments.get("proposed_local_attempts", [])

    # Incorporate optional operator context into the reasoning trace as structured dicts
    operator_context: list[str] = []
    if summary:
        operator_context.append(f"OPERATOR_SUMMARY: {summary}")
    if state_summary:
        operator_context.append(f"STATE_SUMMARY: {state_summary}")
    reasoning_trace = cast(
        list[dict[str, object]],
        [{"text": s} for s in operator_context + list(proposed_local_attempts)],
    )

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

    submitted_trace_id = await _orchestrator.submit(packet)
    timeout_s = _config.cloud_llm_timeout_s + 10.0
    plan_result = await _poll_result(_orchestrator, submitted_trace_id, timeout_s)

    if plan_result is None:
        return {
            "decision": "timeout",
            "plan": None,
            "confidence": 0.0,
            "rationale": "Planning timed out.",
            "incident_id": submitted_trace_id,
        }

    # Mark source="mcp" on the persisted incident; look up by trace_id (not incident_id)
    incident = await _repo.get_incident_by_trace_id(submitted_trace_id)

    incident_id = incident.incident_id if incident is not None else submitted_trace_id
    confidence: float = 0.0
    rationale: str = ""
    if incident is not None:
        confidence = incident.confidence or 0.0
        rationale = incident.rationale or ""
        incident.source = "mcp"
        await _repo.save_incident(incident)

    plan_dict: dict[str, Any] | None = None
    if plan_result.plan is not None:
        plan_dict = plan_result.plan.model_dump(mode="json")

    return {
        "decision": plan_result.decision,
        "plan": plan_dict,
        "confidence": confidence,
        "rationale": rationale,
        "incident_id": incident_id,
    }


async def _handle_explain_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    """Explain a plan in plain English using the explain client."""
    _get_mcp_counter("explain_plan")
    if _explain_client is None:
        raise RuntimeError("_explain_client is not initialised")
    incident_id: str | None = arguments.get("incident_id")
    plan_arg: dict[str, Any] | None = arguments.get("plan")
    rationale_arg: str | None = arguments.get("rationale")

    if incident_id is None and plan_arg is None:
        return {"explanation": "Error: at least one of incident_id or plan must be provided."}

    anomaly_key = "unknown"
    severity = "unknown"
    rationale = rationale_arg or ""
    steps: list[str] = []

    if incident_id is not None:
        record = await _repo.get_incident(incident_id)
        if record is None:
            return {"explanation": f"Error: incident {incident_id!r} not found."}
        anomaly_key = record.anomaly_key
        severity = record.severity
        rationale = rationale or record.rationale or ""
        if record.candidate_plan_json:
            try:
                plan_data = json.loads(record.candidate_plan_json)
                steps = [
                    f"{i + 1}. {s.get('description', s.get('action', str(s)))}"
                    for i, s in enumerate(plan_data.get("steps", []))
                ]
            except (json.JSONDecodeError, AttributeError):
                steps = []
    elif plan_arg is not None:
        raw_steps: list[dict[str, Any]] = plan_arg.get("steps", [])
        steps = [
            f"{i + 1}. {s.get('description', s.get('action', str(s)))}"
            for i, s in enumerate(raw_steps)
        ]

    steps_text = "\n".join(steps) if steps else "(no steps)"
    prompt = (
        "You are a safety system analyst. Explain the following recovery plan in plain English.\n"
        "Describe what each step does, why it was chosen, and what the expected outcome is.\n"
        "Be concise and clear — the audience is a hardware operator, not a software engineer.\n\n"
        f"Anomaly: {anomaly_key} (severity: {severity})\n"
        f"Rationale: {rationale}\n"
        f"Plan steps:\n{steps_text}"
    )

    explanation = await _explain_client.explain(prompt)
    return {"explanation": explanation}


async def _handle_propose_alternative_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    """Propose an alternative plan that avoids a specified step or approach."""
    _get_mcp_counter("propose_alternative_plan")
    if _orchestrator is None:
        raise RuntimeError("_orchestrator is not initialised")
    incident_id: str = arguments["incident_id"]
    avoid: str = arguments["avoid"]

    original = await _repo.get_incident(incident_id)
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

    # Normalize any existing string entries into dicts, then append constraint as a dict
    normalized = []
    for t in list(packet.reasoning_trace):
        if isinstance(t, dict):
            normalized.append(t)
        else:
            normalized.append({"text": str(t)})
    normalized.append({"text": f"CONSTRAINT: {avoid}"})
    packet.reasoning_trace = normalized
    new_trace_id = str(uuid.uuid4())
    packet.trace_id = new_trace_id

    submitted_trace_id = await _orchestrator.submit(packet)
    timeout_s = _config.cloud_llm_timeout_s + 10.0
    plan_result = await _poll_result(_orchestrator, submitted_trace_id, timeout_s)

    if plan_result is None:
        return {
            "decision": "timeout",
            "plan": None,
            "confidence": 0.0,
            "rationale": "Alternative planning timed out.",
            "incident_id": submitted_trace_id,
        }

    # Update new incident with parent_incident_id and source="mcp"; look up by trace_id
    new_incident = await _repo.get_incident_by_trace_id(submitted_trace_id)

    new_incident_id = new_incident.incident_id if new_incident is not None else submitted_trace_id
    new_confidence: float = 0.0
    new_rationale: str = ""
    if new_incident is not None:
        new_confidence = new_incident.confidence or 0.0
        new_rationale = new_incident.rationale or ""
        new_incident.parent_incident_id = incident_id
        new_incident.source = "mcp"
        await _repo.save_incident(new_incident)

    plan_dict: dict[str, Any] | None = None
    if plan_result.plan is not None:
        plan_dict = plan_result.plan.model_dump(mode="json")

    return {
        "decision": plan_result.decision,
        "plan": plan_dict,
        "confidence": new_confidence,
        "rationale": new_rationale,
        "incident_id": new_incident_id,
    }


async def _handle_record_operator_resolution(arguments: dict[str, Any]) -> dict[str, Any]:
    """Record an operator's resolution for an incident and re-embed the enriched summary."""
    _get_mcp_counter("record_operator_resolution")
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

    if _retrieval_store is None:
        raise RuntimeError("_retrieval_store is not initialised")

    resolution: dict[str, Any] = {
        "actions_taken": actions_taken,
        "outcome": outcome,
        "notes": notes,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    updated = await _repo.update_operator_resolution(incident_id, json.dumps(resolution))
    if not updated:
        return {
            "incident_id": incident_id,
            "recorded": False,
            "error": f"Incident {incident_id!r} not found.",
        }

    # Re-embed with enriched summary that incorporates operator resolution context
    embedding_updated = False
    record = await _repo.get_incident(incident_id)
    if record is not None:
        enriched = (
            f"device={record.device_id} anomaly={record.anomaly_key} "
            f"severity={record.severity} decision={record.decision} "
            f"actions_taken={actions_taken} outcome={outcome}"
        )
        record.summary = enriched
        try:
            await _retrieval_store.index_incident(record)
            embedding_updated = True
        except Exception:
            logger.exception("Failed to re-embed incident %s", incident_id)

    _get_resolution_counter(outcome)

    return {
        "incident_id": incident_id,
        "recorded": True,
        "embedding_updated": embedding_updated,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Initialise all dependencies and start the MCP stdio server."""
    global _repo, _orchestrator, _explain_client, _validator, _retrieval_store, _config

    _config = CloudConfig()

    if _config.incident_store == "sqlite":
        sqlite_repo = SQLiteIncidentRepository(_config.db_path)
        await sqlite_repo.initialize()
        _repo = sqlite_repo
    else:
        _repo = InMemoryIncidentRepository()

    embedder = get_embedder(_config.embedder_backend)

    from cortexguard.cloud.retrieval.vector_store import VectorStoreProtocol

    _vector_store: VectorStoreProtocol
    if _config.vector_store_backend == "qdrant":
        qdrant_store = QdrantVectorStore(_config.qdrant_url)
        await qdrant_store.initialize()
        _vector_store = qdrant_store
    else:
        _vector_store = InMemoryVectorStore()

    _retrieval_store = RetrievalStore(
        embedder,
        _vector_store,
        _repo,
        outcome_boost=_config.cloud_retrieval_outcome_boost,
        failure_penalty=_config.cloud_retrieval_failure_penalty,
    )

    llm_client = get_llm_client(_config.llm_backend, api_key=_config.anthropic_api_key)
    throttled_llm = LLMThrottler(llm_client, _config)

    adapter = CapabilityAdapter.load_default()
    _validator = PlanValidator(adapter, min_confidence=_config.min_confidence)

    _orchestrator = CloudOrchestrator(
        repo=_repo,
        retrieval_store=_retrieval_store,
        llm_client=throttled_llm,
        validator=_validator,
    )

    _explain_client = get_explain_client(_config)

    logger.info("CortexGuard MCP server starting")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
