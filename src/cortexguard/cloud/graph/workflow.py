"""Assembly and execution helpers for the cloud LangGraph planning workflow."""

from __future__ import annotations

import functools
import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from cortexguard.cloud.graph.nodes import (
    make_generate_candidate_plan_node,
    make_persist_incident_node,
    make_retrieve_similar_incidents_node,
    make_validate_candidate_plan_node,
    pause_for_operator,
    route_after_decision,
    route_decision,
)
from cortexguard.cloud.graph.state import CloudPlanningState
from cortexguard.cloud.persistence.repository import IncidentRepositoryProtocol
from cortexguard.cloud.planner.llm_client import LLMClientProtocol
from cortexguard.cloud.retrieval.store import RetrievalStoreProtocol
from cortexguard.cloud.validation.plan_validator import PlanValidatorProtocol
from cortexguard.edge.models.mayday_packet import MaydayPacket

logger = logging.getLogger(__name__)

_AnyGraph = Any


@functools.cache
def _load_capability_catalog() -> str:
    try:
        from cortexguard.edge.models.capability_registry import CapabilityRegistry

        registry = CapabilityRegistry.load_from_yaml()
        return registry.get_llm_tool_catalog()
    except Exception:
        logger.warning("Failed to load capability registry; using empty catalog")
        return "[]"


async def create_checkpointer(store: str, db_path: str | None = None) -> BaseCheckpointSaver[Any]:
    """Async factory for checkpoint backends.

    ``"sqlite"`` creates an :class:`langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`.
    ``"postgres"`` creates a :class:`langgraph.checkpoint.postgres.PostgresSaver`.
    All other values (including ``"memory"``) return a new ``MemorySaver``.
    """
    if store == "sqlite":
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            conn = await aiosqlite.connect(db_path or "checkpoints.db")
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            return saver
        except ImportError:
            logger.warning("AsyncSqliteSaver not available, falling back to MemorySaver")
            return MemorySaver()
    if store == "postgres":
        try:
            import asyncpg  # type: ignore[import-untyped]
            from langgraph.checkpoint.postgres import PostgresSaver

            pool = await asyncpg.create_pool(db_path or "postgresql:///cortexguard")
            return PostgresSaver(pool)
        except ImportError:
            logger.warning("PostgresSaver not available, falling back to MemorySaver")
            return MemorySaver()
    return MemorySaver()


def build_graph(
    repo: IncidentRepositoryProtocol,
    retrieval_store: RetrievalStoreProtocol | None,
    llm_client: LLMClientProtocol | None,
    validator: PlanValidatorProtocol | None,
    checkpoint_store: str | None = None,
    checkpoint_db_path: str | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> _AnyGraph:
    graph: Any = StateGraph(CloudPlanningState)

    graph.add_node("persist_incident", make_persist_incident_node(repo))
    graph.add_node(
        "retrieve_similar_incidents",
        make_retrieve_similar_incidents_node(retrieval_store, repo),
    )
    capability_catalog_json = _load_capability_catalog()
    graph.add_node(
        "generate_candidate_plan",
        make_generate_candidate_plan_node(llm_client, capability_catalog_json),
    )
    graph.add_node("validate_candidate_plan", make_validate_candidate_plan_node(validator))
    graph.add_node("route_decision", route_decision)
    graph.add_node("pause_for_operator", pause_for_operator)

    graph.add_edge("persist_incident", "retrieve_similar_incidents")
    graph.add_edge("retrieve_similar_incidents", "generate_candidate_plan")
    graph.add_edge("generate_candidate_plan", "validate_candidate_plan")
    graph.add_edge("validate_candidate_plan", "route_decision")

    graph.add_conditional_edges(
        "route_decision",
        route_after_decision,
    )

    graph.set_entry_point("persist_incident")

    compiled: Any
    if checkpointer is not None:
        compiled = graph.compile(checkpointer=checkpointer)
    elif checkpoint_store:
        if checkpoint_store == "memory":
            compiled = graph.compile(checkpointer=MemorySaver())
        else:
            logger.warning(
                "checkpoint_store=%s requires a pre-built checkpointer, call "
                "create_checkpointer() and pass via checkpointer= kwarg. "
                "Falling back to MemorySaver.",
                checkpoint_store,
            )
            compiled = graph.compile(checkpointer=MemorySaver())
    else:
        compiled = graph.compile()
    return compiled


async def run_planning_workflow(
    packet: MaydayPacket,
    graph: _AnyGraph,
    thread_id: str | None = None,
) -> CloudPlanningState:
    initial_state: CloudPlanningState = {
        "request": packet,
        "incident_id": None,
        "retrieved_incidents": [],
        "retrieved_incident_records": [],
        "candidate_plan": None,
        "validation_result": None,
        "decision": None,
        "rationale": None,
        "confidence": None,
        "needs_human_review": False,
        "errors": [],
        "operator_response": None,
    }
    config: dict[str, Any] | None = None
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}
    result: CloudPlanningState = await graph.ainvoke(initial_state, config)
    return result
