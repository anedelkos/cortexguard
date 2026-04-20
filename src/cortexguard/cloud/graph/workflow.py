"""Assembly and execution helpers for the cloud LangGraph planning workflow."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph

from cortexguard.cloud.graph.nodes import (
    make_generate_candidate_plan_node,
    make_persist_incident_node,
    make_retrieve_similar_incidents_node,
    make_validate_candidate_plan_node,
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


def _load_capability_catalog() -> str:
    try:
        from cortexguard.edge.models.capability_registry import CapabilityRegistry

        registry = CapabilityRegistry.load_from_yaml()
        return registry.get_llm_tool_catalog()
    except Exception:
        logger.warning("Failed to load capability registry; using empty catalog")
        return "[]"


def build_graph(
    repo: IncidentRepositoryProtocol,
    retrieval_store: RetrievalStoreProtocol | None,
    llm_client: LLMClientProtocol | None,
    validator: PlanValidatorProtocol | None,
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

    graph.add_edge("persist_incident", "retrieve_similar_incidents")
    graph.add_edge("retrieve_similar_incidents", "generate_candidate_plan")
    graph.add_edge("generate_candidate_plan", "validate_candidate_plan")
    graph.add_edge("validate_candidate_plan", "route_decision")

    graph.set_entry_point("persist_incident")

    compiled: Any = graph.compile()
    return compiled


async def run_planning_workflow(
    packet: MaydayPacket,
    graph: _AnyGraph,
) -> CloudPlanningState:
    initial_state: CloudPlanningState = {
        "request": packet,
        "incident_id": None,
        "retrieved_incidents": [],
        "candidate_plan": None,
        "validation_result": None,
        "decision": None,
        "rationale": None,
        "confidence": None,
        "needs_human_review": False,
        "errors": [],
    }
    result: CloudPlanningState = await graph.ainvoke(initial_state)
    return result
