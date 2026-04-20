from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cortexguard.cloud.graph.workflow import build_graph, run_planning_workflow
from cortexguard.cloud.persistence.repository import InMemoryIncidentRepository
from cortexguard.cloud.retrieval.vector_store import SearchResult
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth


def _make_packet() -> MaydayPacket:
    return MaydayPacket(
        trace_id="trace-001",
        device_id="robot-01",
        timestamp=datetime.now(UTC),
        health=SystemHealth(),
    )


@pytest.mark.asyncio
async def test_run_planning_workflow_returns_incident_id() -> None:
    repo = InMemoryIncidentRepository()
    graph = build_graph(repo, None, None, None)
    final_state = await run_planning_workflow(_make_packet(), graph)

    assert final_state["incident_id"] is not None


@pytest.mark.asyncio
async def test_run_planning_workflow_decision_is_valid() -> None:
    repo = InMemoryIncidentRepository()
    graph = build_graph(repo, None, None, None)
    final_state = await run_planning_workflow(_make_packet(), graph)

    assert final_state["decision"] in {"plan_ready", "needs_human", "no_safe_plan"}


@pytest.mark.asyncio
async def test_run_planning_workflow_errors_is_list() -> None:
    repo = InMemoryIncidentRepository()
    graph = build_graph(repo, None, None, None)
    final_state = await run_planning_workflow(_make_packet(), graph)

    assert isinstance(final_state["errors"], list)


class _BrokenRetrievalStore:
    async def retrieve_similar(self, packet: MaydayPacket, top_k: int = 5) -> list[SearchResult]:
        raise RuntimeError("store unavailable")


@pytest.mark.asyncio
async def test_node_exception_results_in_needs_human() -> None:
    repo = InMemoryIncidentRepository()
    broken_store = _BrokenRetrievalStore()
    graph = build_graph(repo, broken_store, None, None)

    final_state = await run_planning_workflow(_make_packet(), graph)

    assert final_state["decision"] == "needs_human"
    assert len(final_state["errors"]) > 0
