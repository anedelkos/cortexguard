from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from cortexguard.cloud.graph.nodes import make_generate_candidate_plan_node
from cortexguard.cloud.graph.state import CloudPlanningState
from cortexguard.cloud.planner.llm_client import PlannerRequest, PlannerResponse
from cortexguard.cloud.planner.mock_client import MockLLMClient
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth


def _make_packet() -> MaydayPacket:
    return MaydayPacket(
        trace_id=str(uuid.uuid4()),
        device_id="robot-arm-01",
        timestamp=datetime.now(UTC),
        health=SystemHealth(),
        anomalies=[
            AnomalyEvent(
                id=str(uuid.uuid4()),
                key="S1.1_MISGRASP",
                timestamp=datetime.now(UTC),
                severity=AnomalySeverity.HIGH,
                score=0.9,
                contributing_detectors=["LogicalRuleDetector"],
            )
        ],
    )


def _make_state(packet: MaydayPacket) -> CloudPlanningState:
    return CloudPlanningState(
        request=packet,
        incident_id="test-incident-id",
        retrieved_incidents=[],
        retrieved_incident_records=[],
        candidate_plan=None,
        validation_result=None,
        decision=None,
        rationale=None,
        confidence=None,
        needs_human_review=False,
        errors=[],
    )


@pytest.mark.asyncio
async def test_generate_node_populates_candidate_plan_and_confidence() -> None:
    client = MockLLMClient()
    node = make_generate_candidate_plan_node(client)
    state = _make_state(_make_packet())
    result = await node(state)
    assert result["candidate_plan"] is not None
    assert isinstance(result["confidence"], float)
    assert result["rationale"] is not None


@pytest.mark.asyncio
async def test_generate_node_none_client_returns_stub() -> None:
    node = make_generate_candidate_plan_node(None)
    state = _make_state(_make_packet())
    result = await node(state)
    assert result["candidate_plan"] is None
    assert result["confidence"] == 0.5


@pytest.mark.asyncio
async def test_generate_node_failure_sets_errors_and_null_plan() -> None:
    class _BrokenClient:
        async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
            raise RuntimeError("LLM unavailable")

    node = make_generate_candidate_plan_node(_BrokenClient())  # type: ignore[arg-type]
    state = _make_state(_make_packet())
    result = await node(state)
    assert result["candidate_plan"] is None
    assert len(result["errors"]) > 0
    assert result["decision"] == "needs_human"


@pytest.mark.asyncio
async def test_generate_node_llm_throttle_error_routes_to_needs_human_without_errors() -> None:
    """LLMThrottleError must route to needs_human with an empty errors list (not a generic crash)."""
    from cortexguard.cloud.planner.throttler import LLMThrottleError

    class _ThrottledClient:
        async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
            raise LLMThrottleError("timeout")

    node = make_generate_candidate_plan_node(_ThrottledClient())  # type: ignore[arg-type]
    state = _make_state(_make_packet())
    result = await node(state)
    assert result["decision"] == "needs_human"
    assert result["candidate_plan"] is None
    # The throttle path must NOT populate errors — it is a known, expected condition
    # distinct from the generic exception fallback which does populate errors.
    assert result.get("errors", []) == []
