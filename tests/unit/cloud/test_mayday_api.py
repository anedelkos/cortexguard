from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortexguard.cloud.api.mayday import CloudOrchestratorProtocol, get_mayday_router
from cortexguard.cloud.orchestrator import PlanningResult
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.mayday_packet import MaydayPacket
from cortexguard.edge.models.plan import Plan, PlanStep, PlanType


def _make_packet(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "trace_id": str(uuid.uuid4()),
        "device_id": "robot-arm-01",
        "timestamp": datetime.now(UTC).isoformat(),
        "health": {"cpu_load_pct": 10.0, "net_rtt_ms": 20, "packet_loss_pct": 0.0},
    }
    base.update(overrides)
    return base


def _make_plan() -> Plan:
    goal = GoalContext(
        goal_id=str(uuid.uuid4()),
        user_prompt="recover from anomaly",
        intent="Remediate detected anomaly",
    )
    step = PlanStep(
        description="Reset device",
        action=AgentToolCall(action_name="RESET_DEVICE"),
    )
    return Plan(context=goal, plan_type=PlanType.REMEDIATION, steps=[step])


class _FixedOrchestrator:
    def __init__(
        self,
        fixed_trace_id: str,
        result: PlanningResult | Literal["pending"] | None,
    ) -> None:
        self._trace_id = fixed_trace_id
        self._result = result

    async def submit(self, packet: MaydayPacket) -> str:
        return self._trace_id

    async def get_result(self, trace_id: str) -> PlanningResult | Literal["pending"] | None:
        if trace_id == self._trace_id:
            return self._result
        return None


def _make_app(orchestrator: CloudOrchestratorProtocol) -> FastAPI:
    app = FastAPI()
    app.include_router(get_mayday_router(orchestrator=orchestrator), prefix="/api/v1")
    return app


class TestSubmitMayday:
    def test_returns_202_and_trace_id(self) -> None:
        fixed_id = str(uuid.uuid4())
        orch = _FixedOrchestrator(fixed_trace_id=fixed_id, result="pending")
        client = TestClient(_make_app(orch))

        response = client.post("/api/v1/mayday", json=_make_packet())
        assert response.status_code == 202
        body = response.json()
        assert body["trace_id"] == fixed_id

    def test_invalid_body_returns_422(self) -> None:
        orch = _FixedOrchestrator(fixed_trace_id="x", result="pending")
        client = TestClient(_make_app(orch))

        response = client.post("/api/v1/mayday", json={"broken": True})
        assert response.status_code == 422


class TestGetMaydayResult:
    def test_unknown_trace_id_returns_404(self) -> None:
        orch = _FixedOrchestrator(fixed_trace_id="known-id", result="pending")
        client = TestClient(_make_app(orch))

        response = client.get("/api/v1/mayday/completely-unknown/result")
        assert response.status_code == 404

    def test_pending_result_returns_202(self) -> None:
        fixed_id = str(uuid.uuid4())
        orch = _FixedOrchestrator(fixed_trace_id=fixed_id, result="pending")
        client = TestClient(_make_app(orch))

        response = client.get(f"/api/v1/mayday/{fixed_id}/result")
        assert response.status_code == 202
        assert response.json()["status"] == "pending"

    def test_completed_plan_returns_200_with_plan_body(self) -> None:
        fixed_id = str(uuid.uuid4())
        plan = _make_plan()
        planning_result = PlanningResult(decision="plan_ready", plan=plan)
        orch = _FixedOrchestrator(fixed_trace_id=fixed_id, result=planning_result)
        client = TestClient(_make_app(orch))

        response = client.get(f"/api/v1/mayday/{fixed_id}/result")
        assert response.status_code == 200
        body = response.json()
        assert body["plan_id"] == plan.plan_id
        assert body["plan_type"] == PlanType.REMEDIATION.value
        assert len(body["steps"]) == 1
        assert body["steps"][0]["action"]["action_name"] == "RESET_DEVICE"

    def test_completed_no_plan_returns_200_with_decision(self) -> None:
        fixed_id = str(uuid.uuid4())
        planning_result = PlanningResult(decision="no_safe_plan", plan=None)
        orch = _FixedOrchestrator(fixed_trace_id=fixed_id, result=planning_result)
        client = TestClient(_make_app(orch))

        response = client.get(f"/api/v1/mayday/{fixed_id}/result")
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "no_safe_plan"
        assert body["plan"] is None
