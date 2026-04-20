"""Mayday submission and result-polling API routes for the cloud planner."""

from __future__ import annotations

import uuid
from typing import Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from cortexguard.cloud.orchestrator import PlanningResult
from cortexguard.cloud.schemas.responses import MaydayAcceptedResponse
from cortexguard.edge.models.mayday_packet import MaydayPacket


class CloudOrchestratorProtocol(Protocol):
    async def submit(self, packet: MaydayPacket) -> str: ...

    async def get_result(self, trace_id: str) -> PlanningResult | Literal["pending"] | None: ...


class _StubOrchestrator:
    async def submit(self, packet: MaydayPacket) -> str:
        return str(uuid.uuid4())

    async def get_result(self, trace_id: str) -> PlanningResult | Literal["pending"] | None:
        return "pending"


def get_mayday_router(
    orchestrator: CloudOrchestratorProtocol | None = None,
) -> APIRouter:
    _orchestrator = orchestrator if orchestrator is not None else _StubOrchestrator()
    router = APIRouter()

    def _get_orchestrator() -> CloudOrchestratorProtocol:
        return _orchestrator

    @router.post(
        "/mayday",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=MaydayAcceptedResponse,
    )
    async def submit_mayday(
        packet: MaydayPacket,
        orch: CloudOrchestratorProtocol = Depends(_get_orchestrator),  # noqa: B008
    ) -> MaydayAcceptedResponse:
        trace_id = await orch.submit(packet)
        return MaydayAcceptedResponse(trace_id=trace_id)

    @router.get("/mayday/{trace_id}/result")
    async def get_mayday_result(
        trace_id: str,
        orch: CloudOrchestratorProtocol = Depends(_get_orchestrator),  # noqa: B008
    ) -> JSONResponse:
        result = await orch.get_result(trace_id)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trace_id not found")
        if result == "pending":
            return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"status": "pending"})
        planning_result: PlanningResult = result
        if planning_result.decision == "plan_ready" and planning_result.plan is not None:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=planning_result.plan.model_dump(mode="json"),
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"decision": planning_result.decision, "plan": None},
        )

    return router
