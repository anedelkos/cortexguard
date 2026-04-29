from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortexguard.cloud.api.health import get_health_router
from cortexguard.cloud.api.mayday import get_mayday_router
from cortexguard.cloud.api.outcomes import get_outcomes_router
from cortexguard.cloud.orchestrator import CloudOrchestrator
from cortexguard.cloud.persistence.repository import InMemoryIncidentRepository
from cortexguard.cloud.planner.mock_client import MockLLMClient
from cortexguard.cloud.retrieval.embedder import MockEmbedder
from cortexguard.cloud.retrieval.store import RetrievalStore
from cortexguard.cloud.retrieval.vector_store import InMemoryVectorStore
from cortexguard.cloud.validation.capability_adapter import CapabilityAdapter
from cortexguard.cloud.validation.plan_validator import PlanValidator
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth
from cortexguard.edge.models.plan import Plan


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


@pytest.mark.integration
def test_mayday_end_to_end_flow() -> None:
    repo = InMemoryIncidentRepository()
    vector_store = InMemoryVectorStore()
    embedder = MockEmbedder()
    retrieval_store = RetrievalStore(embedder=embedder, vector_store=vector_store)
    llm_client = MockLLMClient()
    validator = PlanValidator(CapabilityAdapter.load_default())
    orchestrator = CloudOrchestrator(
        repo=repo,
        retrieval_store=retrieval_store,
        llm_client=llm_client,
        validator=validator,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(get_health_router(readiness_checks=[]))
    app.include_router(get_mayday_router(orchestrator=orchestrator), prefix="/api/v1")
    app.include_router(get_outcomes_router(repo=repo), prefix="/api/v1")

    packet = _make_packet()

    with TestClient(app, raise_server_exceptions=True) as client:
        post_resp = client.post(
            "/api/v1/mayday",
            content=packet.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert post_resp.status_code == 202
        trace_id = post_resp.json()["trace_id"]
        assert trace_id == packet.trace_id

        result_data: dict[str, object] | None = None
        for _ in range(20):
            get_resp = client.get(f"/api/v1/mayday/{trace_id}/result")
            if get_resp.status_code == 200:
                result_data = get_resp.json()
                break
            time.sleep(0.1)

        assert result_data is not None, "Timed out waiting for plan result"
        assert "plan_id" in result_data, f"Expected plan_ready result, got: {result_data}"
        plan = Plan.model_validate(result_data)
        assert plan is not None

        outcome_resp = client.post(
            "/api/v1/outcomes",
            json={
                "escalation_id": trace_id,
                "decision_id": str(uuid.uuid4()),
                "device_id": "robot-arm-01",
                "status": "completed",
                "completed_at": datetime.now(UTC).isoformat(),
            },
        )
        assert outcome_resp.status_code == 200
        assert outcome_resp.json()["ok"] is True

        recent_resp = client.get("/api/v1/outcomes/recent")
        assert recent_resp.status_code == 200
        recent = recent_resp.json()
        assert len(recent) == 1
        assert recent[0]["escalation_id"] == trace_id
        assert recent[0]["status"] == "completed"
