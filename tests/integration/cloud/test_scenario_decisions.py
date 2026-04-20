from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortexguard.cloud.api.health import get_health_router
from cortexguard.cloud.api.mayday import get_mayday_router
from cortexguard.cloud.orchestrator import CloudOrchestrator
from cortexguard.cloud.persistence.repository import InMemoryIncidentRepository
from cortexguard.cloud.planner.llm_client import PlannerResponse
from cortexguard.cloud.planner.mock_client import MockLLMClient
from cortexguard.cloud.retrieval.embedder import MockEmbedder
from cortexguard.cloud.retrieval.store import RetrievalStore
from cortexguard.cloud.retrieval.vector_store import InMemoryVectorStore
from cortexguard.cloud.validation.capability_adapter import CapabilityAdapter
from cortexguard.cloud.validation.plan_validator import PlanValidator
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth
from cortexguard.edge.models.plan import Plan, PlanSource, PlanStep, PlanType


def _make_packet(
    anomaly_key: str, severity: AnomalySeverity = AnomalySeverity.HIGH
) -> MaydayPacket:
    return MaydayPacket(
        trace_id=str(uuid.uuid4()),
        device_id="robot-arm-01",
        timestamp=datetime.now(UTC),
        health=SystemHealth(),
        anomalies=[
            AnomalyEvent(
                id=str(uuid.uuid4()),
                key=anomaly_key,
                timestamp=datetime.now(UTC),
                severity=severity,
                score=0.9,
                contributing_detectors=["LogicalRuleDetector"],
            )
        ],
    )


def _invalid_plan() -> Plan:
    return Plan(
        plan_id=str(uuid.uuid4()),
        context=GoalContext(
            goal_id=str(uuid.uuid4()),
            user_prompt="test",
            intent="test",
        ),
        plan_type=PlanType.REMEDIATION,
        source=PlanSource.CLOUD_AGENT,
        steps=[
            PlanStep(
                description="unknown action",
                action=AgentToolCall(action_name="NONEXISTENT_CAPABILITY", arguments={}),
            )
        ],
    )


def _make_client(llm_client: MockLLMClient) -> TestClient:
    repo = InMemoryIncidentRepository()
    retrieval_store = RetrievalStore(MockEmbedder(), InMemoryVectorStore())
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
    return TestClient(app, raise_server_exceptions=True)


def _poll_result(client: TestClient, trace_id: str, attempts: int = 30) -> dict[str, object]:
    for _ in range(attempts):
        resp = client.get(f"/api/v1/mayday/{trace_id}/result")
        if resp.status_code == 200:
            return cast(dict[str, object], resp.json())
        time.sleep(0.1)
    pytest.fail(f"Timed out waiting for result for trace_id={trace_id}")


@pytest.mark.integration
class TestScenarioDecisions:
    def test_s1_1_misgrasp_returns_plan_ready(self) -> None:
        packet = _make_packet("S1.1_MISGRASP")
        with _make_client(MockLLMClient()) as client:
            resp = client.post(
                "/api/v1/mayday",
                content=packet.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 202
            result = _poll_result(client, packet.trace_id)
        assert "plan_id" in result, f"Expected plan_ready, got: {result}"

    def test_s1_2_vision_occlusion_returns_plan_ready(self) -> None:
        packet = _make_packet("S1.2_VISION_OCCLUSION", severity=AnomalySeverity.MEDIUM)
        with _make_client(MockLLMClient()) as client:
            resp = client.post(
                "/api/v1/mayday",
                content=packet.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 202
            result = _poll_result(client, packet.trace_id)
        assert "plan_id" in result, f"Expected plan_ready, got: {result}"

    def test_s2_3_sensor_freeze_returns_plan_ready(self) -> None:
        packet = _make_packet("S2.3_SENSOR_FREEZE", severity=AnomalySeverity.MEDIUM)
        with _make_client(MockLLMClient()) as client:
            resp = client.post(
                "/api/v1/mayday",
                content=packet.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 202
            result = _poll_result(client, packet.trace_id)
        assert "plan_id" in result, f"Expected plan_ready, got: {result}"

    def test_s3_0_unknown_fault_returns_plan_ready(self) -> None:
        packet = _make_packet("S3.0_UNKNOWN_FAULT")
        with _make_client(MockLLMClient()) as client:
            resp = client.post(
                "/api/v1/mayday",
                content=packet.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 202
            result = _poll_result(client, packet.trace_id)
        assert "plan_id" in result, f"Expected plan_ready, got: {result}"

    def test_invalid_capability_returns_needs_human(self) -> None:
        fixed = PlannerResponse(
            candidate_plan=_invalid_plan(),
            confidence=0.8,
            needs_human_review=False,
            rationale="bad plan",
            raw_provider_metadata={},
        )
        packet = _make_packet("S1.1_MISGRASP")
        with _make_client(MockLLMClient(fixed_response=fixed)) as client:
            resp = client.post(
                "/api/v1/mayday",
                content=packet.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 202
            result = _poll_result(client, packet.trace_id)
        assert result.get("decision") == "needs_human", f"Expected needs_human, got: {result}"
