from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortexguard.cloud.api.outcomes import get_outcomes_router
from cortexguard.cloud.config import CloudConfig
from cortexguard.cloud.persistence.repository import InMemoryIncidentRepository
from cortexguard.cloud.runtime import create_cloud_app
from cortexguard.cloud.schemas.responses import ExecutionOutcomeStatus


def _make_app(repo: InMemoryIncidentRepository | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(get_outcomes_router(repo=repo), prefix="/api/v1")
    return app


def _make_outcome(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "escalation_id": str(uuid.uuid4()),
        "decision_id": str(uuid.uuid4()),
        "device_id": "robot-arm-01",
        "status": ExecutionOutcomeStatus.completed,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    base.update(overrides)
    return base


class TestRecordOutcome:
    def test_valid_outcome_returns_200(self) -> None:
        client = TestClient(_make_app())
        payload = _make_outcome()
        response = client.post("/api/v1/outcomes", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["escalation_id"] == payload["escalation_id"]

    @pytest.mark.parametrize(
        "missing_field",
        ["escalation_id", "decision_id", "device_id", "status", "completed_at"],
    )
    def test_missing_required_field_returns_422(self, missing_field: str) -> None:
        client = TestClient(_make_app())
        payload = _make_outcome()
        del payload[missing_field]
        response = client.post("/api/v1/outcomes", json=payload)
        assert response.status_code == 422


class TestListRecentOutcomes:
    def test_returns_recent_outcomes_newest_first(self) -> None:
        repo = InMemoryIncidentRepository()
        client = TestClient(_make_app(repo=repo))

        older = _make_outcome(status=ExecutionOutcomeStatus.failed, notes="older")
        newer = _make_outcome(status=ExecutionOutcomeStatus.completed, notes="newer")
        client.post("/api/v1/outcomes", json=older)
        client.post("/api/v1/outcomes", json=newer)

        response = client.get("/api/v1/outcomes/recent")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 2
        assert body[0]["escalation_id"] == newer["escalation_id"]
        assert body[0]["notes"] == "newer"
        assert body[1]["escalation_id"] == older["escalation_id"]

    def test_limit_query_param_is_applied(self) -> None:
        repo = InMemoryIncidentRepository()
        client = TestClient(_make_app(repo=repo))

        first = _make_outcome(notes="first")
        second = _make_outcome(notes="second")
        client.post("/api/v1/outcomes", json=first)
        client.post("/api/v1/outcomes", json=second)

        response = client.get("/api/v1/outcomes/recent", params={"limit": 1})
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["notes"] == "second"


class TestMcpEventEndpoint:
    def _app(self) -> FastAPI:
        return create_cloud_app(CloudConfig())

    def test_tool_event_returns_ok(self) -> None:
        client = TestClient(self._app())
        response = client.post("/internal/mcp-event", json={"tool": "validate_plan"})
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_outcome_event_returns_ok(self) -> None:
        client = TestClient(self._app())
        response = client.post("/internal/mcp-event", json={"outcome": "resolved"})
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_malformed_body_is_silent(self) -> None:
        client = TestClient(self._app())
        response = client.post(
            "/internal/mcp-event",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200
