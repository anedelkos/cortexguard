from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortexguard.cloud.api.outcomes import get_outcomes_router
from cortexguard.cloud.schemas.responses import ExecutionOutcomeStatus


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(get_outcomes_router(), prefix="/api/v1")
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
