from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortexguard.edge.api import health
from cortexguard.edge.api.health import get_health_router


def _make_client(subsystems_healthy: dict[str, bool]) -> TestClient:
    async def mock_health_check() -> dict[str, bool]:
        return subsystems_healthy

    app = FastAPI()
    app.include_router(health.router)
    app.include_router(get_health_router(mock_health_check))
    return TestClient(app)


@pytest.fixture
def healthy_client() -> TestClient:
    return _make_client({"runtime": True, "blackboard": True, "orchestrator": True})


@pytest.fixture
def degraded_client() -> TestClient:
    return _make_client({"runtime": True, "blackboard": False, "orchestrator": True})


def test_liveness_returns_200(healthy_client: TestClient) -> None:
    response = healthy_client.get("/healthz/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_returns_200_when_all_healthy(healthy_client: TestClient) -> None:
    response = healthy_client.get("/healthz/ready")
    assert response.status_code == 200
    assert all(response.json().values())


def test_readiness_returns_503_when_subsystem_down(degraded_client: TestClient) -> None:
    response = degraded_client.get("/healthz/ready")
    assert response.status_code == 503
    assert response.json()["blackboard"] is False


def test_liveness_unaffected_when_subsystem_down(degraded_client: TestClient) -> None:
    """Liveness must stay 200 even when readiness is degraded."""
    response = degraded_client.get("/healthz/live")
    assert response.status_code == 200


def test_legacy_health_endpoint(healthy_client: TestClient) -> None:
    response = healthy_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
