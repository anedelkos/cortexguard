from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortexguard.cloud.api.health import get_health_router


def _make_app(checks: list[Callable[[], Awaitable[None]]] | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(get_health_router(readiness_checks=checks))
    return app


class TestLiveness:
    def test_returns_200(self) -> None:
        client = TestClient(_make_app())
        response = client.get("/healthz/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestReadiness:
    def test_returns_200_when_no_checks(self) -> None:
        client = TestClient(_make_app())
        response = client.get("/healthz/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_returns_200_when_all_checks_pass(self) -> None:
        async def passing_check() -> None:
            pass

        client = TestClient(_make_app(checks=[passing_check]))
        response = client.get("/healthz/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_returns_503_when_check_raises(self) -> None:
        async def failing_check() -> None:
            raise RuntimeError("db unreachable")

        client = TestClient(_make_app(checks=[failing_check]))
        response = client.get("/healthz/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unavailable"
        assert "db unreachable" in body["detail"]

    def test_first_failing_check_short_circuits(self) -> None:
        calls: list[str] = []

        async def check_a() -> None:
            calls.append("a")
            raise RuntimeError("a failed")

        async def check_b() -> None:
            calls.append("b")

        client = TestClient(_make_app(checks=[check_a, check_b]))
        response = client.get("/healthz/ready")
        assert response.status_code == 503
        assert "b" not in calls
