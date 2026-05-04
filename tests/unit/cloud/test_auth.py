"""Unit tests for ApiKeyMiddleware."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortexguard.cloud.auth import ApiKeyMiddleware


def _make_app(api_key: str | None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=api_key)

    @app.get("/api/v1/test")
    def _test() -> dict[str, str]:
        return {"ok": "true"}

    @app.get("/healthz/live")
    def _live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/metrics")
    def _metrics() -> str:
        return ""

    return app


def test_no_api_key_configured_allows_all() -> None:
    client = TestClient(_make_app(api_key=None))
    assert client.get("/api/v1/test").status_code == 200


def test_valid_key_accepted() -> None:
    client = TestClient(_make_app(api_key="secret"))
    resp = client.get("/api/v1/test", headers={"X-CortexGuard-Key": "secret"})
    assert resp.status_code == 200


def test_wrong_key_rejected() -> None:
    client = TestClient(_make_app(api_key="secret"))
    resp = client.get("/api/v1/test", headers={"X-CortexGuard-Key": "wrong"})
    assert resp.status_code == 401


def test_missing_key_rejected() -> None:
    client = TestClient(_make_app(api_key="secret"))
    resp = client.get("/api/v1/test")
    assert resp.status_code == 401


def test_healthz_exempt_from_auth() -> None:
    client = TestClient(_make_app(api_key="secret"))
    resp = client.get("/healthz/live")
    assert resp.status_code == 200


def test_metrics_exempt_from_auth() -> None:
    client = TestClient(_make_app(api_key="secret"))
    resp = client.get("/metrics")
    assert resp.status_code == 200
