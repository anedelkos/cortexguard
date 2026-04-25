"""Unit tests for Phase 1 inbound API rate limiting."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from cortexguard.cloud.api.mayday import get_mayday_router
from cortexguard.cloud.api.outcomes import get_outcomes_router
from cortexguard.cloud.orchestrator import PlanningResult
from cortexguard.cloud.persistence.repository import InMemoryIncidentRepository
from cortexguard.edge.models.mayday_packet import MaydayPacket


def _make_packet(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "trace_id": str(uuid.uuid4()),
        "device_id": "robot-arm-01",
        "timestamp": datetime.now(UTC).isoformat(),
        "health": {"cpu_load_pct": 10.0, "net_rtt_ms": 20, "packet_loss_pct": 0.0},
    }
    base.update(overrides)
    return base


class _StubOrchestrator:
    async def submit(self, packet: MaydayPacket) -> str:
        return packet.trace_id

    async def get_result(self, trace_id: str) -> PlanningResult | Literal["pending"] | None:
        return "pending"


def _make_rate_limited_app(
    mayday_rate_limit: str = "2/minute",
    result_rate_limit: str = "2/minute",
    outcome_rate_limit: str = "2/minute",
) -> FastAPI:
    """Build a minimal app mirroring runtime wiring; fresh Limiter per call."""
    fresh_limiter = Limiter(key_func=get_remote_address)
    app = FastAPI()
    app.state.limiter = fresh_limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_handler(request: Request, exc: RateLimitExceeded) -> Response:
        return Response(
            content='{"error": "rate limited"}',
            status_code=429,
            media_type="application/json",
        )

    orch = _StubOrchestrator()
    repo = InMemoryIncidentRepository()

    app.include_router(
        get_mayday_router(
            orchestrator=orch,
            mayday_rate_limit=mayday_rate_limit,
            result_rate_limit=result_rate_limit,
            _limiter=fresh_limiter,
        ),
        prefix="/api/v1",
    )
    app.include_router(
        get_outcomes_router(
            repo=repo,
            outcome_rate_limit=outcome_rate_limit,
            _limiter=fresh_limiter,
        ),
        prefix="/api/v1",
    )
    return app


class TestMaydayRateLimit:
    def test_submit_within_limit_succeeds(self) -> None:
        app = _make_rate_limited_app(mayday_rate_limit="10/minute")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/mayday", json=_make_packet())
        assert resp.status_code == 202

    def test_submit_exceeds_limit_returns_429(self) -> None:
        app = _make_rate_limited_app(mayday_rate_limit="1/minute")
        client = TestClient(app, raise_server_exceptions=False)

        resp1 = client.post("/api/v1/mayday", json=_make_packet())
        assert resp1.status_code == 202

        resp2 = client.post("/api/v1/mayday", json=_make_packet())
        assert resp2.status_code == 429

    def test_result_exceeds_limit_returns_429(self) -> None:
        tid = str(uuid.uuid4())
        app = _make_rate_limited_app(result_rate_limit="1/minute")
        client = TestClient(app, raise_server_exceptions=False)

        resp1 = client.get(f"/api/v1/mayday/{tid}/result")
        assert resp1.status_code in (202, 404)

        resp2 = client.get(f"/api/v1/mayday/{tid}/result")
        assert resp2.status_code == 429


class TestOutcomeRateLimit:
    def test_outcome_within_limit_succeeds(self) -> None:
        app = _make_rate_limited_app(outcome_rate_limit="10/minute")
        client = TestClient(app, raise_server_exceptions=False)
        payload = {
            "escalation_id": str(uuid.uuid4()),
            "decision_id": str(uuid.uuid4()),
            "device_id": "robot-arm-01",
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
        }
        resp = client.post("/api/v1/outcomes", json=payload)
        assert resp.status_code == 200

    def test_outcome_exceeds_limit_returns_429(self) -> None:
        app = _make_rate_limited_app(outcome_rate_limit="1/minute")
        client = TestClient(app, raise_server_exceptions=False)
        payload = {
            "escalation_id": str(uuid.uuid4()),
            "decision_id": str(uuid.uuid4()),
            "device_id": "robot-arm-01",
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
        }

        resp1 = client.post("/api/v1/outcomes", json=payload)
        assert resp1.status_code == 200

        resp2 = client.post("/api/v1/outcomes", json=payload)
        assert resp2.status_code == 429
