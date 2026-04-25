"""Integration test: repeated POST /mayday calls hit 429 when limit is exceeded."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from cortexguard.cloud.api.mayday import get_mayday_router
from cortexguard.cloud.orchestrator import PlanningResult
from cortexguard.edge.models.mayday_packet import MaydayPacket


class _StubOrchestrator:
    async def submit(self, packet: MaydayPacket) -> str:
        return packet.trace_id

    async def get_result(self, trace_id: str) -> PlanningResult | Literal["pending"] | None:
        return "pending"


def _build_app(rate_limit: str) -> FastAPI:
    fresh_limiter = Limiter(key_func=get_remote_address)
    app = FastAPI()
    app.state.limiter = fresh_limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _handler(request: Request, exc: RateLimitExceeded) -> Response:
        return Response(
            content='{"error": "rate limited"}',
            status_code=429,
            media_type="application/json",
        )

    app.include_router(
        get_mayday_router(
            orchestrator=_StubOrchestrator(),
            mayday_rate_limit=rate_limit,
            _limiter=fresh_limiter,
        ),
        prefix="/api/v1",
    )
    return app


def _make_packet() -> dict[str, object]:
    return {
        "trace_id": str(uuid.uuid4()),
        "device_id": "robot-arm-01",
        "timestamp": datetime.now(UTC).isoformat(),
        "health": {"cpu_load_pct": 10.0, "net_rtt_ms": 20, "packet_loss_pct": 0.0},
    }


@pytest.mark.integration
def test_repeated_mayday_posts_hit_429() -> None:
    """Three rapid POST /mayday calls with limit=2/minute should yield one 429."""
    app = _build_app(rate_limit="2/minute")
    responses: list[int] = []

    with TestClient(app, raise_server_exceptions=False) as client:
        for _ in range(3):
            resp = client.post("/api/v1/mayday", json=_make_packet())
            responses.append(resp.status_code)

    assert 429 in responses, f"Expected at least one 429, got: {responses}"
    assert responses.count(202) == 2, f"Expected two 202s, got: {responses}"
