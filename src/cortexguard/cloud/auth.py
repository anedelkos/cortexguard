"""Shared-secret API key authentication for the cloud planner service.

When ``CLOUD_API_KEY`` is set, all requests must carry the header:

    X-CortexGuard-Key: <key>

Exempt paths (no auth required regardless of config):
- ``/healthz/*`` — liveness and readiness probes
- ``/metrics``   — Prometheus scrape endpoint

When ``CLOUD_API_KEY`` is unset the middleware is a no-op, which is correct
for local development where the service is not publicly reachable.
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_EXEMPT_PREFIXES = ("/healthz/", "/metrics")
_AUTH_HEADER = "X-CortexGuard-Key"


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests missing a valid ``X-CortexGuard-Key`` header.

    Instantiate with ``api_key=None`` to disable (local / no-auth deployments).
    """

    def __init__(self, app: object, api_key: str | None) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self._api_key is None:
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(p) for p in _EXEMPT_PREFIXES) or path == "/metrics":
            return await call_next(request)

        provided = request.headers.get(_AUTH_HEADER, "")
        if not hmac.compare_digest(provided, self._api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)
