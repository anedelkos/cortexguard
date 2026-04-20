"""Health and readiness routes for the cloud planner service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Response, status


def get_health_router(
    readiness_checks: list[Callable[[], Awaitable[None]]] | None = None,
) -> APIRouter:
    checks = readiness_checks or []
    router = APIRouter()

    @router.get("/healthz/live", status_code=status.HTTP_200_OK)
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/healthz/ready")
    async def readiness(response: Response) -> dict[str, str]:
        for check in checks:
            try:
                await check()
            except Exception as exc:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                return {"status": "unavailable", "detail": str(exc)}
        return {"status": "ok"}

    return router
