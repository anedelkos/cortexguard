from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Response, status

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """Basic health check for container orchestration."""
    return {"status": "ok"}


def get_health_router(
    health_check_fn: Callable[[], Awaitable[dict[str, bool]]],
) -> APIRouter:
    """
    Factory that creates the /healthz liveness and readiness routers.
    Accepts the runtime health_check coroutine to avoid circular imports.
    """
    health_router = APIRouter()

    @health_router.get("/healthz/live", status_code=status.HTTP_200_OK)
    async def liveness() -> dict[str, str]:
        """Liveness probe — process is running."""
        return {"status": "alive"}

    @health_router.get("/healthz/ready")
    async def readiness(response: Response) -> dict[str, bool]:
        """Readiness probe — all subsystems are initialised and running."""
        health = await health_check_fn()
        if not all(health.values()):
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return health

    return health_router
