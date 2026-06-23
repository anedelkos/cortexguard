"""Buffered telemetry client that sends step outcomes to the cloud API."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import httpx

from cortexguard.edge.models.fusion_snapshot import FusionSnapshot
from cortexguard.edge.utils.metrics import telemetry_flush_failures_total

logger = logging.getLogger(__name__)


@runtime_checkable
class TelemetryTransport(Protocol):
    async def post(self, url: str, **kwargs: Any) -> Any: ...
    async def aclose(self) -> None: ...


class TelemetryClient:
    def __init__(
        self,
        cloud_api_url: str,
        cloud_api_key: str | None = None,
        batch_size: int = 100,
        flush_interval_s: float = 60.0,
        timeout_s: float = 10.0,
        http_client: TelemetryTransport | None = None,
    ) -> None:
        self._url = f"{cloud_api_url.rstrip('/')}/api/v1/telemetry"
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._timeout_s = timeout_s
        self._headers: dict[str, str] = {}
        if cloud_api_key is not None:
            self._headers["X-CortexGuard-Key"] = cloud_api_key
        self._client = http_client or httpx.AsyncClient()
        self._owns_client = http_client is None
        self._buffer: list[dict[str, Any]] = []
        self._task: asyncio.Task[None] | None = None

    async def send(self, device_id: str, key: str, outcome: str, snapshot: FusionSnapshot) -> None:
        self._buffer.append(
            {
                "device_id": device_id,
                "key": key,
                "outcome": outcome,
                "timestamp": datetime.now(UTC).isoformat(),
                "sensor_snapshot": snapshot.model_dump(mode="json"),
            }
        )
        if len(self._buffer) >= self._batch_size:
            await self._flush()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._periodic_flush())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._buffer:
            await self._flush()
        if self._owns_client:
            await self._client.aclose()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer
        self._buffer = []
        try:
            await self._client.post(
                self._url,
                headers=self._headers,
                json={"records": batch},
                timeout=self._timeout_s,
            )
        except Exception:
            telemetry_flush_failures_total.inc()
            logger.warning("Telemetry flush failed, %d records dropped", len(batch), exc_info=True)

    async def _periodic_flush(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._flush_interval_s)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unexpected error in periodic telemetry flush")
