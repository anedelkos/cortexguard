"""HTTP client used by the edge runtime to submit mayday escalations to the cloud."""

from __future__ import annotations

import asyncio
import logging

import httpx

from cortexguard.cloud.schemas.responses import MaydayAcceptedResponse
from cortexguard.edge.models.mayday_packet import MaydayPacket
from cortexguard.edge.models.plan import Plan

logger = logging.getLogger(__name__)


_AUTH_HEADER = "X-CortexGuard-Key"


class HttpCloudAgentClient:
    def __init__(
        self,
        cloud_base_url: str,
        poll_interval_s: float = 2.0,
        http_client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
    ) -> None:
        self._base_url = cloud_base_url.rstrip("/")
        self._poll_interval_s = poll_interval_s
        self._http_client = http_client
        self._api_key = api_key

    def _auth_headers(self) -> dict[str, str]:
        if self._api_key is not None:
            return {_AUTH_HEADER: self._api_key}
        return {}

    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        try:
            client = self._http_client or httpx.AsyncClient()
            async with client if self._http_client is None else _nullctx(client):
                post_resp = await client.post(
                    f"{self._base_url}/api/v1/mayday",
                    content=packet.model_dump_json(),
                    headers={"Content-Type": "application/json", **self._auth_headers()},
                )
                if post_resp.status_code not in (200, 202):
                    logger.warning("Mayday POST returned %d", post_resp.status_code)
                    return None

                accepted = MaydayAcceptedResponse.model_validate(post_resp.json())
                trace_id = accepted.trace_id

                while True:
                    get_resp = await client.get(
                        f"{self._base_url}/api/v1/mayday/{trace_id}/result",
                        headers=self._auth_headers(),
                    )
                    if get_resp.status_code == 200:
                        body = get_resp.json()
                        if isinstance(body, dict) and "plan_id" in body:
                            return Plan.model_validate(body)
                        return None  # no_safe_plan or needs_human
                    if get_resp.status_code == 404:
                        return None
                    if get_resp.status_code != 202:
                        logger.warning(
                            "Cloud poll returned unexpected status %d; aborting",
                            get_resp.status_code,
                        )
                        return None
                    await asyncio.sleep(self._poll_interval_s)
        except httpx.HTTPError:
            logger.warning("Cloud agent unreachable")
            return None


class _nullctx:
    def __init__(self, obj: httpx.AsyncClient) -> None:
        self._obj = obj

    async def __aenter__(self) -> httpx.AsyncClient:
        return self._obj

    async def __aexit__(self, *_args: object) -> None:
        pass
