from __future__ import annotations

import asyncio

from cortexguard.core.interfaces.base_cloud_agent_client import BaseCloudAgentClient
from cortexguard.edge.models.mayday_packet import MaydayPacket
from cortexguard.edge.models.plan import Plan


class MockCloudAgentClient(BaseCloudAgentClient):
    """
    Minimal test double for BaseCloudAgentClient.

    - `response` is returned for every call (can be None to simulate "no plan").
    - Records `call_count` and `last_packet` for assertions.
    - Optional `delay` simulates network latency.
    """

    def __init__(self, response: Plan | None = None, *, delay: float = 0.0) -> None:
        self.response = response
        self.delay = float(delay)
        self.call_count = 0
        self.last_packet: MaydayPacket | None = None

    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        self.call_count += 1
        self.last_packet = packet
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return self.response
