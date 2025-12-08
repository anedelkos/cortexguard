from __future__ import annotations

from typing import Protocol

from kitchenwatch.edge.models.mayday_packet import MaydayPacket
from kitchenwatch.edge.models.plan import Plan


class BaseCloudAgentClient(Protocol):
    """
    Edge-side protocol for sending an escalation packet to a cloud planner
    and receiving an executable Plan in response.
    """

    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        """
        Send a MaydayPacket to the cloud planner.

        Returns a Plan (with plan_type=REMEDIATION and trace_id set to packet.trace_id)
        when the cloud returns a recovery plan, or None if no plan is available.
        """
        ...
