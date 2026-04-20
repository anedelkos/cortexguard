from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
import pytest

from cortexguard.core.http_cloud_client import HttpCloudAgentClient
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth
from cortexguard.edge.models.plan import Plan, PlanSource, PlanStep, PlanType


def _make_packet() -> MaydayPacket:
    return MaydayPacket(
        trace_id=str(uuid.uuid4()),
        device_id="robot-arm-01",
        timestamp=datetime.now(UTC),
        health=SystemHealth(),
    )


def _make_plan() -> Plan:
    return Plan(
        plan_id=str(uuid.uuid4()),
        context=GoalContext(
            goal_id=str(uuid.uuid4()),
            user_prompt="safe park",
            intent="Park arm safely",
        ),
        plan_type=PlanType.REMEDIATION,
        source=PlanSource.CLOUD_AGENT,
        steps=[
            PlanStep(
                description="park arm",
                action=AgentToolCall(
                    action_name="PLACE_ITEM",
                    arguments={
                        "item_name": "arm",
                        "target_location": "home",
                        "tool_id": "RoboticGripper_A",
                    },
                ),
            )
        ],
    )


class _SequentialTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        resp = self._responses[self._index % len(self._responses)]
        self._index += 1
        return resp


class _ConnectErrorTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")


@pytest.mark.asyncio
async def test_post_then_poll_returns_plan() -> None:
    trace_id = str(uuid.uuid4())
    plan = _make_plan()

    responses = [
        httpx.Response(202, json={"trace_id": trace_id}),
        httpx.Response(202, json={"status": "pending"}),
        httpx.Response(
            200,
            content=plan.model_dump_json().encode(),
            headers={"Content-Type": "application/json"},
        ),
    ]
    transport = _SequentialTransport(responses)
    client = httpx.AsyncClient(transport=transport, base_url="http://cloud")

    agent = HttpCloudAgentClient(
        cloud_base_url="http://cloud",
        poll_interval_s=0.0,
        http_client=client,
    )
    result = await agent.send_escalation(_make_packet())
    assert result is not None
    assert isinstance(result, Plan)


@pytest.mark.asyncio
async def test_get_404_returns_none() -> None:
    trace_id = str(uuid.uuid4())

    responses = [
        httpx.Response(202, json={"trace_id": trace_id}),
        httpx.Response(404, json={"detail": "not found"}),
    ]
    transport = _SequentialTransport(responses)
    client = httpx.AsyncClient(transport=transport, base_url="http://cloud")

    agent = HttpCloudAgentClient(
        cloud_base_url="http://cloud",
        poll_interval_s=0.0,
        http_client=client,
    )
    result = await agent.send_escalation(_make_packet())
    assert result is None


@pytest.mark.asyncio
async def test_connect_error_returns_none() -> None:
    client = httpx.AsyncClient(transport=_ConnectErrorTransport(), base_url="http://cloud")
    agent = HttpCloudAgentClient(
        cloud_base_url="http://cloud",
        poll_interval_s=0.0,
        http_client=client,
    )
    result = await agent.send_escalation(_make_packet())
    assert result is None
