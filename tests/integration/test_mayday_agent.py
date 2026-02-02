from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, cast

import pytest

from cortexguard.edge.mayday_agent import MaydayAgent
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.goal import GoalContext
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth
from cortexguard.edge.models.plan import Plan, PlanSource, PlanStatus, PlanStep, PlanType
from cortexguard.edge.models.reasoning_trace_entry import TraceSeverity
from cortexguard.edge.models.remediation_policy import RemediationPolicy


# -------------------------
# Test doubles / helpers
# -------------------------
class RecordingTraceSink:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def post_trace_entry(
        self,
        source: object | str,
        event_type: str,
        reasoning_text: str,
        metadata: dict[str, Any] | None = None,
        *,
        severity: TraceSeverity = TraceSeverity.INFO,
        refs: dict[str, str] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.calls.append(
            {
                "event_type": event_type,
                "reasoning_text": reasoning_text,
                "metadata": metadata or {},
                "severity": severity,
                "duration_ms": duration_ms,
            }
        )

    def nonblocking_post_trace_entry(
        self,
        source: object | str,
        event_type: str,
        reasoning_text: str,
        metadata: dict[str, Any] | None = None,
        *,
        severity: TraceSeverity = TraceSeverity.INFO,
        refs: dict[str, str] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        asyncio.create_task(
            self.post_trace_entry(
                source,
                event_type,
                reasoning_text,
                metadata,
                severity=severity,
                refs=refs,
                duration_ms=duration_ms,
            )
        )


class FakeSceneGraphBad:
    def to_compact_dict(self):
        raise RuntimeError("boom")


class StateWithToDict:
    def to_dict(self) -> dict[str, Any]:
        return {"foo": "bar"}


class StateWithDictAttr:
    def __init__(self) -> None:
        self.x: int = 1


class FakeBlackboard:
    def __init__(self) -> None:
        self._scene: Any | None = None
        self._state: Any | None = None
        self._anoms: dict[str, AnomalyEvent] = {}
        self._store: dict[str, Any] = {}

    async def get_latest_state_estimate(self) -> Any:
        return self._state

    async def set_latest_state_estimate(self, state: Any) -> None:
        self._state = state

    async def get_scene_graph(self) -> Any:
        return self._scene

    async def set_scene_graph(self, sg: Any) -> None:
        self._scene = sg

    async def get_active_anomalies(self) -> dict[str, AnomalyEvent]:
        return self._anoms

    async def get_current_plan(self) -> Any:
        return None

    async def get_current_step(self) -> Any:
        return None

    async def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)


# --- Fake cloud clients used in tests -------------------------------------
class FakeCloudClientSuccess:
    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        step = PlanStep(
            id="step-1",
            description="Demo recovery step",
            action=AgentToolCall(action_name="noop", arguments={}),
        )
        goal = GoalContext(goal_id="g-1", user_prompt="recover", intent="recover", priority=0)
        return Plan(
            plan_id=f"cloud-{packet.trace_id}",
            context=goal,
            plan_type=PlanType.REMEDIATION,
            steps=[step],
            status=PlanStatus.PENDING,
            source=PlanSource.CLOUD_AGENT,
            trace_id=packet.trace_id,
        )


class FlakyCloudClient:
    def __init__(self) -> None:
        self.calls = 0

    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        self.calls += 1
        if self.calls == 1:
            # simulate a long-running call that will trigger asyncio.TimeoutError in agent
            await asyncio.sleep(0.5)
            return None
        step = PlanStep(
            id="step-1",
            description="recovery",
            action=AgentToolCall(action_name="noop", arguments={}),
        )
        goal = GoalContext(goal_id="g-1", user_prompt="recover", intent="recover", priority=0)
        return Plan(
            plan_id=f"cloud-{packet.trace_id}",
            context=goal,
            plan_type=PlanType.REMEDIATION,
            steps=[step],
            status=PlanStatus.PENDING,
            source=PlanSource.CLOUD_AGENT,
            trace_id=packet.trace_id,
        )


class AlwaysSlowCloudClient:
    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        await asyncio.sleep(0.5)
        return None


# -------------------------
# Policy helper
# -------------------------
def make_policy(
    with_replay: bool = False, include_model_dump: bool = True
) -> RemediationPolicy | object:
    anomaly = AnomalyEvent(
        id="anom-1",
        key="test_anomaly",
        timestamp=datetime.now(),
        severity=AnomalySeverity.HIGH,
        score=0.99,
        contributing_detectors=["detector-a"],
        metadata={},
    )
    step = PlanStep(
        id="pstep-1",
        description="Test corrective step",
        action=AgentToolCall(action_name="noop", arguments={}),
    )

    if with_replay:
        anomaly.metadata = dict(anomaly.metadata or {})
        anomaly.metadata["replay"] = {
            "window": {"start_ts": datetime.now(), "end_ts": None},
            "compressed_data": "dummy-compressed-base64",
            "format": "lz4",
        }

    policy = RemediationPolicy(
        policy_id="policy-1",
        trigger_event=anomaly,
        reasoning_trace="test reasoning",
        risk_assessment="HIGH",
        corrective_steps=[step],
        escalation_required=True,
        created_at=datetime.now(),
    )

    if not include_model_dump:

        class SimplePolicy:
            def __init__(self, base: RemediationPolicy) -> None:
                self.policy_id = base.policy_id
                self.trigger_event = base.trigger_event
                self.reasoning_trace = base.reasoning_trace
                self.risk_assessment = base.risk_assessment
                self.corrective_steps = base.corrective_steps
                self.escalation_required = base.escalation_required
                self.created_at = base.created_at

        return SimplePolicy(policy)

    return policy


# -------------------------
# Integration tests
# -------------------------
@pytest.mark.asyncio
async def test_build_and_send_escalation_success_integration() -> None:
    trace_sink = RecordingTraceSink()
    client = FakeCloudClientSuccess()
    agent = MaydayAgent(cloud_agent_client=client, device_id="dev-int-1", trace_sink=trace_sink)

    bb = FakeBlackboard()
    policy = make_policy(with_replay=True)
    packet = await agent.build_packet_from_policy(
        policy=cast(RemediationPolicy, policy),
        blackboard=cast(Blackboard, bb),
        health=SystemHealth(
            cpu_load_pct=0.2, net_rtt_ms=5, packet_loss_pct=0.0, disk_pressure_pct=0.0
        ),
        trace_id="int-success",
    )

    assert packet.trace_id == "int-success"
    assert packet.device_id == "dev-int-1"
    assert packet.replay_data is not None

    plan = await agent.send_escalation(packet)
    assert plan is not None
    assert plan.trace_id == packet.trace_id

    # traces should include PACKET_BUILT and ESCALATION_SUCCESS
    assert any(c["event_type"] == "PACKET_BUILT" for c in trace_sink.calls)
    assert any(c["event_type"] == "ESCALATION_SUCCESS" for c in trace_sink.calls)


@pytest.mark.asyncio
async def test_send_escalation_retries_and_succeeds_integration() -> None:
    trace_sink = RecordingTraceSink()
    client = FlakyCloudClient()
    agent = MaydayAgent(
        cloud_agent_client=client,
        device_id="dev-int-2",
        trace_sink=trace_sink,
        timeout_seconds=0.05,
        max_retries=1,
        backoff_factor=1.0,
    )

    bb = FakeBlackboard()
    policy = make_policy()
    packet = await agent.build_packet_from_policy(
        policy=cast(RemediationPolicy, policy),
        blackboard=cast(Blackboard, bb),
        health=SystemHealth(
            cpu_load_pct=None, net_rtt_ms=None, packet_loss_pct=None, disk_pressure_pct=None
        ),
        trace_id="int-retry",
    )

    plan = await agent.send_escalation(packet)
    assert plan is not None
    # ensure the flaky client was called at least twice (first timed out, second succeeded)
    assert client.calls >= 2

    # traces should include ESCALATION_ATTEMPT and ESCALATION_SUCCESS
    assert any(c["event_type"] == "ESCALATION_ATTEMPT" for c in trace_sink.calls)
    assert any(c["event_type"] == "ESCALATION_SUCCESS" for c in trace_sink.calls)


@pytest.mark.asyncio
async def test_send_escalation_exhausts_retries_integration() -> None:
    trace_sink = RecordingTraceSink()
    client = AlwaysSlowCloudClient()
    agent = MaydayAgent(
        cloud_agent_client=client,
        device_id="dev-int-3",
        trace_sink=trace_sink,
        timeout_seconds=0.05,
        max_retries=2,
        backoff_factor=1.0,
    )

    bb = FakeBlackboard()
    policy = make_policy()
    packet = await agent.build_packet_from_policy(
        policy=cast(RemediationPolicy, policy),
        blackboard=cast(Blackboard, bb),
        health=SystemHealth(
            cpu_load_pct=None, net_rtt_ms=None, packet_loss_pct=None, disk_pressure_pct=None
        ),
        trace_id="int-exhaust",
    )

    plan = await agent.send_escalation(packet)
    assert plan is None

    # should have emitted attempts and a final failure trace
    assert any(c["event_type"] == "ESCALATION_ATTEMPT" for c in trace_sink.calls)
    assert any(c["event_type"] == "ESCALATION_FAILURE" for c in trace_sink.calls)
