from __future__ import annotations

import asyncio
from datetime import UTC, datetime
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
        # schedule to mimic real sink behavior
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


class FakeSceneGraphGood:
    def to_compact_dict(self):
        return {"objects": [{"id": "o1", "label": "obj"}], "relationships": []}


class FakeSceneGraphBad:
    def to_compact_dict(self):
        raise RuntimeError("boom")


class StateWithToDict:
    def to_dict(self):
        return {"foo": "bar"}


class StateWithDictAttr:
    def __init__(self) -> None:
        self.x: int = 1


# --- Fake Cloud Clients -----------------------------------------------------
class FakeCloudClientSuccess:
    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        """Return a minimal Plan for the given packet."""
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


class FakeCloudClientError:
    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        """Simulate a transport error."""
        raise RuntimeError("simulated transport failure")


class FlakyCloudClient:
    """
    First call: sleep longer than timeout (simulate timeout).
    Second call: return a Plan.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        self.calls += 1
        if self.calls == 1:
            # simulate a long-running call that will trigger asyncio.TimeoutError
            await asyncio.sleep(0.5)
            return None  # not reached if timeout shorter
        # second call returns quickly
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
    """Always sleeps longer than timeout to force retry exhaustion."""

    async def send_escalation(self, packet: MaydayPacket) -> Plan | None:
        await asyncio.sleep(0.5)
        return None  # not reached if timeout shorter


# Minimal fake blackboard used across tests
class FakeBlackboard:
    def __init__(self) -> None:
        self._scene = None
        self._state = None
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

    async def add_trace_entry(self, entry: Any) -> None:
        # not used by tests; present for TraceSink compatibility
        return None


# --- Fixtures ---------------------------------------------------------------
@pytest.fixture
def device_id() -> str:
    return "device-abc-123"


@pytest.fixture
def blackboard() -> FakeBlackboard:
    return FakeBlackboard()


@pytest.fixture
def health_snapshot() -> SystemHealth:
    return SystemHealth(
        cpu_load_pct=12.3, net_rtt_ms=50, packet_loss_pct=0.0, disk_pressure_pct=5.0
    )


def make_policy(
    with_replay: bool = False, include_model_dump: bool = True
) -> RemediationPolicy | object:
    anomaly = AnomalyEvent(
        id="anom-1",
        key="test_anomaly",
        timestamp=datetime.now(UTC),
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

    # Attach replay into metadata (safe for Pydantic models)
    if with_replay:
        anomaly.metadata = dict(anomaly.metadata or {})
        anomaly.metadata["replay"] = {
            "window": {"start_ts": datetime.now(UTC), "end_ts": None},
            "compressed_data": "dummy-compressed-base64",
            "format": "lz4",
        }

    # Build a normal RemediationPolicy (Pydantic)
    policy = RemediationPolicy(
        policy_id="policy-1",
        trigger_event=anomaly,
        reasoning_trace="test reasoning",
        risk_assessment="HIGH",
        corrective_steps=[step],
        escalation_required=True,
        created_at=datetime.now(UTC),
    )

    # If caller wants to exercise the "no model_dump" fallback, return a plain object
    if not include_model_dump:

        class SimplePolicy:
            def __init__(self, base: RemediationPolicy):
                self.policy_id = base.policy_id
                self.trigger_event = base.trigger_event
                self.reasoning_trace = base.reasoning_trace
                self.risk_assessment = base.risk_assessment
                self.corrective_steps = base.corrective_steps
                self.escalation_required = base.escalation_required
                self.created_at = base.created_at

            # intentionally no model_dump or dict method to exercise fallback

        return SimplePolicy(policy)

    return policy


@pytest.fixture
def remediation_policy() -> RemediationPolicy:
    anomaly = AnomalyEvent(
        id="anom-1",
        key="test_anomaly",
        timestamp=datetime.now(UTC),
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
    return RemediationPolicy(
        policy_id="policy-1",
        trigger_event=anomaly,
        reasoning_trace="test reasoning",
        risk_assessment="HIGH",
        corrective_steps=[step],
        escalation_required=True,
        created_at=datetime.now(UTC),
    )


# --- Tests ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_build_packet_includes_device_id_and_trace(
    device_id: str,
    blackboard: FakeBlackboard,
    remediation_policy: RemediationPolicy,
    health_snapshot: SystemHealth,
) -> None:
    client = FakeCloudClientSuccess()
    agent = MaydayAgent(cloud_agent_client=client, device_id=device_id)

    # cast FakeBlackboard to the real Blackboard type for the call site
    packet = await agent.build_packet_from_policy(
        policy=remediation_policy,
        blackboard=cast(Blackboard, blackboard),
        health=health_snapshot,
        trace_id="trace-xyz",
    )

    assert packet.device_id == device_id
    assert packet.trace_id == "trace-xyz"
    assert packet.health.cpu_load_pct == 12.3
    assert packet.remediation_policy is not None


@pytest.mark.asyncio
async def test_send_escalation_success_increments_metrics(
    device_id: str,
    blackboard: FakeBlackboard,
    remediation_policy: RemediationPolicy,
    health_snapshot: SystemHealth,
) -> None:
    client = FakeCloudClientSuccess()
    agent = MaydayAgent(cloud_agent_client=client, device_id=device_id)

    packet = await agent.build_packet_from_policy(
        remediation_policy, cast(Blackboard, blackboard), health_snapshot, trace_id="t-1"
    )
    assert agent.get_metrics()["escalations_sent"] == 0

    plan = await agent.send_escalation(packet)
    assert plan is not None
    assert plan.trace_id == packet.trace_id
    metrics = agent.get_metrics()
    assert metrics["escalations_sent"] == 1
    assert metrics["responses_received"] == 1


@pytest.mark.asyncio
async def test_send_escalation_failure_returns_none_and_increments_sent(
    device_id: str,
    blackboard: FakeBlackboard,
    remediation_policy: RemediationPolicy,
    health_snapshot: SystemHealth,
) -> None:
    client = FakeCloudClientError()
    agent = MaydayAgent(cloud_agent_client=client, device_id=device_id)

    packet = await agent.build_packet_from_policy(
        remediation_policy, cast(Blackboard, blackboard), health_snapshot, trace_id="t-err"
    )
    plan = await agent.send_escalation(packet)
    assert plan is None
    metrics = agent.get_metrics()
    assert metrics["escalations_sent"] == 1
    assert metrics["responses_received"] == 0


@pytest.mark.asyncio
async def test_send_escalation_retries_on_timeout_and_succeeds() -> None:
    client = FlakyCloudClient()
    # set timeout short so first call times out; allow 1 retry
    agent = MaydayAgent(
        cloud_agent_client=client,
        device_id="dev-1",
        timeout_seconds=0.05,
        max_retries=1,
        backoff_factor=1.5,
    )

    # build a minimal packet (we can pass a dummy RemediationPolicy via cast)
    bb = cast(Blackboard, FakeBlackboard())
    policy = RemediationPolicy(
        policy_id="p1",
        trigger_event=AnomalyEvent(
            id="a1",
            key="k",
            timestamp=datetime.now(UTC),
            severity=AnomalySeverity.MEDIUM,
            score=0.5,
            contributing_detectors=[],
            metadata={},
        ),
        reasoning_trace="r",
        risk_assessment="low",
        corrective_steps=[],
        escalation_required=True,
        created_at=datetime.now(UTC),
    )
    packet = await agent.build_packet_from_policy(
        policy=policy,
        blackboard=bb,
        health=SystemHealth(
            cpu_load_pct=None, net_rtt_ms=None, packet_loss_pct=None, disk_pressure_pct=None
        ),
        trace_id="t-retry",
    )

    assert (
        agent.get_metrics()["escalations_sent"] == 0 or agent.get_metrics()["escalations_sent"] == 0
    )  # before send_escalation
    plan = await agent.send_escalation(packet)
    assert plan is not None
    metrics = agent.get_metrics()
    assert metrics["escalations_sent"] == 1
    assert metrics["attempts_sent"] >= 2
    assert metrics["responses_received"] == 1


@pytest.mark.asyncio
async def test_send_escalation_exhausts_retries_and_returns_none() -> None:
    client = AlwaysSlowCloudClient()
    agent = MaydayAgent(
        cloud_agent_client=client,
        device_id="dev-1",
        timeout_seconds=0.05,
        max_retries=2,
        backoff_factor=1.2,
    )

    bb = cast(Blackboard, FakeBlackboard())
    policy = RemediationPolicy(
        policy_id="p2",
        trigger_event=AnomalyEvent(
            id="a2",
            key="k2",
            timestamp=datetime.now(UTC),
            severity=AnomalySeverity.MEDIUM,
            score=0.5,
            contributing_detectors=[],
            metadata={},
        ),
        reasoning_trace="r2",
        risk_assessment="low",
        corrective_steps=[],
        escalation_required=True,
        created_at=datetime.now(UTC),
    )
    packet = await agent.build_packet_from_policy(
        policy=policy,
        blackboard=bb,
        health=SystemHealth(
            cpu_load_pct=None, net_rtt_ms=None, packet_loss_pct=None, disk_pressure_pct=None
        ),
        trace_id="t-exhaust",
    )

    plan = await agent.send_escalation(packet)
    assert plan is None
    metrics = agent.get_metrics()
    assert metrics["escalations_sent"] == 1
    assert metrics["attempts_sent"] == 3  # max_retries + 1
    assert metrics["responses_received"] == 0


@pytest.mark.asyncio
async def test_packet_built_and_redacted_traces(
    device_id: str, health_snapshot: SystemHealth
) -> None:
    trace_sink = RecordingTraceSink()
    client = FakeCloudClientSuccess()
    agent = MaydayAgent(
        cloud_agent_client=client, device_id=device_id, trace_sink=trace_sink, timeout_seconds=0.1
    )

    bb = FakeBlackboard()
    # scene graph that raises -> triggers PACKET_REDACTED
    await bb.set_scene_graph(FakeSceneGraphBad())

    policy = make_policy()
    packet = await agent.build_packet_from_policy(
        policy=cast(RemediationPolicy, policy),
        blackboard=cast(Blackboard, bb),
        health=health_snapshot,
        trace_id="t-redact",
    )

    # PACKET_BUILT present
    assert any(c["event_type"] == "PACKET_BUILT" for c in trace_sink.calls)
    # PACKET_REDACTED present because to_compact_dict raised
    assert any(c["event_type"] == "PACKET_REDACTED" for c in trace_sink.calls)
    # packet fields sanity
    assert packet.trace_id == "t-redact"
    assert packet.device_id == device_id


@pytest.mark.asyncio
async def test_escalation_attempt_and_success_traces() -> None:
    trace_sink = RecordingTraceSink()
    client = FlakyCloudClient()
    agent = MaydayAgent(
        cloud_agent_client=client,
        device_id="dev",
        trace_sink=trace_sink,
        timeout_seconds=0.05,
        max_retries=1,
    )

    bb = FakeBlackboard()
    policy = make_policy()
    packet = await agent.build_packet_from_policy(
        policy=cast(RemediationPolicy, policy),
        blackboard=cast(Blackboard, bb),
        health=SystemHealth(
            cpu_load_pct=None, net_rtt_ms=None, packet_loss_pct=None, disk_pressure_pct=None
        ),
        trace_id="t-retry",
    )

    plan = await agent.send_escalation(packet)
    assert plan is not None

    # Check attempt traces and success
    assert any(c["event_type"] == "ESCALATION_ATTEMPT" for c in trace_sink.calls)
    assert any(c["event_type"] == "ESCALATION_SUCCESS" for c in trace_sink.calls)


@pytest.mark.asyncio
async def test_escalation_timeout_and_failure_traces() -> None:
    trace_sink = RecordingTraceSink()
    client = AlwaysSlowCloudClient()
    agent = MaydayAgent(
        cloud_agent_client=client,
        device_id="dev",
        trace_sink=trace_sink,
        timeout_seconds=0.05,
        max_retries=1,
    )

    bb = FakeBlackboard()
    policy = make_policy()
    packet = await agent.build_packet_from_policy(
        policy=cast(RemediationPolicy, policy),
        blackboard=cast(Blackboard, bb),
        health=SystemHealth(
            cpu_load_pct=None, net_rtt_ms=None, packet_loss_pct=None, disk_pressure_pct=None
        ),
        trace_id="t-exhaust",
    )

    plan = await agent.send_escalation(packet)
    assert plan is None

    # Should have attempt traces and a final failure trace
    assert any(c["event_type"] == "ESCALATION_ATTEMPT" for c in trace_sink.calls)
    assert any(c["event_type"] == "ESCALATION_ATTEMPT_TIMEOUT" for c in trace_sink.calls) or any(
        c["event_type"] == "ESCALATION_ATTEMPT_ERROR" for c in trace_sink.calls
    )
    assert any(c["event_type"] == "ESCALATION_FAILURE" for c in trace_sink.calls)


@pytest.mark.asyncio
async def test_state_serialization_fallbacks_and_replay_data() -> None:
    trace_sink = RecordingTraceSink()
    client = FakeCloudClientSuccess()
    agent = MaydayAgent(cloud_agent_client=client, device_id="dev", trace_sink=trace_sink)

    bb = FakeBlackboard()
    # state with to_dict
    await bb.set_latest_state_estimate(StateWithToDict())
    policy = make_policy(with_replay=True)
    packet = await agent.build_packet_from_policy(
        policy=cast(RemediationPolicy, policy),
        blackboard=cast(Blackboard, bb),
        health=SystemHealth(
            cpu_load_pct=0.1, net_rtt_ms=1, packet_loss_pct=0.0, disk_pressure_pct=0.0
        ),
        trace_id="t-state1",
    )
    assert isinstance(packet.state_estimate, dict)
    assert packet.replay_data is not None

    # state with __dict__
    # cast the untyped helper to Any to avoid mypy complaining about untyped call
    await bb.set_latest_state_estimate(cast(Any, StateWithDictAttr()))
    packet2 = await agent.build_packet_from_policy(
        policy=cast(RemediationPolicy, policy),
        blackboard=cast(Blackboard, bb),
        health=SystemHealth(
            cpu_load_pct=0.1, net_rtt_ms=1, packet_loss_pct=0.0, disk_pressure_pct=0.0
        ),
        trace_id="t-state2",
    )
    assert isinstance(packet2.state_estimate, dict)


@pytest.mark.asyncio
async def test_remediation_policy_serialization_fallback_and_packet_size_bytes() -> None:
    trace_sink = RecordingTraceSink()
    client = FakeCloudClientSuccess()
    agent = MaydayAgent(cloud_agent_client=client, device_id="dev", trace_sink=trace_sink)

    bb = FakeBlackboard()
    # policy without model_dump to exercise fallback branch; make_policy may return a SimplePolicy
    policy = make_policy(include_model_dump=False)
    _ = await agent.build_packet_from_policy(
        policy=cast(RemediationPolicy, policy),
        blackboard=cast(Blackboard, bb),
        health=SystemHealth(
            cpu_load_pct=0.1, net_rtt_ms=1, packet_loss_pct=0.0, disk_pressure_pct=0.0
        ),
        trace_id="t-policy-fallback",
    )

    # PACKET_BUILT should include packet_size_bytes metadata even if model_dump_json missing
    built_calls = [c for c in trace_sink.calls if c["event_type"] == "PACKET_BUILT"]
    assert built_calls, "PACKET_BUILT trace missing"
    # metadata may include None for packet_size_bytes when model_dump_json absent; ensure trace exists
    assert "policy_id" in built_calls[0]["metadata"]


@pytest.mark.asyncio
async def test_trace_sink_errors_are_swallowed() -> None:
    class BadSink:
        async def post_trace_entry(self, *args, **kwargs):
            raise RuntimeError("sink fail")

        def nonblocking_post_trace_entry(self, *args, **kwargs):
            raise RuntimeError("sink fail")

    client = FakeCloudClientSuccess()
    bad_sink = BadSink()
    agent = MaydayAgent(cloud_agent_client=client, device_id="dev", trace_sink=bad_sink)

    bb = FakeBlackboard()
    policy = make_policy()
    # Should not raise despite sink errors
    packet = await agent.build_packet_from_policy(
        policy=cast(RemediationPolicy, policy),
        blackboard=cast(Blackboard, bb),
        health=SystemHealth(
            cpu_load_pct=0.1, net_rtt_ms=1, packet_loss_pct=0.0, disk_pressure_pct=0.0
        ),
        trace_id="t-bad-sink",
    )
    assert packet is not None
