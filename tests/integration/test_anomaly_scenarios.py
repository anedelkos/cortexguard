import logging
import random
from collections import deque

import pytest
import pytest_asyncio

from cortexguard.common.constants import DATA_DIR
from cortexguard.edge.models.capability_registry import FunctionSchema, RiskLevel
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot
from cortexguard.edge.models.reasoning_trace_entry import TraceSeverity
from cortexguard.edge.runtime import EdgeRuntime, create_runtime
from cortexguard.edge.safety_agent import SafetyCommand
from cortexguard.simulation.models.windowed_fused_record import SensorReading, WindowedFusedRecord
from tests.integration.chaos_engine import ChaosEngine
from tests.integration.scenario_loader import Scenario, load_scenarios

logger = logging.getLogger(__name__)


def make_synthetic_baseline(
    n_records: int = 20, window_size: int = 5, dt_ns: int = 10_000_000, seed: int = 42
) -> list[WindowedFusedRecord]:
    rng = random.Random(seed)
    base_ts = 1_700_000_000_000_000_000
    records = []

    drift = {
        k: 0.0
        for k in [
            "force_x",
            "force_y",
            "force_z",
            "torque_x",
            "torque_y",
            "torque_z",
            "pos_x",
            "pos_y",
            "pos_z",
            "temp_c",
            "smoke_ppm",
        ]
    }

    for i in range(n_records):
        ts = base_ts + i * dt_ns

        # small drift
        for k in drift:
            drift[k] += rng.uniform(-0.005, 0.005)

        sensor_window = []
        for j in range(window_size):
            sensor_window.append(
                SensorReading(
                    timestamp_ns=ts - (window_size - j) * 1_000_000,
                    force_x=drift["force_x"] + rng.uniform(-0.02, 0.02),
                    force_y=drift["force_y"] + rng.uniform(-0.02, 0.02),
                    force_z=drift["force_z"] + rng.uniform(-0.02, 0.02),
                    torque_x=drift["torque_x"] + rng.uniform(-0.02, 0.02),
                    torque_y=drift["torque_y"] + rng.uniform(-0.02, 0.02),
                    torque_z=drift["torque_z"] + rng.uniform(-0.02, 0.02),
                    pos_x=drift["pos_x"] + rng.uniform(-0.02, 0.02),
                    pos_y=drift["pos_y"] + rng.uniform(-0.02, 0.02),
                    pos_z=drift["pos_z"] + rng.uniform(-0.02, 0.02),
                    temp_c=drift["temp_c"] + rng.uniform(-0.02, 0.02),
                    smoke_ppm=drift["smoke_ppm"] + rng.uniform(-0.02, 0.02),
                )
            )

        records.append(
            WindowedFusedRecord(
                timestamp_ns=ts,
                rgb_path="",
                depth_path=None,
                window_size_s=0.1,
                n_samples=len(sensor_window),
                sensor_window=sensor_window,
            )
        )

    return records


@pytest_asyncio.fixture
async def runtime():
    rt = create_runtime(profile="simulation")

    # --- Inject slowdown capability so value_freeze policies validate ---
    rt.capability_registry.capabilities["set_speed_limit"] = FunctionSchema(
        description="Reduce system motion speed by applying a speed factor.",
        parameters={
            "type": "object",
            "properties": {
                "tool_id": {"type": "string"},
                "speed_factor": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["tool_id", "speed_factor"],
        },
        risk_level=RiskLevel.LOW,
    )

    async with rt.managed() as started:
        yield started


@pytest.mark.parametrize("scenario", load_scenarios(DATA_DIR / "anomaly_scenarios.yaml").values())
@pytest.mark.asyncio
async def test_anomaly_scenario(runtime: EdgeRuntime, scenario: Scenario) -> None:
    chaos = ChaosEngine(scenario)

    baseline_records: list[WindowedFusedRecord] = make_synthetic_baseline()

    def vision_stub(record, image_embedding):
        vision_objects = chaos._vision_sidecar.get(record.timestamp_ns, [])
        return vision_objects, chaos._vision_occlusion

    runtime.edge_fusion.vision_inference = vision_stub

    snapshots: list[FusionSnapshot] = []
    for record in chaos.inject(iter(baseline_records)):
        await runtime.edge_fusion.process_record(record)
        snapshot = await runtime.blackboard.get_fusion_snapshot()
        if snapshot:
            snapshots.append(snapshot)

    await runtime.anomaly_detector.stop()
    await runtime.policy_agent.stop()

    # Branch assertions based on expected outcome
    match scenario.expected_outcome:
        case "hard_stop":

            state_estimate = await runtime.blackboard.get_latest_state_estimate()
            assert state_estimate is not None
            logger.info(
                "symbolic_system_state=%s flags=%s",
                state_estimate.symbolic_system_state,
                state_estimate.flags,
            )
            logger.info("residuals=%s", state_estimate.residuals)
            logger.info("uncertainty=%s", state_estimate.uncertainty)

            match scenario.scenario_id:
                case "S0.1":
                    snapshot = await runtime.blackboard.get_fusion_snapshot()
                    assert (
                        snapshot is not None
                    ), f"{scenario.scenario_id}: no fusion snapshot available"

                    await runtime.anomaly_detector._run_tick()
                    anomaly_present = await runtime.blackboard.is_anomaly_present()
                    assert anomaly_present

                    anomalies = await runtime.blackboard.get_active_anomalies()
                    assert "HUMAN_PROXIMITY_VIOLATION" in anomalies

                    traces = runtime.blackboard.reasoning_traces
                    assert "HUMAN_PROXIMITY_VIOLATION" in traces[-1].reasoning_text
                    assert traces[-1].severity == TraceSeverity.CRITICAL

                case "S0.2":
                    snapshot = await runtime.blackboard.get_fusion_snapshot()
                    assert (
                        snapshot is not None
                    ), f"{scenario.scenario_id}: no fusion snapshot available"

                    await runtime.anomaly_detector._run_tick()
                    anomaly_present = await runtime.blackboard.is_anomaly_present()
                    assert anomaly_present

                    anomalies = await runtime.blackboard.get_active_anomalies()
                    assert "overheat_smoke_combo" in anomalies

                    traces = runtime.blackboard.reasoning_traces
                    assert "overheat_smoke_combo" in traces[-1].reasoning_text
                    assert traces[-1].severity == TraceSeverity.CRITICAL

            cmd = await runtime.orchestrator._check_safety()
            assert cmd
            expected_cmd = SafetyCommand(action="E-STOP")
            assert (
                cmd.action == expected_cmd.action
            ), f"{scenario.scenario_id}: no safety breach detected"

            flags = runtime.blackboard.safety_flags
            assert "emergency_stop" in flags
            assert flags["emergency_stop"]

        case "escalate_to_cloud":

            snapshot = await runtime.blackboard.get_fusion_snapshot()
            assert snapshot is not None, f"{scenario.scenario_id}: no fusion snapshot available"

            for i in range(-3, 0, 1):
                await runtime.blackboard.update_fusion_snapshot(snapshots[i])
                await runtime.anomaly_detector._run_tick()

            anomaly_present = await runtime.blackboard.is_anomaly_present()
            assert anomaly_present

            anomalies = await runtime.blackboard.get_active_anomalies()
            assert "repeated_system_failure" in anomalies
            assert (
                anomalies["repeated_system_failure"].metadata.get("failure_key") == "grasp_success"
            )
            assert anomalies["repeated_system_failure"].metadata.get("failure_count") == 3

            await runtime.policy_agent._process_active_anomalies_tick()
            assert any(
                "ESCALATION_ATTEMPT" in e.event_type for e in runtime.blackboard.reasoning_traces
            ), f"{scenario.scenario_id}: no escalation trace found"

        case "recovery_success":
            snapshot = await runtime.blackboard.get_fusion_snapshot()
            assert snapshot is not None, f"{scenario.scenario_id}: no fusion snapshot available"

            await runtime.anomaly_detector._run_tick()
            anomaly_present = await runtime.blackboard.is_anomaly_present()
            assert anomaly_present

            anomalies = await runtime.blackboard.get_active_anomalies()
            assert "temp_c_value_freeze" in anomalies

            await runtime.policy_agent._process_active_anomalies_tick()

            assert not any(
                "ESCALATION_ATTEMPT" in e.event_type for e in runtime.blackboard.reasoning_traces
            ), f"{scenario.scenario_id}: no escalation trace found"

            cmd = await runtime.orchestrator._check_safety()
            assert cmd
            expected_cmd = SafetyCommand(action="NOMINAL")
            assert (
                cmd.action == expected_cmd.action
            ), f"{scenario.scenario_id}: safety breach detected"

        case "resilient_recovery_or_escalate":
            # Sanity
            assert snapshots, f"{scenario.scenario_id}: no snapshots collected"

            # Find a failing snapshot to drive the detector (we only need one snapshot to simulate repeated failures)
            failing_snaps = [
                s for s in snapshots if s is not None and s.derived.get("grasp_success") is False
            ]
            assert (
                failing_snaps
            ), f"{scenario.scenario_id}: no failing snapshot (grasp_success=False) to drive detector"

            # Drive the detector deterministically: replay the same failing fusion snapshot 3x
            for i in range(3):
                await runtime.blackboard.update_fusion_snapshot(list(reversed(failing_snaps))[i])
                await runtime.anomaly_detector._run_tick()
                logger.info("replay tick %d done for scenario %s", i + 1, scenario.scenario_id)

            # Assert aggregated detector outcome (system-level)
            assert (
                await runtime.blackboard.is_anomaly_present()
            ), f"{scenario.scenario_id}: no anomaly detected after driving detector"
            anomalies = await runtime.blackboard.get_active_anomalies()
            logger.info("active anomalies: %s", list(anomalies.keys()))
            assert (
                "repeated_system_failure" in anomalies
            ), f"{scenario.scenario_id}: expected repeated_system_failure, got {list(anomalies.keys())}"
            assert "VISION_OCCLUSION_PERSISTENT" in anomalies

            # Drive policy and assert system-level decision: either recover or escalate
            await runtime.policy_agent._process_active_anomalies_tick()
            recovery = getattr(runtime.blackboard, "recovery_status", False)
            traces = getattr(runtime.blackboard, "reasoning_traces", deque()) or deque()
            logger.info("policy recovery=%s traces=%s", recovery, traces)
            assert recovery or any(
                "ESCALATE" in getattr(t, "event_type", "") for t in traces
            ), f"{scenario.scenario_id}: neither recovery nor escalation detected"

            assert any("COMM_TIMING_DEGRADATION" in t.event_type for t in traces)

        case _:
            # Normal operation, no anomalies
            cmd = await runtime.orchestrator._check_safety()
            assert cmd
            expected_cmd = SafetyCommand(action="NOMINAL")
            assert (
                cmd.action == expected_cmd.action
            ), f"{scenario.scenario_id}: no safety breach detected"

            snapshot = await runtime.blackboard.get_fusion_snapshot()
            logger.info("fusion_snapshot=%s", snapshot)
            scene_graph = await runtime.blackboard.get_scene_graph()
            logger.info("scene_graph=%s", scene_graph)
            state = await runtime.blackboard.get_latest_state_estimate()
            assert state
            logger.info("state.flags=%s symbolic=%s", state.flags, state.symbolic_system_state)

            assert snapshot is not None, f"{scenario.scenario_id}: no fusion snapshot available"

            await runtime.anomaly_detector._run_tick()
            active_anomalies = await runtime.blackboard.get_active_anomalies()
            assert not active_anomalies

            anomaly_present = await runtime.blackboard.is_anomaly_present()
            assert not anomaly_present

            highest = await runtime.blackboard.get_highest_anomaly_severity()
            assert highest is None

            traces = runtime.blackboard.reasoning_traces
            assert traces[-1].metadata["anomalies_emitted"] == 0
            flags = runtime.blackboard.safety_flags
            assert len(flags) == 0

            logger.info("Normal operation, no anomalies scenario passed.")
