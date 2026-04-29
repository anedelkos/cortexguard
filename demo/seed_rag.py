"""Seed Qdrant with resolved historical incidents so the RAG planner has context.

Run inside the cloud-api container:

    docker exec cortexguard-cloud-api python demo/seed_rag.py

The script connects to the same SQLite DB and Qdrant instance the API uses,
inserts synthetic resolved incidents, and indexes them so the next planning run
can retrieve them and generate a plan with actual steps.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cortexguard.cloud.config import CloudConfig
from cortexguard.cloud.persistence.models import IncidentRecord
from cortexguard.cloud.persistence.repository import SQLiteIncidentRepository
from cortexguard.cloud.retrieval.embedder import get_embedder
from cortexguard.cloud.retrieval.store import RetrievalStore
from cortexguard.cloud.retrieval.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStoreProtocol,
)

_RESOLVED_INCIDENTS: list[dict[str, Any]] = [
    {
        "anomaly_key": "ft_impact_impulse",
        "severity": "high",
        "device_id": "device-01",
        "summary": (
            "device=device-01 anomaly=ft_impact_impulse severity=high "
            "actions_taken=stopped device, inspected end-effector, cleared partial obstruction, resumed "
            "outcome=resolved"
        ),
        "candidate_plan_json": json.dumps(
            {
                "steps": [
                    {"action": "emergency_stop", "description": "Immediately halt all motion"},
                    {
                        "action": "inspect_end_effector",
                        "description": "Visually inspect end-effector for damage or obstruction",
                    },
                    {
                        "action": "clear_obstruction",
                        "description": "Manually clear any partial obstruction from the work area",
                    },
                    {
                        "action": "run_diagnostics",
                        "description": "Run force/torque sensor self-test to verify calibration",
                    },
                    {
                        "action": "resume_nominal",
                        "description": "Resume operation at reduced speed and monitor for recurrence",
                    },
                ]
            }
        ),
        "resolution": {
            "actions_taken": "stopped device, inspected end-effector, cleared partial obstruction, resumed",
            "outcome": "resolved",
            "notes": "Obstruction was a displaced workpiece. Force spike was a contact event, not mechanical failure.",
        },
    },
    {
        "anomaly_key": "ft_impact_impulse",
        "severity": "medium",
        "device_id": "device-02",
        "summary": (
            "device=device-02 anomaly=ft_impact_impulse severity=medium "
            "actions_taken=recalibrated force sensor, adjusted trajectory, resumed "
            "outcome=resolved"
        ),
        "candidate_plan_json": json.dumps(
            {
                "steps": [
                    {"action": "pause_operation", "description": "Pause current task safely"},
                    {
                        "action": "recalibrate_force_sensor",
                        "description": "Run force/torque sensor recalibration routine",
                    },
                    {
                        "action": "adjust_approach_trajectory",
                        "description": "Increase approach clearance by 10mm",
                    },
                    {
                        "action": "resume_nominal",
                        "description": "Resume operation and monitor first 3 cycles",
                    },
                ]
            }
        ),
        "resolution": {
            "actions_taken": "recalibrated force sensor, adjusted trajectory, resumed",
            "outcome": "resolved",
            "notes": "Sensor drift caused false positive. Recalibration resolved issue.",
        },
    },
    {
        "anomaly_key": "repeated_system_failure",
        "severity": "high",
        "device_id": "device-01",
        "summary": (
            "device=device-01 anomaly=repeated_system_failure severity=high "
            "actions_taken=power cycled controller, re-homed axes, resumed "
            "outcome=resolved"
        ),
        "candidate_plan_json": json.dumps(
            {
                "steps": [
                    {"action": "emergency_stop", "description": "Halt all motion immediately"},
                    {
                        "action": "power_cycle_controller",
                        "description": "Power cycle the motion controller",
                    },
                    {
                        "action": "rehome_axes",
                        "description": "Re-home all axes from known reference positions",
                    },
                    {
                        "action": "run_self_test",
                        "description": "Run full self-test sequence before resuming",
                    },
                    {
                        "action": "resume_nominal",
                        "description": "Resume at reduced speed for first 10 cycles",
                    },
                ]
            }
        ),
        "resolution": {
            "actions_taken": "power cycled controller, re-homed axes, resumed",
            "outcome": "resolved",
            "notes": "Controller firmware had entered an error state after 3 consecutive step failures. Power cycle cleared it.",
        },
    },
    {
        "anomaly_key": "repeated_system_failure",
        "severity": "high",
        "device_id": "device-03",
        "summary": (
            "device=device-03 anomaly=repeated_system_failure severity=high "
            "actions_taken=replaced gripper jaw, recalibrated, resumed "
            "outcome=hardware_replaced"
        ),
        "candidate_plan_json": json.dumps(
            {
                "steps": [
                    {"action": "emergency_stop", "description": "Halt all motion immediately"},
                    {
                        "action": "inspect_gripper",
                        "description": "Inspect gripper for worn or damaged jaws",
                    },
                    {
                        "action": "replace_gripper_jaw",
                        "description": "Replace worn gripper jaw with spare",
                    },
                    {
                        "action": "recalibrate_gripper",
                        "description": "Recalibrate grip force and position",
                    },
                    {
                        "action": "run_grasp_test",
                        "description": "Run 5 test grasp cycles before resuming production",
                    },
                ]
            }
        ),
        "resolution": {
            "actions_taken": "replaced gripper jaw, recalibrated, resumed",
            "outcome": "hardware_replaced",
            "notes": "Repeated misgrasp was due to worn gripper jaw reducing grip surface area.",
        },
    },
    {
        "anomaly_key": "S1.1_MISGRASP",
        "severity": "high",
        "device_id": "device-01",
        "summary": (
            "device=device-01 anomaly=S1.1_MISGRASP severity=high "
            "actions_taken=adjusted grasp pose, slowed approach, resolved "
            "outcome=resolved"
        ),
        "candidate_plan_json": json.dumps(
            {
                "steps": [
                    {"action": "pause_operation", "description": "Pause current grasp cycle"},
                    {
                        "action": "adjust_grasp_pose",
                        "description": "Offset grasp pose by 5mm in Z to clear edge",
                    },
                    {
                        "action": "reduce_approach_speed",
                        "description": "Reduce approach speed to 30% for next 5 cycles",
                    },
                    {
                        "action": "resume_nominal",
                        "description": "Resume production after successful test grasp",
                    },
                ]
            }
        ),
        "resolution": {
            "actions_taken": "adjusted grasp pose by 5mm in Z, slowed approach speed to 30%",
            "outcome": "resolved",
            "notes": "Workpiece placement drift caused repeated edge contact. Pose offset resolved it.",
        },
    },
]


async def main() -> None:
    config = CloudConfig()

    print(f"Connecting to DB: {config.db_path}")
    repo = SQLiteIncidentRepository(config.db_path)
    await repo.initialize()

    embedder = get_embedder(config.embedder_backend)

    vector_store: VectorStoreProtocol
    if config.vector_store_backend == "qdrant":
        print(f"Connecting to Qdrant: {config.qdrant_url}")
        qdrant = QdrantVectorStore(config.qdrant_url)
        await qdrant.initialize()
        vector_store = qdrant
    else:
        print("Using in-memory vector store")
        vector_store = InMemoryVectorStore()

    retrieval_store = RetrievalStore(
        embedder,
        vector_store,
        repo,
        outcome_boost=config.cloud_retrieval_outcome_boost,
        failure_penalty=config.cloud_retrieval_failure_penalty,
    )

    now = datetime.now(UTC)
    seeded = 0

    for i, spec in enumerate(_RESOLVED_INCIDENTS):
        incident_id = str(uuid.uuid4())
        resolution: dict[str, str] = dict(spec["resolution"])
        resolution["recorded_at"] = now.isoformat()

        record = IncidentRecord(
            incident_id=incident_id,
            escalation_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            device_id=str(spec["device_id"]),
            anomaly_key=str(spec["anomaly_key"]),
            anomaly_type="detected",
            severity=str(spec["severity"]),
            summary=str(spec["summary"]),
            raw_packet_json="{}",
            retrieved_incident_ids_json="[]",
            retrieved_incidents_json="[]",
            candidate_plan_json=str(spec["candidate_plan_json"]),
            validation_errors_json="[]",
            decision="needs_human",
            confidence=0.75,
            rationale="Escalated for operator review.",
            operator_resolution_json=json.dumps(resolution),
            source="seed",
            created_at=now,
        )

        await repo.save_incident(record)
        await retrieval_store.index_incident(record)
        seeded += 1
        print(
            f"  [{i + 1}/{len(_RESOLVED_INCIDENTS)}] seeded {spec['anomaly_key']} ({spec['severity']}) → {resolution.get('outcome', '')}"
        )

    print(f"\nDone. {seeded} incidents seeded into RAG.")
    print("The next cloud planning run will retrieve these as context.")


if __name__ == "__main__":
    asyncio.run(main())
