"""Unit tests for SafetyAgent safety rules."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cortexguard.edge.models.scene_graph import SceneGraph, SceneObject, SceneRelationship
from cortexguard.edge.models.state_estimate import StateEstimate
from cortexguard.edge.safety_agent import SafetyAgent


def _make_state() -> StateEstimate:
    return MagicMock(spec=StateEstimate)


def _make_agent() -> SafetyAgent:
    bb = MagicMock()
    bb.get_scene_graph = AsyncMock(return_value=None)
    return SafetyAgent(blackboard=bb)


# ---------------------------------------------------------------------------
# Regression: label case mismatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hand_label, hazard_label",
    [
        ("human_hand", "blade"),
        ("human_hand", "grill"),
    ],
)
def test_rule_human_hand_near_hazard_fires_with_lowercased_labels(
    hand_label: str, hazard_label: str
) -> None:
    """Rule must fire with lowercased labels as produced by EdgeFusion._to_scene_object."""
    agent = _make_agent()
    state = _make_state()

    scene = SceneGraph(
        objects=[
            SceneObject(id="hand1", label=hand_label, location_2d=None, pose_3d=None),
            SceneObject(id="hazard1", label=hazard_label, location_2d=None, pose_3d=None),
        ],
        relationships=[
            SceneRelationship(source_id="hand1", relationship="near", target_id="hazard1"),
        ],
    )

    cmd = agent._rule_human_hand_near_hazard(state, scene)

    assert cmd.action == "E-STOP"


def test_rule_human_hand_near_hazard_nominal_when_no_hazard() -> None:
    """No relationship to a hazard object → NOMINAL."""
    agent = _make_agent()
    state = _make_state()

    scene = SceneGraph(
        objects=[
            SceneObject(id="hand1", label="human_hand", location_2d=None, pose_3d=None),
            SceneObject(id="table1", label="table", location_2d=None, pose_3d=None),
        ],
        relationships=[
            SceneRelationship(source_id="hand1", relationship="near", target_id="table1"),
        ],
    )

    cmd = agent._rule_human_hand_near_hazard(state, scene)
    assert cmd.action == "NOMINAL"


def test_rule_human_hand_near_hazard_nominal_when_no_scene() -> None:
    """None scene → NOMINAL (no crash)."""
    agent = _make_agent()
    state = _make_state()
    cmd = agent._rule_human_hand_near_hazard(state, None)
    assert cmd.action == "NOMINAL"


def test_rule_human_hand_near_hazard_fires_on_reverse_relationship() -> None:
    """Hazard as source, hand as target → still E-STOP."""
    agent = _make_agent()
    state = _make_state()

    scene = SceneGraph(
        objects=[
            SceneObject(id="hand1", label="human_hand", location_2d=None, pose_3d=None),
            SceneObject(id="blade1", label="blade", location_2d=None, pose_3d=None),
        ],
        relationships=[
            SceneRelationship(source_id="blade1", relationship="near", target_id="hand1"),
        ],
    )

    cmd = agent._rule_human_hand_near_hazard(state, scene)
    assert cmd.action == "E-STOP"


def test_rule_human_hand_near_hazard_fires_on_safety_critical_property() -> None:
    """Object with safety_critical=True property should still trigger E-STOP."""
    agent = _make_agent()
    state = _make_state()

    scene = SceneGraph(
        objects=[
            SceneObject(id="hand1", label="human_hand", location_2d=None, pose_3d=None),
            SceneObject(
                id="custom1",
                label="lathe",
                location_2d=None,
                pose_3d=None,
                properties={"safety_critical": True},
            ),
        ],
        relationships=[
            SceneRelationship(source_id="hand1", relationship="touching", target_id="custom1"),
        ],
    )

    cmd = agent._rule_human_hand_near_hazard(state, scene)
    assert cmd.action == "E-STOP"


# ---------------------------------------------------------------------------
# _is_near proximity semantics
# ---------------------------------------------------------------------------


def test_is_near_returns_false_for_objects_far_from_robot() -> None:
    """Two objects both far from the robot must not be classified as near each other."""
    from cortexguard.edge.edge_fusion import _is_near
    from cortexguard.edge.models.scene_graph import SceneObject

    far_a = SceneObject(
        id="a",
        label="object_a",
        location_2d=None,
        pose_3d=None,
        properties={"distance_m": 10.0},
    )
    far_b = SceneObject(
        id="b",
        label="object_b",
        location_2d=None,
        pose_3d=None,
        properties={"distance_m": 10.2},
    )

    result = _is_near(far_a, far_b)

    assert result is False


# ---------------------------------------------------------------------------
# SafetyAgent rule coverage
# ---------------------------------------------------------------------------


def test_rule_critical_system_state_fires_on_critical_symbolic_state() -> None:
    """CRITICAL keyword in symbolic_system_state triggers E-STOP."""
    from datetime import UTC, datetime

    from cortexguard.edge.models.state_estimate import StateEstimate

    agent = _make_agent()
    state = StateEstimate(
        timestamp=datetime.now(UTC),
        label="test",
        confidence=1.0,
        symbolic_system_state={"motor_driver": "CRITICAL_FAULT"},
    )

    cmd = agent._rule_critical_system_state(state, None)
    assert cmd.action == "E-STOP"


def test_rule_critical_system_state_nominal_when_no_critical() -> None:
    """No CRITICAL keyword → NOMINAL."""
    from datetime import UTC, datetime

    from cortexguard.edge.models.state_estimate import StateEstimate

    agent = _make_agent()
    state = StateEstimate(
        timestamp=datetime.now(UTC),
        label="test",
        confidence=1.0,
        symbolic_system_state={"motor_driver": "nominal"},
    )

    cmd = agent._rule_critical_system_state(state, None)
    assert cmd.action == "NOMINAL"


def test_rule_immediate_human_proximity_fires_within_radius() -> None:
    """Human within safety radius in observations → E-STOP."""
    from datetime import UTC, datetime

    from cortexguard.edge.models.state_estimate import StateEstimate

    agent = _make_agent()
    state = StateEstimate(
        timestamp=datetime.now(UTC),
        label="test",
        confidence=1.0,
        symbolic_system_state={},
        observations={"vision_nearest_human_m": 0.3},  # within 0.5m radius
    )

    cmd = agent._rule_immediate_human_proximity(state, None)
    assert cmd.action == "E-STOP"


def test_rule_immediate_human_proximity_nominal_when_far() -> None:
    """Human beyond safety radius → NOMINAL."""
    from datetime import UTC, datetime

    from cortexguard.edge.models.state_estimate import StateEstimate

    agent = _make_agent()
    state = StateEstimate(
        timestamp=datetime.now(UTC),
        label="test",
        confidence=1.0,
        symbolic_system_state={},
        observations={"vision_nearest_human_m": 2.0},
    )

    cmd = agent._rule_immediate_human_proximity(state, None)
    assert cmd.action == "NOMINAL"


def test_rule_immediate_human_proximity_fires_from_scene_graph() -> None:
    """Human object in scene graph within radius → E-STOP (fallback path)."""
    from datetime import UTC, datetime

    from cortexguard.edge.models.scene_graph import SceneGraph, SceneObject
    from cortexguard.edge.models.state_estimate import StateEstimate

    agent = _make_agent()
    state = StateEstimate(
        timestamp=datetime.now(UTC),
        label="test",
        confidence=1.0,
        symbolic_system_state={},
    )
    scene = SceneGraph(
        objects=[
            SceneObject(
                id="p1",
                label="person",
                location_2d=None,
                pose_3d=None,
                properties={"distance_m": 0.2},
            )
        ]
    )

    cmd = agent._rule_immediate_human_proximity(state, scene)
    assert cmd.action == "E-STOP"


def test_rule_detector_short_circuit_fires_on_explicit_stop_key() -> None:
    """Anomaly with key in EXPLICIT_STOP_KEYS → E-STOP."""
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, MagicMock

    from cortexguard.edge.models.anomaly_event import AnomalyEvent, AnomalySeverity

    bb = MagicMock()
    bb.get_scene_graph = AsyncMock(return_value=None)
    agent = SafetyAgent(blackboard=bb)

    anomaly = AnomalyEvent(
        id="a1",
        key="HUMAN_PROXIMITY_VIOLATION",
        severity=AnomalySeverity.HIGH,
        score=1.0,
        timestamp=datetime.now(UTC),
        metadata={},
        contributing_detectors=["VisionSafetyDetector"],
    )
    agent._anomalies = {"a1": anomaly}
    state = _make_state()

    cmd = agent._rule_detector_short_circuit(state, None)
    assert cmd.action == "E-STOP"


def test_rule_detector_short_circuit_nominal_when_no_anomalies() -> None:
    """Empty anomaly dict → NOMINAL."""
    from unittest.mock import AsyncMock, MagicMock

    bb = MagicMock()
    bb.get_scene_graph = AsyncMock(return_value=None)
    agent = SafetyAgent(blackboard=bb)
    agent._anomalies = {}
    state = _make_state()

    cmd = agent._rule_detector_short_circuit(state, None)
    assert cmd.action == "NOMINAL"


# ---------------------------------------------------------------------------
# _rule_detector_short_circuit: additional code paths
# ---------------------------------------------------------------------------


def test_rule_detector_short_circuit_fires_on_string_severity_high() -> None:
    """Non-explicit-key anomaly with string severity "HIGH" must fire E-STOP via the severity branch."""
    from typing import Any

    agent = _make_agent()
    dict_anomalies: dict[str, Any] = {
        "a1": {
            "key": "sensor_spike",  # NOT in EXPLICIT_STOP_KEYS
            "severity": "HIGH",  # string, not AnomalySeverity enum
            "score": 0.9,
        }
    }
    agent._anomalies = dict_anomalies  # type: ignore[assignment]
    cmd = agent._rule_detector_short_circuit(_make_state(), None)
    assert cmd.action == "E-STOP"


def test_rule_detector_short_circuit_fires_on_dict_anomaly_with_explicit_key() -> None:
    """Dict-form anomaly with an explicit stop key must trigger E-STOP the same as an AnomalyEvent."""
    from typing import Any

    agent = _make_agent()
    dict_anomalies: dict[str, Any] = {
        "a1": {
            "key": "HUMAN_PROXIMITY_VIOLATION",  # in EXPLICIT_STOP_KEYS
            "severity": "high",
            "score": 1.0,
        }
    }
    agent._anomalies = dict_anomalies  # type: ignore[assignment]
    cmd = agent._rule_detector_short_circuit(_make_state(), None)
    assert cmd.action == "E-STOP"
