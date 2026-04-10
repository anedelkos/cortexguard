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
# C1 regression: label case mismatch
# EdgeFusion lowercases all labels before storing them in SceneGraph.
# _rule_human_hand_near_hazard must match the lowercased labels it actually
# receives, not the original mixed-case strings.
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
    """
    C1 regression: EdgeFusion stores lowercased labels.
    The rule must fire when given lowercased labels — if it only matches
    mixed-case strings it is permanently dead in production.
    """
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

    assert cmd.action == "E-STOP", (
        f"Expected E-STOP for hand label '{hand_label}' near hazard label '{hazard_label}'. "
        f"Got '{cmd.action}'. Likely cause: rule uses mixed-case string literals that don't "
        f"match the lowercased labels produced by EdgeFusion._to_scene_object."
    )


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
