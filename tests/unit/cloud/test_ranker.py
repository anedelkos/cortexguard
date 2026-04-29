"""Unit tests for the learning-to-rank scoring function."""

from __future__ import annotations

import json

import pytest

from cortexguard.cloud.retrieval.ranker import score_incident
from tests.unit.cloud.factories import make_incident


def test_resolved_operator_resolution_boosts_score() -> None:
    resolution = json.dumps({"outcome": "resolved"})
    record = make_incident(operator_resolution_json=resolution)
    base = make_incident()

    score_with_boost = score_incident(
        record, similarity=0.5, outcome_boost=0.2, failure_penalty=0.1
    )
    score_without_boost = score_incident(
        base, similarity=0.5, outcome_boost=0.2, failure_penalty=0.1
    )

    assert score_with_boost > score_without_boost


def test_needs_human_no_resolution_penalises_score() -> None:
    record = make_incident(decision="needs_human", operator_resolution_json=None)
    base = make_incident(decision="plan_ready", operator_resolution_json=None)

    penalised = score_incident(record, similarity=0.5, outcome_boost=0.2, failure_penalty=0.1)
    baseline = score_incident(base, similarity=0.5, outcome_boost=0.2, failure_penalty=0.1)

    assert penalised < baseline


def test_high_confidence_plan_ready_mild_boost() -> None:
    record = make_incident(decision="plan_ready", confidence=0.9)
    base = make_incident(decision="plan_ready", confidence=0.4)

    boosted = score_incident(record, similarity=0.5, outcome_boost=0.2, failure_penalty=0.1)
    not_boosted = score_incident(base, similarity=0.5, outcome_boost=0.2, failure_penalty=0.1)

    assert boosted > not_boosted
    assert boosted > 0.5


def test_score_clamped_to_unit_interval() -> None:
    resolution = json.dumps({"outcome": "resolved"})
    record = make_incident(
        decision="plan_ready",
        confidence=0.9,
        operator_resolution_json=resolution,
    )
    result = score_incident(record, similarity=0.95, outcome_boost=0.2, failure_penalty=0.1)
    assert result <= 1.0
    assert result >= 0.0


def test_malformed_operator_resolution_json_no_boost_no_crash() -> None:
    record = make_incident(operator_resolution_json="not-valid-json")
    result = score_incident(record, similarity=0.5, outcome_boost=0.2, failure_penalty=0.1)
    assert result == pytest.approx(0.5)


def test_no_safe_plan_no_resolution_penalises_score() -> None:
    record = make_incident(decision="no_safe_plan", operator_resolution_json=None)
    base = make_incident(decision="plan_ready", operator_resolution_json=None)

    penalised = score_incident(record, similarity=0.5, outcome_boost=0.2, failure_penalty=0.1)
    baseline = score_incident(base, similarity=0.5, outcome_boost=0.2, failure_penalty=0.1)

    assert penalised < baseline
