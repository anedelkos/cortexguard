"""Composite scoring for learning-to-rank RAG retrieval results."""

from __future__ import annotations

import json

from cortexguard.cloud.persistence.models import IncidentRecord

_FAILURE_DECISIONS = {"needs_human", "no_safe_plan"}


def score_incident(
    record: IncidentRecord,
    similarity: float,
    outcome_boost: float,
    failure_penalty: float,
) -> float:
    """Compute a composite score for ranking a retrieved incident.

    Args:
        record: The full incident record loaded from the repository.
        similarity: Raw embedding similarity score from the vector store.
        outcome_boost: Amount added when an operator marked the incident resolved.
        failure_penalty: Amount subtracted when the incident ended without resolution.

    Returns:
        Composite score clamped to ``[0.0, 1.0]``.
    """
    score = similarity

    if record.operator_resolution_json is not None:
        try:
            resolution = json.loads(record.operator_resolution_json)
            if resolution.get("outcome") == "resolved":
                score += outcome_boost
        except (json.JSONDecodeError, AttributeError):
            pass

    if (
        record.decision == "plan_ready"
        and record.confidence is not None
        and record.confidence >= 0.8
    ):
        score += outcome_boost * 0.5

    if record.decision in _FAILURE_DECISIONS and record.operator_resolution_json is None:
        score -= failure_penalty

    return max(0.0, min(1.0, score))
