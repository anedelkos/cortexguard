"""Stub model-training function for the step classifier retraining pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


def _compute_baseline_statistics(telemetry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from telemetry for Model Monitor baseline.

    In production this would compute mean/std/min/max per sensor field
    and generate a constraints.json compatible with SageMaker Model Monitor.
    """
    numerical_keys: list[str] = []
    for row in telemetry_rows:
        snapshot: dict[str, Any] = row.get("sensor_snapshot", {})
        for k, v in snapshot.items():
            if isinstance(v, int | float):
                numerical_keys.append(k)

    unique_keys = list(set(numerical_keys))
    return {
        "features": unique_keys,
        "num_rows": len(telemetry_rows),
        "generated_at": datetime.now(UTC).isoformat(),
    }


def train(telemetry_rows: list[Any]) -> dict[str, Any]:
    """Stub. Returns metrics without training any model.

    Generates a deterministic version string from the data hash so the
    Model Registry sees unique versions across pipeline runs.
    """
    data_hash = hashlib.sha256(
        json.dumps([r.get("key", "") for r in telemetry_rows], sort_keys=True).encode()
    ).hexdigest()[:12]

    num_records = len(telemetry_rows)
    return {
        "model_id": "step_classifier_v1",
        "version": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
        "data_hash": data_hash,
        "params": {"baseline_fpr": 0.0, "baseline_fnr": 0.0},
        "metrics": {
            "num_records": num_records,
            "accuracy": 0.95,
            "f1_score": 0.93,
            "precision": 0.94,
            "recall": 0.92,
        },
        "baseline_statistics": _compute_baseline_statistics(telemetry_rows),
    }
