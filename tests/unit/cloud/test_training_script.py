"""Unit tests for the training script stub."""

from __future__ import annotations

import tarfile
from pathlib import Path

from cortexguard.cloud.retraining.run import _write_model_artifact
from cortexguard.cloud.retraining.training_script import train


class TestTrainingScript:
    def test_train_returns_valid_output(self) -> None:
        result = train([])
        assert "model_id" in result
        assert "version" in result
        assert "params" in result
        assert "metrics" in result

    def test_train_counts_records(self) -> None:
        result = train([{"key": "step_1"}, {"key": "step_2"}, {"key": "step_3"}])
        assert result["metrics"]["num_records"] == 3

    def test_train_empty_telemetry(self) -> None:
        result = train([])
        assert result["metrics"]["num_records"] == 0
        assert result["params"]["baseline_fpr"] == 0.0
        assert result["params"]["baseline_fnr"] == 0.0

    def test_write_model_artifact_uses_sagemaker_code_layout(self, tmp_path: Path) -> None:
        artifact_path = _write_model_artifact(str(tmp_path), {"model_id": "x", "version": "1"})

        with tarfile.open(artifact_path, "r:gz") as tar:
            names = sorted(member.name for member in tar.getmembers())

        assert names == [
            "code/inference.py",
            "code/setup.py",
            "training_result.json",
        ]
