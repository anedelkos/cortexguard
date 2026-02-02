import json
from pathlib import Path

import pandas as pd
import pytest

from cortexguard.simulation.fusion_strategies.windowed import WindowedFusion
from cortexguard.simulation.modalities_fuser import ModalityFuser
from cortexguard.simulation.models.trial import Trial


@pytest.fixture
def tmp_trial(tmp_path: Path) -> tuple[Trial, Path]:
    """Create a minimal trial with fake sensor CSV and image/depth files."""
    # Sensor CSV
    sensor_csv = tmp_path / "sensors.csv"
    df = pd.DataFrame(
        {
            "timestamp_ns": [1_000_000_000, 1_050_000_000, 2_000_000_000],
            "force_x": [1.0, 1.1, 2.0],
            "force_y": [0.1, 0.2, 0.3],
            "force_z": [0.0, 0.1, 0.0],
            "torque_x": [0.01, 0.02, 0.03],
            "torque_y": [0.0, 0.01, 0.02],
            "torque_z": [0.0, 0.0, 0.01],
            "pos_x": [0.0, 0.1, 0.2],
            "pos_y": [0.0, 0.1, 0.2],
            "pos_z": [0.0, 0.0, 0.1],
        }
    )
    df.to_csv(sensor_csv, index=False)

    # Image/depth folders
    rgb_folder = tmp_path / "rgb"
    depth_folder = tmp_path / "depth"
    rgb_folder.mkdir()
    depth_folder.mkdir()

    (rgb_folder / "1000000000_rgb.jpg").touch()
    (rgb_folder / "2000000000_rgb.jpg").touch()
    (depth_folder / "1000000000_depth.png").touch()
    (depth_folder / "2000000000_depth.png").touch()

    trial = Trial(
        trial_id="test_001",
        subject="subject_1",
        food_type="bagel",
        trial_num=1,
        sensor_file=str(sensor_csv),
        image_folder=str(rgb_folder),
        depth_folder=str(depth_folder),
        fusion_window=0.1,
        seed=42,
    )
    return trial, tmp_path


@pytest.fixture
def manifest_file(tmp_trial: tuple[Trial, Path], tmp_path: Path) -> Path:
    """Write a minimal manifest YAML for the trial."""
    from cortexguard.simulation.manifest_loader import ManifestLoader

    trial, _ = tmp_trial
    manifest_path = tmp_path / "manifest.yaml"
    loader = ManifestLoader(manifest_path)
    loader.trials = [trial]
    loader.save(loader.trials)
    return manifest_path


def test_integration_fuse_and_save_trial(
    tmp_trial: tuple[Trial, Path], manifest_file: Path
) -> None:
    """Full end-to-end pipeline with WindowedFusion, writing JSONL and updating manifest."""
    trial, tmp_path = tmp_trial

    fuser = ModalityFuser(manifest_path=manifest_file)

    # Use WindowedFusion explicitly
    strategy = WindowedFusion(window_size_s=0.1, min_samples=1)
    out_path = fuser.fuse_and_save_trial(trial.trial_id, tmp_path, fusion_strategy=strategy)

    # JSONL file exists
    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) >= 1

    # Each line parses as a dict
    first_record = json.loads(lines[0])
    assert "timestamp_ns" in first_record
    assert "sensor_window" in first_record
    assert isinstance(first_record["sensor_window"], list)

    # Manifest updated
    from cortexguard.simulation.manifest_loader import ManifestLoader

    loader = ManifestLoader(manifest_file)
    loader.load()
    updated_trial = loader.get_trial_by_id(trial.trial_id)
    assert hasattr(updated_trial, "fused_file")
    assert updated_trial.fused_file == out_path
