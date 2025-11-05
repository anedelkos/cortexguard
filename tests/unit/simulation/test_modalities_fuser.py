import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from kitchenwatch.simulation.fusion_strategies.windowed import WindowedFusion
from kitchenwatch.simulation.modalities_fuser import ModalityFuser
from kitchenwatch.simulation.models.fused_record import FusedRecord
from kitchenwatch.simulation.models.trial import Trial
from kitchenwatch.simulation.models.windowed_fused_record import WindowedFusedRecord


@pytest.fixture
def dummy_trial(tmp_path: Path) -> Trial:
    """Minimal fake trial with one sensor file and image/depth dirs."""
    trial_dir = tmp_path / "trial_data"
    trial_dir.mkdir()

    sensor_csv = trial_dir / "sensors.csv"
    df = pd.DataFrame(
        {
            "timestamp_ns": [1000, 2000],
            "force_x": [1.0, 2.0],
            "force_y": [0.5, 0.6],
            "force_z": [0.1, 0.2],
            "torque_x": [0.01, 0.02],
            "torque_y": [0.03, 0.04],
            "torque_z": [0.05, 0.06],
            "pos_x": [0.0, 0.1],
            "pos_y": [0.0, 0.2],
            "pos_z": [0.0, 0.3],
        }
    )
    df.to_csv(sensor_csv, index=False)

    image_folder = trial_dir / "rgb"
    depth_folder = trial_dir / "depth"
    image_folder.mkdir()
    depth_folder.mkdir()

    # Simulate a single frame file with matching timestamps
    (image_folder / "1000_rgb.jpg").touch()
    (depth_folder / "1000_depth.png").touch()

    return Trial(
        trial_id="bagel_001",
        subject="subject_1",
        food_type="bagel",
        trial_num=1,
        sensor_file=str(sensor_csv),
        image_folder=str(image_folder),
        depth_folder=str(depth_folder),
        fusion_window=0.03,
        seed=42,
    )


@pytest.fixture
def mock_manifest_loader(dummy_trial: Trial) -> MagicMock:
    """Return a ManifestLoader mock with one fake trial."""
    mock_loader = MagicMock()
    mock_loader.trials = [dummy_trial]
    mock_loader.get_trial_by_id.return_value = dummy_trial
    mock_loader.save = MagicMock()
    return mock_loader


def test_fuse_trial_uses_strategy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_manifest_loader: MagicMock,
    dummy_trial: Trial,
) -> None:
    """fuse_trial should call the fusion strategy and return FusedRecords."""
    monkeypatch.setattr(
        "kitchenwatch.simulation.modalities_fuser.ManifestLoader", lambda _: mock_manifest_loader
    )

    mock_strategy = MagicMock()
    mock_strategy.fuse.return_value = [
        {
            "timestamp_ns": 1000,
            "rgb_path": "rgb/1000_rgb.jpg",
            "depth_path": "depth/1000_depth.png",
            "force_x": 1.0,
            "force_y": 2.0,
            "force_z": 3.0,
            "torque_x": 0.1,
            "torque_y": 0.2,
            "torque_z": 0.3,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": 0.0,
        }
    ]

    fuser = ModalityFuser(manifest_loader=mock_manifest_loader)
    fuser.loader = mock_manifest_loader

    fused_records = fuser.fuse_trial(dummy_trial, fusion_strategy=mock_strategy)

    mock_strategy.fuse.assert_called_once()
    assert len(fused_records) == 1
    assert isinstance(fused_records[0], FusedRecord)
    assert fused_records[0].force_x == 1.0


def test_fuse_trial_windowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dummy_trial: Trial,
    mock_manifest_loader: MagicMock,
) -> None:
    monkeypatch.setattr(
        "kitchenwatch.simulation.modalities_fuser.ManifestLoader", lambda _: mock_manifest_loader
    )

    fuser = ModalityFuser(manifest_loader=mock_manifest_loader)
    fuser.loader = mock_manifest_loader

    strategy = WindowedFusion(window_size_s=0.05)
    fused_records = fuser.fuse_trial(dummy_trial, fusion_strategy=strategy)

    assert len(fused_records) > 0
    assert isinstance(fused_records[0], WindowedFusedRecord)
    assert "sensor_window" in fused_records[0].model_dump()


def test_save_fused_records_writes_jsonl(tmp_path: Path, mock_manifest_loader: MagicMock) -> None:
    """save_fused_records should correctly write JSON lines."""
    fuser = ModalityFuser(manifest_loader=mock_manifest_loader)

    record = FusedRecord(
        timestamp_ns=123,
        rgb_path="rgb.jpg",
        depth_path="depth.png",
        force_x=1.0,
        force_y=2.0,
        force_z=3.0,
        torque_x=0.1,
        torque_y=0.2,
        torque_z=0.3,
        pos_x=0.0,
        pos_y=0.0,
        pos_z=0.0,
    )

    output_path = tmp_path / "out.jsonl"
    written = fuser.save_fused_records([record], output_path)

    assert written.exists()
    content = output_path.read_text().strip().splitlines()
    data = json.loads(content[0])
    assert data["timestamp_ns"] == 123
    assert data["force_x"] == 1.0


def test_load_frames_parses_timestamps(mock_manifest_loader: MagicMock, tmp_path: Path) -> None:
    """Ensure _load_frames correctly extracts timestamps from filenames."""
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()

    # Create some fake image files
    filenames = [
        "1637705108501665831_rgb.jpg",
        "1637705108535263300_rgb.jpg",
    ]
    for name in filenames:
        (rgb_dir / name).touch()

    fuser = ModalityFuser(manifest_loader=mock_manifest_loader)
    frames = fuser._load_frames(rgb_dir, "_rgb.jpg")

    assert len(frames) == 2
    assert frames[0]["timestamp_ns"] == 1637705108501665831
    assert frames[1]["timestamp_ns"] == 1637705108535263300
    assert all(Path(str(f["path"])).exists() for f in frames)


def test_load_frames_handles_bad_filenames(mock_manifest_loader: MagicMock, tmp_path: Path) -> None:
    """Ensure _load_frames skips or errors on invalid filenames."""
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()

    # Create a malformed filename
    bad_file = rgb_dir / "invalid_rgb.jpg"
    bad_file.touch()

    fuser = ModalityFuser(manifest_loader=mock_manifest_loader)

    with pytest.raises(ValueError):
        fuser._load_frames(rgb_dir, "_rgb.jpg")
