import pandas as pd
import pytest

from kitchenwatch.simulation.fusion_strategies.windowed import WindowedFusion
from kitchenwatch.simulation.models.windowed_fused_record import SensorReading, WindowedFusedRecord


@pytest.fixture
def sensor_df() -> pd.DataFrame:
    """Create fake sensor data with timestamps and forces."""
    data = {
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
    return pd.DataFrame(data)


@pytest.fixture
def rgb_frames() -> list[dict[str, str | int]]:
    return [
        {"timestamp_ns": 1_000_000_000, "path": "rgb/1.jpg"},
        {"timestamp_ns": 2_000_000_000, "path": "rgb/2.jpg"},
    ]


@pytest.fixture
def depth_frames() -> list[dict[str, str | int]]:
    return [
        {"timestamp_ns": 1_000_000_000, "path": "depth/1.png"},
        {"timestamp_ns": 2_000_000_000, "path": "depth/2.png"},
    ]


def test_basic_windowed_fusion(
    sensor_df: pd.DataFrame,
    rgb_frames: list[dict[str, str | int]],
    depth_frames: list[dict[str, str | int]],
) -> None:
    strategy = WindowedFusion(window_size_s=0.1, min_samples=1)
    fused = strategy.fuse(sensor_df, rgb_frames, depth_frames)

    assert len(fused) == 2  # one per image
    for record in fused:
        assert isinstance(record["sensor_window"], list)
        assert all("timestamp_ns" in s for s in record["sensor_window"])
        # Ensure depth matched
        assert record["depth_path"] is not None
        # Check n_samples matches sensor_window length
        assert record["n_samples"] == len(record["sensor_window"])
        # Check window_size_s is recorded
        assert record["window_size_s"] == 0.1


def test_delta_t_normalization(
    sensor_df: pd.DataFrame, rgb_frames: list[dict[str, str | int]]
) -> None:
    strategy = WindowedFusion(window_size_s=0.1)
    fused = strategy.fuse(sensor_df, rgb_frames)

    for record in fused:
        for s in record["sensor_window"]:
            delta_t = s["timestamp_ns"] - record["timestamp_ns"]
            assert isinstance(delta_t, int)


def test_min_samples_filter(
    sensor_df: pd.DataFrame, rgb_frames: list[dict[str, str | int]]
) -> None:
    strategy = WindowedFusion(window_size_s=0.01, min_samples=2)
    fused = strategy.fuse(sensor_df, rgb_frames)
    # Window is tiny; first image should skip
    assert all(r["n_samples"] >= 2 for r in fused)


def test_sensor_window_model(
    sensor_df: pd.DataFrame, rgb_frames: list[dict[str, str | int]]
) -> None:
    """Ensure fused dict can be instantiated as WindowedFusedRecord with SensorReading objects."""
    strategy = WindowedFusion(window_size_s=0.1)
    fused_dicts = strategy.fuse(sensor_df, rgb_frames)
    for d in fused_dicts:
        sensor_objs = [SensorReading(**s) for s in d["sensor_window"]]
        wf = WindowedFusedRecord(
            timestamp_ns=d["timestamp_ns"],
            rgb_path=d["rgb_path"],
            depth_path=d.get("depth_path"),
            window_size_s=d["window_size_s"],
            n_samples=d["n_samples"],
            sensor_window=sensor_objs,
        )
        assert isinstance(wf, WindowedFusedRecord)
        assert len(wf.sensor_window) == wf.n_samples
