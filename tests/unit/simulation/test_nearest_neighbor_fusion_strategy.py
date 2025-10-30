import pandas as pd

from kitchenwatch.simulation.fusion_strategies.nearest_neighbor import NearestNeighborFusion


def test_nearest_neighbor_fusion_basic() -> None:
    """Test that the first 2 sensor readings get averaged and mapped to image 1,
    and the 3rd one to image 2.
    """
    sensor_df = pd.DataFrame(
        {
            "Time (sec)": [1.0, 1.05, 2.0],
            "Force X (N)": [10, 20, 30],
            "Force Y (N)": [0, 5, 10],
            "Force Z (N)": [1, 2, 3],
            "Torque X (Nm)": [0.1, 0.2, 0.3],
            "Torque Y (Nm)": [0.01, 0.02, 0.03],
            "Torque Z (Nm)": [0.001, 0.002, 0.003],
            "Forktip Pose X (m)": [0, 0.1, 0.2],
            "Forktip Pose Y (m)": [0, 0.2, 0.4],
            "Forktip Pose Z (m)": [0, 0.3, 0.6],
            "Forktip Pose Roll (rad)": [0, 0.01, 0.02],
            "Forktip Pose Pitch (rad)": [0, 0.02, 0.04],
            "Forktip Pose Yaw (rad)": [0, 0.03, 0.06],
        }
    )

    rgb_frames = [
        {"timestamp_ns": 1_000_000_000, "path": "frame1.jpg"},  # 1.0 sec
        {"timestamp_ns": 2_000_000_000, "path": "frame2.jpg"},  # 2.0 sec
    ]

    strategy = NearestNeighborFusion(window_size_s=0.1)
    fused = strategy.fuse(sensor_df, rgb_frames)

    assert len(fused) == 2
    assert fused[0]["Force X (N)"] == 15  # avg of 10, 20 near 1.0 sec
    assert fused[1]["Force X (N)"] == 30


def test_nearest_neighbor_fusion_complex() -> None:
    """Test multiple frames, with some sensor readings outside the window."""
    sensor_df = pd.DataFrame(
        {
            "Time (sec)": [1.02, 1.05, 2.07, 3.11],
            "Force X (N)": [10, 20, 30, 40],
            "Force Y (N)": [0, 5, 10, 20],
            "Force Z (N)": [1, 2, 3, 4],
            "Torque X (Nm)": [0.1, 0.2, 0.3, 0.4],
            "Torque Y (Nm)": [0.01, 0.02, 0.03, 0.04],
            "Torque Z (Nm)": [0.001, 0.002, 0.003, 0.004],
            "Forktip Pose X (m)": [0, 0.1, 0.2, 0.3],
            "Forktip Pose Y (m)": [0, 0.2, 0.4, 0.6],
            "Forktip Pose Z (m)": [0, 0.3, 0.6, 0.9],
            "Forktip Pose Roll (rad)": [0, 0.01, 0.02, 0.03],
            "Forktip Pose Pitch (rad)": [0, 0.02, 0.04, 0.06],
            "Forktip Pose Yaw (rad)": [0, 0.03, 0.06, 0.09],
        }
    )

    rgb_frames = [
        {"timestamp_ns": 1_000_000_000, "path": "frame1.jpg"},
        {"timestamp_ns": 2_000_000_000, "path": "frame2.jpg"},
        {"timestamp_ns": 3_000_000_000, "path": "frame3.jpg"},
    ]

    strategy = NearestNeighborFusion(window_size_s=0.1)
    fused = strategy.fuse(sensor_df, rgb_frames)

    # Only the first two frames should have fused sensor readings
    assert len(fused) == 2
    assert fused[0]["Force X (N)"] == 15  # avg of 10, 20 near 1.0 sec
    assert fused[1]["Force X (N)"] == 30  # single reading near 2.0 sec
