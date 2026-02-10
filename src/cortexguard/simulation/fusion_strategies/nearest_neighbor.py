from __future__ import annotations

from typing import Any

import pandas as pd

from cortexguard.core.interfaces.fusion_strategy import BaseFusionStrategy


class NearestNeighborFusion(BaseFusionStrategy):
    """
    Fuses each image with the closest sensor readings within a small time window.
    """

    def __init__(self, window_size_s: float = 0.03):
        super().__init__(window_size_s)

    def fuse(
        self,
        sensor_df: pd.DataFrame,
        rgb_frames: list[dict[str, Any]],
        depth_frames: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        fused = []
        if "Time (sec)" not in sensor_df.columns:
            raise ValueError("Expected 'Time (sec)' column in sensor CSV")

        for frame in rgb_frames:
            frame_ts = frame["timestamp_ns"] / 1e9  # ns → sec
            window_mask = (sensor_df["Time (sec)"] >= frame_ts - self.window_size_s) & (
                sensor_df["Time (sec)"] <= frame_ts + self.window_size_s
            )

            numeric_cols = [
                "force_x",
                "force_y",
                "force_z",
                "torque_x",
                "torque_y",
                "torque_z",
                "pos_x",
                "pos_y",
                "pos_z",
            ]

            nearby = sensor_df.loc[window_mask, numeric_cols]

            if nearby.empty:
                continue

            averaged = nearby.mean(numeric_only=True).to_dict()

            fused.append(
                {
                    "timestamp_ns": int(frame["timestamp_ns"]),
                    "rgb_path": frame["path"],
                    "depth_path": frame.get("depth_path"),
                    **averaged,
                }
            )

        return fused
