from typing import Any

import pandas as pd

from cortexguard.core.interfaces.fusion_strategy import BaseFusionStrategy


class WindowedFusion(BaseFusionStrategy):
    """
    Fuses sensor and image data by grouping sensor samples
    within a symmetric time window around each RGB frame timestamp.

    Keeps full temporal information for downstream sequence/anomaly models.
    """

    def __init__(self, window_size_s: float = 0.05, min_samples: int = 1):
        """
        Args:
            window_size_s: total window size in seconds (±window_size_s/2 around each image)
            min_samples: skip windows with fewer than this many sensor readings
        """
        super().__init__(window_size_s)
        self.min_samples = min_samples

    def fuse(
        self,
        sensor_df: pd.DataFrame,
        rgb_frames: list[dict[str, Any]],
        depth_frames: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if sensor_df.empty or not rgb_frames:
            return []

        fused_records: list[dict[str, Any]] = []
        half_window_ns = int(self.window_size_s * 1e9 / 2)

        for frame in rgb_frames:
            t_img = int(frame["timestamp_ns"])

            # Select sensor samples within the time window
            mask = (sensor_df["timestamp_ns"] >= t_img - half_window_ns) & (
                sensor_df["timestamp_ns"] <= t_img + half_window_ns
            )
            window_df = sensor_df.loc[mask].copy()

            if len(window_df) < self.min_samples:
                continue

            # Normalize timestamps relative to image timestamp
            window_df["delta_t_ns"] = window_df["timestamp_ns"] - t_img

            # Match corresponding depth frame (nearest within window)
            depth_path = None
            if depth_frames:
                nearest = min(
                    depth_frames,
                    key=lambda d: abs(d["timestamp_ns"] - t_img),
                    default=None,
                )
                if nearest and abs(nearest["timestamp_ns"] - t_img) <= half_window_ns:
                    depth_path = nearest["path"]

            fused_records.append(
                {
                    "timestamp_ns": t_img,
                    "rgb_path": frame["path"],
                    "depth_path": depth_path,
                    "sensor_window": window_df.to_dict(orient="records"),
                    "window_size_s": self.window_size_s,
                    "n_samples": len(window_df),
                }
            )

        return fused_records
