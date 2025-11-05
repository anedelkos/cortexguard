from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseFusionStrategy(ABC):
    """
    Abstract base for fusion strategies.

    Concrete implementations should define how sensor data and image/depth
    frames are temporally aligned into fused records.
    """

    def __init__(self, window_size_s: float):
        """
        Args:
            window_size_s: Temporal window (in seconds) to include
                multiple sensor readings around each frame timestamp.
        """
        self.window_size_s = window_size_s

    @abstractmethod
    def fuse(
        self,
        sensor_df: pd.DataFrame,
        rgb_frames: list[dict[str, Any]],
        depth_frames: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fuse sensor data and image/depth frames into aligned records.

        Implementations may:
          - Use nearest-neighbor matching (single sample per frame)
          - Use windowed aggregation (multiple sensor samples per frame)
          - Apply interpolation, smoothing, or statistical reduction

        Returns:
            A list of fused records as dicts, suitable for serialization.
        """
        raise NotImplementedError
