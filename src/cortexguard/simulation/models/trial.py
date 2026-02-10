from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, HttpUrl


class DownloadFile(BaseModel):
    """Metadata for a downloadable dataset component (e.g., RGB, depth, wrenches)."""

    url: HttpUrl
    target_dir: Path


class Trial(BaseModel):
    """Definition of a CortexGuard simulation trial."""

    # Identification
    trial_id: str
    subject: str | None = None
    food_type: str | None = None
    trial_num: int | None = None

    # Local file references
    sensor_file: str
    image_folder: str
    depth_folder: str | None = None

    # Processing and simulation parameters
    fusion_window: float = 0.03
    anomaly_scenario: str | None = None
    seed: int | None = None

    # Output (filled by fuser)
    fused_file: Path | None = None

    # Optional remote dataset download info
    download_files: dict[str, DownloadFile] | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # Helpers
    def has_remote_assets(self) -> bool:
        """Return True if this trial includes downloadable components."""
        return bool(self.download_files)
