import logging
from pathlib import Path

import pandas as pd

from kitchenwatch.common.constants import DEFAULT_FUSED_DATA_PATH
from kitchenwatch.core.interfaces.fusion_strategy import BaseFusionStrategy
from kitchenwatch.simulation.fusion_strategies.nearest_neighbor import NearestNeighborFusion
from kitchenwatch.simulation.manifest_loader import ManifestLoader
from kitchenwatch.simulation.models.fused_record import FusedRecord
from kitchenwatch.simulation.models.trial import Trial
from kitchenwatch.simulation.models.windowed_fused_record import WindowedFusedRecord


class ModalityFuser:
    """Fuse RGB, depth, and wrench/pose data streams into synchronized records."""

    def __init__(
        self,
        manifest_path: Path | None = None,
        manifest_loader: ManifestLoader | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Modality fuser for simulation trials. Fuses modalities into FusedRecords.

        Args:
            manifest_path: Optional path to the YAML manifest file for instantiating loader if not provided.
            manifest_loader: Optional manifest loader; either this or manifest_path should be provided.
            logger: Optional logger to use; defaults to module logger.
        """
        self.logger = logger or logging.getLogger(__name__)
        if manifest_loader is not None:
            self.loader = manifest_loader
        elif manifest_path:
            self.manifest_path = manifest_path
            self.loader = ManifestLoader(manifest_path)
            self.loader.load()
        else:
            raise ValueError("Either loader or manifest_path must be provided")

    def _load_frames(self, folder: Path, suffix: str) -> list[dict[str, int | str]]:
        if not folder.exists():
            return []
        return [
            {"timestamp_ns": int(f.name.replace(suffix, "")), "path": str(f)}
            for f in sorted(folder.glob(f"*{suffix}"))
        ]

    def fuse_trial(
        self, trial: Trial, fusion_strategy: BaseFusionStrategy | None = None
    ) -> list[WindowedFusedRecord | FusedRecord]:
        """Fuse modalities for a single trial."""
        sensor_df = pd.read_csv(trial.sensor_file)
        rgb_frames = self._load_frames(Path(trial.image_folder), "_rgb.jpg")
        depth_frames = (
            self._load_frames(Path(trial.depth_folder), "_depth.png") if trial.depth_folder else []
        )

        strategy = fusion_strategy or NearestNeighborFusion()
        fused_dicts = strategy.fuse(sensor_df, rgb_frames, depth_frames)

        record_type = WindowedFusedRecord if "sensor_window" in fused_dicts[0] else FusedRecord
        fused_records = [record_type(**record) for record in fused_dicts]

        self.logger.info(f"Fused {len(fused_records)} records for trial {trial.trial_id}")
        return fused_records

    def fuse_and_save_trial(
        self,
        trial_id: str,
        output_dir: Path = Path(DEFAULT_FUSED_DATA_PATH),
        fusion_strategy: BaseFusionStrategy | None = None,
    ) -> Path:
        """
        Fuse modalities for a given trial ID and save the result to disk.
        """
        trial = self.loader.get_trial_by_id(trial_id)
        if trial is None:
            raise ValueError(f"No trial found with id '{trial_id}'")

        fused_records = self.fuse_trial(trial, fusion_strategy=fusion_strategy)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{trial.trial_id}.jsonl"

        self.save_fused_records(fused_records, output_path)

        trial.fused_file = output_path
        self.loader.save(self.loader.trials)
        self.logger.info(f"Fused trial {trial.trial_id} → {output_path}")

        return output_path

    def save_fused_records(
        self,
        fused_records: list[FusedRecord | WindowedFusedRecord],
        output_path: str | Path,
    ) -> Path:
        """
        Save either FusedRecord or WindowedFusedRecord objects to a JSONL file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving {len(fused_records)} fused records → {output_path}")

        with output_path.open("w", encoding="utf-8") as f:
            for record in fused_records:
                f.write(record.model_dump_json() + "\n")

        return output_path

    def fuse_all(self, output_dir: Path | None = None) -> None:
        """Fuse all trials and update manifest."""
        output_dir = output_dir or Path(DEFAULT_FUSED_DATA_PATH)
        updated_trials = []
        for trial in self.loader.trials:
            _ = self.fuse_and_save_trial(trial.trial_id, output_dir)
            updated_trials.append(trial)

        # Save updated manifest with fused paths
        self.loader.save(updated_trials)
        self.logger.info(
            f"Updated manifest saved to {getattr(self.loader, 'path', self.manifest_path)}"
        )
