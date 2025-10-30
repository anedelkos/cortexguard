# src/kitchenwatch/demo/fuse_trial.py
import argparse
from pathlib import Path

from kitchenwatch.simulation.fusion_strategies.windowed import WindowedFusion
from kitchenwatch.simulation.modalities_fuser import ModalityFuser


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse raw trial data into a single dataset")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/dataset_manifest.yaml"),
        help="Path to the manifest YAML file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fused/fused_dataset.jsonl"),
        help="Path to write the fused JSONL dataset",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fuser = ModalityFuser(manifest_path=args.manifest)

    strategy = WindowedFusion(window_size_s=0.1)
    # Here we assume you want to fuse all trials in the manifest
    for trial in fuser.loader.trials:
        output_path = fuser.fuse_and_save_trial(
            trial.trial_id, output_dir=args.output.parent, fusion_strategy=strategy
        )
        print(f"Fused trial {trial.trial_id} → {output_path}")


if __name__ == "__main__":
    main()
