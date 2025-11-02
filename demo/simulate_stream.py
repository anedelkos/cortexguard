#!/usr/bin/env python3
import argparse
import logging
import time
from pathlib import Path

from kitchenwatch.common.constants import (
    DEFAULT_FULL_MANIFEST_PATH,
    DEFAULT_FUSED_DATA_PATH,
)
from kitchenwatch.edge.local_edge_receiver import LocalEdgeReceiver
from kitchenwatch.simulation.manifest_loader import ManifestLoader
from kitchenwatch.simulation.models.base_record import BaseFusedRecord
from kitchenwatch.simulation.models.trial import Trial
from kitchenwatch.simulation.streamers.local_streamer import LocalStreamer
from kitchenwatch.simulation.utils.load_fused_records import load_fused_records

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate streaming of fused trial data to the local edge module."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_FUSED_DATA_PATH,
        help="Path to fused dataset directory OR single .jsonl file (default: data/fused/).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_FULL_MANIFEST_PATH,
        help="Path to dataset manifest (default: data/manifests/dataset_manifest.yaml).",
    )
    parser.add_argument(
        "--trial-id",
        type=str,
        default=None,
        help="Specific trial ID to stream (default: all trials in manifest or dataset).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Streaming rate multiplier (1.0 = real-time).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the stream (0 = infinite).",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Print fused records.",
    )
    args = parser.parse_args()

    logger.info(f"🗂  Dataset: {args.dataset}")
    logger.info(f"📜 Manifest: {args.manifest if args.manifest.exists() else '(not found)'}")
    logger.info(f"⏱️  Streaming rate: {args.rate:.2f}×")
    logger.info(f"🔁 Repeat count: {'∞' if args.repeat == 0 else args.repeat}")

    receiver = LocalEdgeReceiver(verbose=args.verbose)
    streamer = LocalStreamer(rate_hz=args.rate, handle_record=receiver.ingest)

    # --- Helper: stream a single trial file ---
    def stream_trial_file(file_path: Path, trial_id: str | None = None) -> None:
        if not file_path.exists():
            logger.error(f"Missing fused file: {file_path}")
            return
        name = trial_id or file_path.stem
        logger.info(f"▶️  Streaming trial: {name}")
        records: list[BaseFusedRecord] = load_fused_records(file_path)
        streamer.stream(records)

    # --- Manifest-based streaming ---
    if args.manifest.exists():
        loader = ManifestLoader(args.manifest)
        trials: list[Trial] = loader.load()

        def manifest_cycle() -> None:
            if args.trial_id:
                trial = next((t for t in trials if t.trial_id == args.trial_id), None)
                if not trial:
                    logger.error(f"Trial ID '{args.trial_id}' not found in manifest.")
                    return
                if not trial.fused_file:
                    logger.error(f"Trial '{trial.trial_id}' has no fused file path.")
                    return
                stream_trial_file(trial.fused_file, trial.trial_id)
            else:
                for trial in trials:
                    if not trial.fused_file:
                        logger.warning(f"Skipping trial {trial.trial_id} (no fused file).")
                        continue
                    stream_trial_file(trial.fused_file, trial.trial_id)

        # --- Repeat logic ---
        iteration = 0
        while args.repeat == 0 or iteration < args.repeat:
            iteration += 1
            logger.info(f"🔁 Manifest cycle {iteration}")
            manifest_cycle()
            if args.repeat != 0:
                logger.info(f"✅ Completed cycle {iteration}/{args.repeat}")
            time.sleep(0.5)

    # --- Dataset fallback mode ---
    else:
        logger.warning("Manifest not found — using dataset fallback mode.")
        dataset = args.dataset

        def dataset_cycle() -> None:
            if dataset.is_file():
                stream_trial_file(dataset)
            elif dataset.is_dir():
                files = sorted(dataset.glob("*.jsonl"))
                if args.trial_id:
                    files = [f for f in files if args.trial_id in f.stem]
                if not files:
                    logger.error(f"No .jsonl files found matching trial '{args.trial_id or '*'}'.")
                    return
                for f in files:
                    stream_trial_file(f)
            else:
                logger.error("Invalid dataset path; must be a file or directory.")

        # --- Repeat logic ---
        iteration = 0
        while args.repeat == 0 or iteration < args.repeat:
            iteration += 1
            logger.info(f"🔁 Dataset cycle {iteration}")
            dataset_cycle()
            if args.repeat != 0:
                logger.info(f"✅ Completed cycle {iteration}/{args.repeat}")
            time.sleep(0.5)

    logger.info("🎉 All streaming cycles completed successfully.")


if __name__ == "__main__":
    main()
