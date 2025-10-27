#!/usr/bin/env python3
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate streaming of fused trial data to the edge module"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/sample"),
        help="Path to the dataset folder to stream",
    )
    parser.add_argument(
        "--trial-id",
        type=str,
        default=None,
        help="Specific trial ID to stream (default: all trials)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Streaming rate multiplier (1.0 = real-time)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.03,
        help="Fusion window size in seconds",
    )
    args = parser.parse_args()

    # TODO: Implement the replay loop
    print(f"Streaming dataset from: {args.dataset}")
    if args.trial_id:
        print(f"Streaming only trial: {args.trial_id}")
    print(f"Streaming rate multiplier: {args.rate}")
    print(f"Fusion window: {args.window}s")


if __name__ == "__main__":
    main()
