# demo/fuse_trial.py
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse raw trial data into a single dataset")
    parser.add_argument("--manifest", type=str, default="data/manifests/dataset_manifest.yaml")
    parser.add_argument("--output", type=str, default="data/fused/fused_dataset.jsonl")
    args = parser.parse_args()
    print(f"[stub] Would fuse dataset using {args.manifest} → {args.output}")


if __name__ == "__main__":
    main()
