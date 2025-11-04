#!/bin/bash

## Run demo with sample data from data/sample

echo "Fusing trial bagel_001..."
PYTHONPATH=src python demo/fuse_trial.py --manifest data/manifests/sample_dataset_manifest.yaml


echo "Streaming fused trial..."
PYTHONPATH=src python demo/simulate_stream.py --manifest data/manifests/sample_dataset_manifest.yaml
