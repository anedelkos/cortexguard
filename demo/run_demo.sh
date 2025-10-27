#!/bin/bash

echo "Fusing trial bagel_001..."
python src/kitchenwatch/simulation/fuse_modalities.py \
  --manifest data/manifests/dataset_manifest.yaml \
  --trial-id bagel_001 \
  --out data/fused/bagel_001.jsonl

echo "Streaming fused trial..."
python src/kitchenwatch/simulation/simulate_stream.py \
  --input data/fused/bagel_001.jsonl \
  --endpoint http://localhost:8080/api/v1/sensor-chunk \
  --mode http
