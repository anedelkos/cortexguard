# CortexGuard Data Directory

This directory contains all dataset-related files and utilities used in the CortexGuard simulation environment.

## Overview

CortexGuard datasets are organized around *trials* — each trial corresponds to a single data collection session with synchronized sensor modalities (e.g. RGB, depth, force).
To simplify local setup, a downloader script automates fetching and extracting datasets from the canonical storage.

## Downloader Script

The `download_dataset.py` script reads a YAML manifest describing available trials and their download URLs.
It will download and extract all missing archives into the appropriate local folders.

### Usage

```bash
# Download full dataset
PYTHONPATH=src python data/download_dataset.py

# Optionally specify a manifest and/or custom target directory
PYTHONPATH=src python data/download_dataset.py \
  --manifest data/manifests/full_dataset_manifest.yaml \
  --data-dir /path/to/local/data
