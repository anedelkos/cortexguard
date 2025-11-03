from pathlib import Path

# --------------------------------------------------------------------
# Resolve paths relative to the project root
# --------------------------------------------------------------------
# This file lives at: src/kitchenwatch/common/constants.py
# So .resolve().parents[2] -> project root (one up from `src/`)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------
# Data and schema directories
# --------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
SCHEMA_DIR = DATA_DIR / "schemas"
DEFAULT_FUSED_DATA_PATH = DATA_DIR / "fused"
DEFAULT_RAW_DATA_PATH = DATA_DIR / "raw"

# --------------------------------------------------------------------
# File paths
# --------------------------------------------------------------------
DEFAULT_FULL_MANIFEST_PATH = DATA_DIR / "manifests" / "full_dataset_manifest.yaml"
DEFAULT_SAMPLE_MANIFEST_PATH = DATA_DIR / "manifests" / "sample_manifest.yaml"
FUSED_EVENT_SCHEMA = SCHEMA_DIR / "fused_event_schema.json"

# --------------------------------------------------------------------
# Optional: sanity check (useful during debugging)
# --------------------------------------------------------------------
if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"FUSED_EVENT_SCHEMA exists: {FUSED_EVENT_SCHEMA.exists()}")
