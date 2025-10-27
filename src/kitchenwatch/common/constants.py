from pathlib import Path

DEFAULT_FULL_MANIFEST_PATH = Path("data/manifests/dataset_manifest.yaml")
DEFAULT_SAMPLE_MANIFEST_PATH = Path("data/manifests/sample_manifest.yaml")
DATA_DIR = Path("data")
SAMPLE_DATA_DIR = DATA_DIR / "sample"
SCHEMA_DIR = DATA_DIR / "schemas"
FUSED_EVENT_SCHEMA = SCHEMA_DIR / "fused_event_schema.json"
