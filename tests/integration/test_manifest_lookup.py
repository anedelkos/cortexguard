import json
from pathlib import Path

from cortexguard.simulation.manifest_loader import ManifestLoader
from cortexguard.simulation.models.base_record import BaseFusedRecord
from cortexguard.simulation.streamers.local_streamer import LocalStreamer


def test_manifest_lookup_integration(tmp_path: Path) -> None:
    """Integration test: LocalStreamer correctly loads fused records via ManifestLoader + Trial."""

    # --- Create fake fused JSONL file ---
    fused_path = tmp_path / "trial_001_fused.jsonl"
    fused_records = [
        {"timestamp_ns": 1_000, "rgb_path": "rgb_1.png"},
        {"timestamp_ns": 2_000, "rgb_path": "rgb_2.png"},
    ]
    fused_path.write_text("\n".join(json.dumps(r) for r in fused_records))

    # --- Create manifest YAML file ---
    manifest_path = tmp_path / "manifest.yaml"
    manifest_yaml = f"""
    trials:
      - trial_id: trial_001
        subject: Alice
        food_type: Pasta
        trial_num: 1
        sensor_file: "sensor_1.csv"
        image_folder: "images/"
        fused_file: "{fused_path}"
    """
    manifest_path.write_text(manifest_yaml)

    # --- Load manifest using real loader ---
    loader = ManifestLoader(path=manifest_path)
    trials = loader.load()
    assert len(trials) == 1
    trial = loader.get_trial_by_id("trial_001")

    # --- Load fused records via LocalStreamer ---
    streamer = LocalStreamer[BaseFusedRecord](rate_hz=10.0)
    records = streamer.load_records_from_trial(trial)

    # --- Assertions ---
    assert len(records) == 2, "Expected two fused records loaded"
    assert all(isinstance(r, BaseFusedRecord) for r in records)
    assert records[0].timestamp_ns == 1_000
    assert records[1].timestamp_ns == 2_000
    assert trial.fused_file == fused_path

    # --- Smoke test streaming (no actual sleep) ---
    streamer.handle_record = lambda r: None  # mock handler
    streamer.rate_hz = 1000.0  # no perceptible delay
    streamer.stream(records)
