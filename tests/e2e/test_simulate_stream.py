import json
import logging
from pathlib import Path
from typing import Any

import pytest

from kitchenwatch.edge.local_edge_receiver import LocalEdgeReceiver
from kitchenwatch.simulation.manifest_loader import ManifestLoader
from kitchenwatch.simulation.streamers.local_streamer import LocalStreamer


@pytest.fixture
def dummy_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake manifest + fused JSONL file in a temporary dataset."""
    fused_path = tmp_path / "trial_001.jsonl"
    fused_record = {"timestamp_ns": 1, "rgb_path": "rgb.jpg"}
    with fused_path.open("w") as f:
        f.write(json.dumps(fused_record) + "\n")

    manifest_path = tmp_path / "manifest.yaml"
    manifest_yaml = f"""
trials:
  - trial_id: trial_001
    fused_file: {fused_path}
    sensor_file: sensor.csv
    image_folder: images
"""
    manifest_path.write_text(manifest_yaml)
    return manifest_path, fused_path


def test_simulate_stream_end_to_end(
    dummy_dataset: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    """End-to-end test: manifest + streamer + receiver cooperate correctly."""
    manifest_path, fused_path = dummy_dataset
    caplog.set_level(logging.INFO)

    # --- Load manifest ---
    loader = ManifestLoader(manifest_path)
    trials = loader.load()
    assert len(trials) == 1
    trial = trials[0]
    assert trial.fused_file is not None
    assert trial.fused_file.exists()

    # --- Dummy receiver to capture streamed records ---
    received = []

    class DummyReceiver(LocalEdgeReceiver):
        def ingest(self, record: Any) -> None:
            received.append(record)

    receiver = DummyReceiver()

    # --- Run streamer ---
    streamer = LocalStreamer(rate_hz=1000.0, handle_record=receiver.ingest)
    records = streamer.load_records_from_trial(trial)
    streamer.stream(records)

    # --- Verify behavior ---
    assert len(received) == 1
    rec = received[0]
    assert rec.timestamp_ns == 1
    assert rec.rgb_path == "rgb.jpg"
    assert "Completed streaming" in caplog.text
    assert "Loaded 1 fused records" in caplog.text
    assert "Starting local stream" in caplog.text
