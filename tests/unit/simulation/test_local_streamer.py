# tests/unit/simulation/streamers/test_local_streamer.py
import json
import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from kitchenwatch.simulation.models.base_record import BaseFusedRecord
from kitchenwatch.simulation.models.fused_record import FusedRecord
from kitchenwatch.simulation.models.trial import Trial
from kitchenwatch.simulation.streamers.local_streamer import LocalStreamer


@pytest.fixture
def tmp_fused_file(tmp_path: Path) -> Path:
    """Create a dummy fused record JSONL file."""
    records = [
        {"timestamp_ns": 111, "rgb_path": "rgb1.jpg"},
        {"timestamp_ns": 222, "rgb_path": "rgb2.jpg"},
    ]
    fused_path = tmp_path / "trial1.jsonl"
    fused_path.write_text("\n".join(json.dumps(r) for r in records))
    return fused_path


@pytest.fixture
def trial(tmp_fused_file: Path) -> Trial:
    """Create a dummy Trial object referencing the fused file."""
    return Trial(
        trial_id="trial1",
        fused_file=tmp_fused_file,
        sensor_file=str(tmp_fused_file),
        image_folder=str(tmp_fused_file.parent),
    )


@pytest.fixture
def manifest(trial: Trial) -> list[Trial]:
    """Return a simple manifest list with one trial."""
    return [trial]


@pytest.fixture
def streamer() -> LocalStreamer[BaseFusedRecord]:
    """Create a LocalStreamer with a mock handler."""
    handler = Mock()
    return LocalStreamer(rate_hz=100.0, handle_record=handler)  # high Hz to make test fast


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------


def test_load_fused_records(monkeypatch: pytest.MonkeyPatch, tmp_fused_file: Path) -> None:
    """Verify that LocalStreamer.load_fused_records calls util function."""
    mock_records: list[FusedRecord] = [
        FusedRecord(
            timestamp_ns=1,
            rgb_path="rgb1.jpg",
            depth_path=None,
            force_x=0,
            force_y=0,
            force_z=0,
            torque_x=0,
            torque_y=0,
            torque_z=0,
            pos_x=0,
            pos_y=0,
            pos_z=0,
        ),
        FusedRecord(
            timestamp_ns=2,
            rgb_path="rgb2.jpg",
            depth_path=None,
            force_x=0,
            force_y=0,
            force_z=0,
            torque_x=0,
            torque_y=0,
            torque_z=0,
            pos_x=0,
            pos_y=0,
            pos_z=0,
        ),
    ]

    monkeypatch.setattr(
        "kitchenwatch.simulation.streamers.local_streamer.load_fused_records",
        lambda path: mock_records,
    )

    streamer: LocalStreamer[BaseFusedRecord] = LocalStreamer()
    result = streamer.load_fused_records(tmp_fused_file)

    assert result == mock_records


def test_load_records_from_trial_success(
    trial: Trial, tmp_fused_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure records are loaded when file exists."""
    mock_records: list[FusedRecord] = [
        FusedRecord(
            timestamp_ns=1,
            rgb_path="rgb1.jpg",
            depth_path=None,
            force_x=0,
            force_y=0,
            force_z=0,
            torque_x=0,
            torque_y=0,
            torque_z=0,
            pos_x=0,
            pos_y=0,
            pos_z=0,
        ),
    ]

    monkeypatch.setattr(
        "kitchenwatch.simulation.streamers.local_streamer.load_fused_records",
        lambda path: mock_records,
    )

    streamer: LocalStreamer[BaseFusedRecord] = LocalStreamer()
    result = streamer.load_records_from_trial(trial)
    assert result == mock_records


def test_load_records_from_trial_missing_file(tmp_path: Path) -> None:
    """Ensure FileNotFoundError is raised if trial's fused file doesn't exist."""
    missing_trial = Trial(
        trial_id="missing",
        fused_file=Path("/nonexistent.jsonl"),
        sensor_file="/nonexistent_sensor.jsonl",
        image_folder=str(tmp_path),
    )
    streamer: LocalStreamer[BaseFusedRecord] = LocalStreamer()
    with pytest.raises(FileNotFoundError):
        streamer.load_records_from_trial(missing_trial)


def test_load_records_by_id_success(
    streamer: LocalStreamer[BaseFusedRecord], manifest: list[Trial], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that load_records_by_id finds and loads the trial."""
    mock_records: list[FusedRecord] = [
        FusedRecord(
            timestamp_ns=1,
            rgb_path="rgb1.jpg",
            depth_path=None,
            force_x=0,
            force_y=0,
            force_z=0,
            torque_x=0,
            torque_y=0,
            torque_z=0,
            pos_x=0,
            pos_y=0,
            pos_z=0,
        ),
    ]
    monkeypatch.setattr(
        "kitchenwatch.simulation.streamers.local_streamer.LocalStreamer.load_records_from_trial",
        lambda self, trial: mock_records,
    )
    result = streamer.load_records_by_id("trial1", manifest)
    assert result == mock_records


def test_load_records_by_id_not_found(
    streamer: LocalStreamer[BaseFusedRecord], manifest: list[Trial]
) -> None:
    """Raise ValueError if trial ID not in manifest."""
    with pytest.raises(ValueError, match="Trial 'missing' not found"):
        streamer.load_records_by_id("missing", manifest)


# ---------------------------------------------------------------------------
# Streaming behavior tests
# ---------------------------------------------------------------------------


def test_stream_calls_handle_record_for_each_record(
    streamer: LocalStreamer[BaseFusedRecord],
) -> None:
    """Ensure handle_record is called for every record in the sequence."""
    records = [
        Mock(timestamp_ns=1),
        Mock(timestamp_ns=2),
        Mock(timestamp_ns=3),
    ]

    streamer.stream(records)
    assert isinstance(streamer.handle_record, Mock)
    assert streamer.handle_record.call_count == len(records)
    streamer.handle_record.assert_any_call(records[0])
    streamer.handle_record.assert_any_call(records[1])
    streamer.handle_record.assert_any_call(records[2])


def test_stream_rate_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify time.sleep is called with correct delay."""
    mock_sleep = Mock()
    monkeypatch.setattr(time, "sleep", mock_sleep)

    handler = Mock()
    streamer = LocalStreamer(rate_hz=20.0, handle_record=handler)
    records = [Mock(timestamp_ns=i) for i in range(5)]

    streamer.stream(records)

    expected_delay = pytest.approx(1.0 / 20.0, rel=1e-3)
    mock_sleep.assert_called_with(expected_delay)
    assert mock_sleep.call_count == len(records)


def test_stream_without_handler_uses_logger(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify logger.debug is used when no handle_record is provided."""
    monkeypatch.setattr(time, "sleep", lambda _: None)

    streamer: LocalStreamer[BaseFusedRecord] = LocalStreamer(rate_hz=100.0, handle_record=None)
    records = [Mock(timestamp_ns=10), Mock(timestamp_ns=20)]

    with caplog.at_level(logging.DEBUG):
        streamer.stream(records)

    assert "ts=10" in caplog.text
    assert "ts=20" in caplog.text


def test_stream_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify graceful handling of KeyboardInterrupt."""
    mock_sleep = Mock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(time, "sleep", mock_sleep)

    handler = Mock()
    streamer = LocalStreamer(rate_hz=10.0, handle_record=handler)
    records = [Mock(timestamp_ns=1), Mock(timestamp_ns=2)]

    # should not raise
    streamer.stream(records)

    handler.assert_called_once()  # interrupted after first record


def test_local_streamer_handles_empty_records(caplog: pytest.LogCaptureFixture) -> None:
    streamer: LocalStreamer[BaseFusedRecord] = LocalStreamer(rate_hz=100)
    with caplog.at_level(logging.DEBUG):
        streamer.stream([])

    assert "Completed streaming 0 records" in caplog.text


def test_local_streamer_handles_handler_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def bad_handler(_: Any) -> None:
        raise RuntimeError("oops")

    streamer: LocalStreamer[FusedRecord] = LocalStreamer(rate_hz=100, handle_record=bad_handler)

    with pytest.raises(RuntimeError):
        streamer.stream([FusedRecord(timestamp_ns=1, rgb_path="some_path")])
