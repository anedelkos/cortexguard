from pathlib import Path

import pytest

from cortexguard.simulation.models.base_record import BaseFusedRecord
from cortexguard.simulation.utils.load_fused_records import load_fused_records


def test_load_fused_records_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_fused_records(missing)


def test_load_fused_records_invalid_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json")
    with pytest.raises(ValueError):
        load_fused_records(bad_file)


def test_load_fused_records_valid_json(tmp_path: Path) -> None:
    file = tmp_path / "good.json"
    # Must be one JSON object per line
    file.write_text(
        '{"timestamp_ns": 1, "temp": 25, "rgb_path": "path_to_image"}\n'
        '{"timestamp_ns": 2, "temp": 27, "rgb_path": "path_to_image2"}\n'
    )

    result: list[BaseFusedRecord] = load_fused_records(file)
    assert len(result) == 2
    assert result[0].timestamp_ns == 1
    assert result[1].timestamp_ns == 2
