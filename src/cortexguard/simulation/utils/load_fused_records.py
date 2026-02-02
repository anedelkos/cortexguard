import json
from pathlib import Path
from typing import TypeVar

from cortexguard.simulation.models.base_record import BaseFusedRecord
from cortexguard.simulation.models.fused_record import FusedRecord
from cortexguard.simulation.models.windowed_fused_record import WindowedFusedRecord

RecordT = TypeVar("RecordT", bound=BaseFusedRecord)


def load_fused_records[T: BaseFusedRecord](
    path: Path, record_type: type[T] | None = None
) -> list[T]:
    """
    Load fused or windowed fused records from a JSONL file.
    Automatically detects record type if not specified.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ValueError: if the file is empty or cannot detect record type.
        json.JSONDecodeError: if a line cannot be parsed as JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Fused data file not found: {path}")

    records: list[T] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON at line {line_no} in {path}: {e.msg}",
                        e.doc,
                        e.pos,
                    ) from e

                # Detect type from the first valid record
                if record_type is None:
                    if "sensor_window" in obj:
                        record_type = WindowedFusedRecord  # type: ignore[assignment]
                    else:
                        record_type = FusedRecord  # type: ignore[assignment]

                try:
                    records.append(record_type(**obj))
                except Exception as e:
                    if record_type is None:
                        raise RuntimeError(
                            "record_type should have been inferred before parsing records"
                        ) from e

                    raise ValueError(
                        f"Failed to parse record at line {line_no} as {record_type.__name__}: {e}"
                    ) from e

        if not records:
            raise ValueError(f"No valid records found in {path}")

    except Exception:
        # Let upstream handlers (e.g. LocalStreamer) log or wrap
        raise

    return records
