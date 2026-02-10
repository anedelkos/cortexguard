from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

from cortexguard.core.interfaces.base_streamer import BaseStreamer
from cortexguard.simulation.models.base_record import BaseFusedRecord
from cortexguard.simulation.models.trial import Trial
from cortexguard.simulation.utils.load_fused_records import load_fused_records

RecordT = TypeVar("RecordT", bound=BaseFusedRecord)

logger = logging.getLogger(__name__)


class LocalStreamer[RecordT: BaseFusedRecord](BaseStreamer[RecordT]):
    """Replay fused records locally at a configurable rate."""

    def __init__(
        self,
        rate_hz: float = 30.0,
        handle_record: Callable[[RecordT], None] | None = None,
        logger: logging.Logger = logger,
    ):
        self.rate_hz = rate_hz
        self.handle_record = handle_record
        self.logger = logger or logging.getLogger(__name__)

    def load_fused_records(self, path: Path) -> list[RecordT]:
        """Convenience wrapper for loading fused or windowed records."""
        try:
            records: list[RecordT] = load_fused_records(path)
            self.logger.info(f"Loaded {len(records)} fused records from {path}")
            return records
        except Exception as e:
            self.logger.error(f"Failed to load fused records: {e}")
            raise

    def load_records_from_trial(self, trial: Trial) -> list[RecordT]:
        """
        Load fused or windowed records associated with a given trial.

        Args:
            trial: Trial object containing the fused_file path.

        Returns:
            list of fused record instances.
        """
        path = trial.fused_file
        if not path:
            self.logger.error(f"Trial path not set: {trial}")
            raise AttributeError(f"Trial path not set: {trial}")

        if not path.exists():
            self.logger.error(f"Trial file not found: {path}")
            raise FileNotFoundError(f"Trial file not found: {path}")
        return self.load_fused_records(path)

    def load_records_by_id(self, trial_id: str, manifest: list[Trial]) -> list[RecordT]:
        """
        Find the given trial by ID in the manifest and load its fused records.

        Args:
            trial_id: Unique ID of the trial to load.
            manifest: List of Trial objects.

        Raises:
            ValueError if trial_id not found.
        """
        trial = next((t for t in manifest if t.trial_id == trial_id), None)
        if not trial:
            raise ValueError(f"Trial {trial_id!r} not found in manifest.")
        return self.load_records_from_trial(trial)

    def stream(self, records: Iterable[RecordT]) -> None:
        """Stream fused records sequentially with simulated timing."""
        delay = 1.0 / self.rate_hz
        self.logger.info(f"Starting local stream at {self.rate_hz:.1f} Hz ({delay:.3f}s delay)")
        start_time = time.time()
        count = 0

        try:
            for record in records:
                if self.handle_record:
                    self.handle_record(record)
                else:
                    self.logger.debug(f"Record {count}: ts={record.timestamp_ns}")
                time.sleep(delay)
                count += 1

        except KeyboardInterrupt:
            self.logger.info("Stream interrupted by user.")

        elapsed = time.time() - start_time
        self.logger.info(f"Completed streaming {count} records in {elapsed:.2f}s.")
