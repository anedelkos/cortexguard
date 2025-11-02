import logging
from typing import Any

from kitchenwatch.core.interfaces.base_receiver import BaseReceiver

logger = logging.getLogger(__name__)


class LocalEdgeReceiver(BaseReceiver[Any]):
    """Minimal receiver that logs incoming fused records."""

    def __init__(self, verbose: bool = False, logger: logging.Logger = logger) -> None:
        self.verbose = verbose
        self.logger = logger
        self.received_count = 0

    def ingest(self, record: Any) -> None:
        """Handle a single fused record (any type)."""
        self.received_count += 1

        if self.verbose:
            self.logger.info(f"[LocalEdgeReceiver] Received record: {record}")
        else:
            ts = getattr(record, "timestamp_ns", None)
            if ts is None and isinstance(record, dict):
                ts = record.get("timestamp_ns", "N/A")

            self.logger.info(f"Received record @ {ts}")

            n_samples = getattr(record, "n_samples", None)
            if n_samples is None and isinstance(record, dict):
                n_samples = record.get("n_samples")

            if n_samples:
                self.logger.info(f"with {n_samples} samples")
