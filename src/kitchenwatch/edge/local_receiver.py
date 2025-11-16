import logging
from typing import Any

from kitchenwatch.core.interfaces.base_receiver import BaseReceiver
from kitchenwatch.edge.edge_fusion import EdgeFusion

logger = logging.getLogger(__name__)


class LocalReceiver(BaseReceiver[Any]):
    """Async receiver that forwards incoming fused records to EdgeFusion."""

    def __init__(
        self,
        edge_fusion: EdgeFusion,
        verbose: bool = False,
        custom_logger: logging.Logger | None = None,
    ) -> None:
        """
        Args:
            edge_fusion: Reference to EdgeFusion instance
            verbose: Whether to log full record details
            custom_logger: Optional logger
        """
        self._edge_fusion: EdgeFusion = edge_fusion
        self._verbose: bool = verbose
        self._logger: logging.Logger = custom_logger or logger
        self._received_count: int = 0

    @property
    def received_count(self) -> int:
        """Total records received by this receiver."""
        return self._received_count

    async def ingest(self, record: Any) -> None:
        """Handle a single fused record asynchronously."""
        self._received_count += 1

        # Logging
        if self._verbose:
            self._logger.info(f"[LocalReceiver] Received record: {record}")
        else:
            ts = getattr(record, "timestamp_ns", None)
            if ts is None and isinstance(record, dict):
                ts = record.get("timestamp_ns", "N/A")
            n_samples = getattr(record, "n_samples", None)
            if n_samples is None and isinstance(record, dict):
                n_samples = record.get("n_samples")
            log_line = f"Received record @ {ts}"
            if n_samples:
                log_line += f" with {n_samples} samples"
            self._logger.info(log_line)

        # Forward record to EdgeFusion
        try:
            await self._edge_fusion.process_record(record)
        except Exception as e:
            self._logger.exception(f"Failed to process record in EdgeFusion: {e}")
