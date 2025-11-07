import logging
from typing import Any

import requests

from kitchenwatch.core.interfaces.base_receiver import BaseReceiver

logger = logging.getLogger(__name__)


class HttpReceiver(BaseReceiver[Any]):
    """HTTP adapter that sends records to the Edge FastAPI ingestion endpoint."""

    def __init__(
        self,
        edge_url: str,
        verbose: bool = False,
        logger: logging.Logger = logger,
        timeout: float = 2.0,
    ) -> None:
        self.edge_url = edge_url
        self.verbose = verbose
        self.logger = logger
        self.sent_count = 0
        self.timeout = timeout

    def ingest(self, record: Any) -> None:
        """Send a fused record to the Edge API via HTTP POST."""
        try:
            response = requests.post(
                self.edge_url,
                json=record if isinstance(record, dict) else record.dict(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            self.sent_count += 1

            if self.verbose:
                self.logger.info(
                    f"Sent record #{self.sent_count} to {self.edge_url} and got back ({response.status_code}): "
                    f"{response.json()}"
                )
        except requests.RequestException as e:
            self.logger.error(f"Failed to send record to {self.edge_url}: {e}")
