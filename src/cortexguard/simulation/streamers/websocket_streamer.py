from collections.abc import Iterable
from typing import TypeVar

from cortexguard.core.interfaces.base_streamer import BaseStreamer
from cortexguard.simulation.models.base_record import BaseFusedRecord

RecordT = TypeVar("RecordT", bound=BaseFusedRecord)


class WebSocketStreamer[RecordT: BaseFusedRecord](BaseStreamer[RecordT]):
    """Placeholder for remote streaming — not yet implemented."""

    def stream(self, records: Iterable[RecordT]) -> None:
        raise NotImplementedError("WebSocket streaming not yet implemented")
