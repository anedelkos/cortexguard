from collections.abc import Iterable
from typing import Protocol, TypeVar

from kitchenwatch.simulation.models.base_record import BaseFusedRecord

RecordT = TypeVar("RecordT", bound=BaseFusedRecord)


class BaseStreamer(Protocol[RecordT]):
    def stream(self, records: Iterable[RecordT]) -> None: ...
