from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar

from cortexguard.simulation.models.base_record import BaseFusedRecord

RecordT = TypeVar("RecordT", bound=BaseFusedRecord)


class BaseStreamer(Protocol[RecordT]):
    def stream(self, records: Iterable[RecordT]) -> None: ...
