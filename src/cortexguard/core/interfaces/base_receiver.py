from collections.abc import Awaitable
from typing import Protocol, TypeVar, runtime_checkable

# Generic type variable to allow any record-like model
RecordT = TypeVar("RecordT")


@runtime_checkable
class BaseReceiver(Protocol[RecordT]):
    """Protocol for any component that can receive fused data records."""

    def ingest(self, record: RecordT) -> None | Awaitable[None]:
        """
        Handle a single fused record of any supported type.

        Can be synchronous (returns None) or asynchronous (returns Awaitable[None]).
        """
        ...
