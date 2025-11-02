from typing import Protocol, TypeVar, runtime_checkable

# Generic type variable to allow any record-like model
RecordT = TypeVar("RecordT")


@runtime_checkable
class BaseReceiver(Protocol[RecordT]):
    """Protocol for any component that can receive fused data records."""

    def ingest(self, record: RecordT) -> None:
        """Handle a single fused record of any supported type."""
        ...
