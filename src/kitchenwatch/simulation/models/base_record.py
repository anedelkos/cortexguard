from pydantic import BaseModel


class BaseFusedRecord(BaseModel):
    """Common base for all fused record types."""

    timestamp_ns: int
    rgb_path: str
    depth_path: str | None = None
