import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from cortexguard.edge.edge_fusion import EdgeFusion
from cortexguard.edge.local_receiver import LocalReceiver
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot


@pytest.mark.asyncio
async def test_local_receiver_calls_edge_fusion() -> None:
    """Test that LocalReceiver correctly calls EdgeFusion.process_record."""
    # Prepare dummy blackboard
    bb = Blackboard()

    # Create EdgeFusion instance with mocked process_record
    fusion = EdgeFusion(bb)

    try:
        # Create a mock that returns a FusionSnapshot
        mock_process = AsyncMock(return_value=MagicMock(spec=FusionSnapshot))

        # Replace the method using object.__setattr__ to avoid method-assign error
        object.__setattr__(fusion, "process_record", mock_process)

        receiver = LocalReceiver(edge_fusion=fusion, verbose=True)

        class DummyRecord:
            timestamp_ns: int = 1234567890
            n_samples: int = 5

            def model_dump(self) -> dict[str, Any]:
                return {"timestamp_ns": self.timestamp_ns, "n_samples": self.n_samples}

        record = DummyRecord()

        await receiver.ingest(record)

        mock_process.assert_awaited_once_with(record)
        assert receiver.received_count == 1
    finally:
        # Cleanup
        fusion.close()


@pytest.mark.asyncio
async def test_local_receiver_handles_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that LocalReceiver handles exceptions from EdgeFusion gracefully."""
    bb = Blackboard()

    class DummyFusion(EdgeFusion):
        async def process_record(self, record: Any) -> FusionSnapshot:
            """Mock process_record that raises an exception."""
            raise RuntimeError("fusion fail")

    fusion = DummyFusion(bb)
    try:
        receiver = LocalReceiver(edge_fusion=fusion, verbose=True)

        class DummyRecord:
            timestamp_ns: int = 1234567890

            def model_dump(self) -> dict[str, Any]:
                return {"timestamp_ns": self.timestamp_ns}

        record = DummyRecord()

        caplog.set_level(logging.DEBUG)
        await receiver.ingest(record)

        assert any(
            "Failed to process record in EdgeFusion" in rec.message for rec in caplog.records
        )
        assert receiver.received_count == 1
    finally:
        # Cleanup
        fusion.close()
