"""
Test suite for EdgeFusion sensor fusion component.

Tests cover:
- EMA initialization and updates
- Window processing behavior
- Statistical computations
- Edge cases and error handling
- State management
"""

import logging
from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from cortexguard.edge.edge_fusion import EdgeFusion
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot
from cortexguard.simulation.models.windowed_fused_record import (
    SensorReading,
    WindowedFusedRecord,
)


@pytest.fixture
def blackboard() -> Blackboard:
    """Create a fresh blackboard for each test."""
    return Blackboard()


@pytest.fixture
def edge_fusion(blackboard: Blackboard) -> Generator[EdgeFusion, None, None]:
    """Create EdgeFusion instance with test blackboard, with cleanup."""
    fusion = EdgeFusion(
        blackboard=blackboard, alpha=0.5
    )  # Higher alpha for faster convergence in tests
    yield fusion
    # Cleanup: shutdown the executor
    fusion.close()


@pytest.fixture
def sample_record() -> WindowedFusedRecord:
    """Create a sample windowed fused record."""
    return WindowedFusedRecord(
        timestamp_ns=1000000000,  # 1 second in nanoseconds
        rgb_path="/path/to/rgb.jpg",
        depth_path="/path/to/depth.jpg",
        window_size_s=0.3,
        n_samples=3,
        sensor_window=[
            SensorReading(
                timestamp_ns=1000000000,
                force_x=1.0,
                force_y=2.0,
                force_z=3.0,
                torque_x=0.1,
                torque_y=0.2,
                torque_z=0.3,
                pos_x=10.0,
                pos_y=20.0,
                pos_z=30.0,
            ),
            SensorReading(
                timestamp_ns=1100000000,
                force_x=1.1,
                force_y=2.1,
                force_z=3.1,
                torque_x=0.11,
                torque_y=0.21,
                torque_z=0.31,
                pos_x=10.1,
                pos_y=20.1,
                pos_z=30.1,
            ),
            SensorReading(
                timestamp_ns=1200000000,
                force_x=1.2,
                force_y=2.2,
                force_z=3.2,
                torque_x=0.12,
                torque_y=0.22,
                torque_z=0.32,
                pos_x=10.2,
                pos_y=20.2,
                pos_z=30.2,
            ),
        ],
    )


class TestEdgeFusionInitialization:
    """Test EdgeFusion initialization and configuration."""

    def test_init_with_custom_alpha(self, blackboard: Blackboard) -> None:
        """Test initialization with custom alpha value."""
        fusion = EdgeFusion(blackboard=blackboard, alpha=0.3)
        assert fusion._alpha == 0.3

    def test_init_with_custom_logger(self, blackboard: Blackboard) -> None:
        """Test initialization with custom logger."""
        custom_logger = logging.getLogger("test_logger")
        fusion = EdgeFusion(blackboard=blackboard, custom_logger=custom_logger)
        assert fusion._logger is custom_logger

    def test_init_with_invalid_alpha_too_low(self, blackboard: Blackboard) -> None:
        """Test that alpha <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="Alpha must be in"):
            EdgeFusion(blackboard=blackboard, alpha=0.0)

        with pytest.raises(ValueError, match="Alpha must be in"):
            EdgeFusion(blackboard=blackboard, alpha=-0.1)

    def test_init_with_invalid_alpha_too_high(self, blackboard: Blackboard) -> None:
        """Test that alpha > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Alpha must be in"):
            EdgeFusion(blackboard=blackboard, alpha=1.1)


class TestEMAComputation:
    """Test EMA state update logic."""

    @pytest.mark.asyncio
    async def test_first_sample_initializes_ema(
        self, edge_fusion: EdgeFusion, blackboard: Blackboard
    ) -> None:
        """Test that first sample initializes EMA state."""
        record = WindowedFusedRecord(
            timestamp_ns=1000000000,
            rgb_path="/test.jpg",
            window_size_s=0.1,
            n_samples=1,
            sensor_window=[
                SensorReading(
                    timestamp_ns=1000000000,
                    force_x=1.0,
                    force_y=2.0,
                    force_z=3.0,
                )
            ],
        )

        await edge_fusion.process_record(record)

        ema_state = await edge_fusion.get_ema_state()
        assert ema_state["force_x"] == 1.0
        assert ema_state["force_y"] == 2.0
        assert ema_state["force_z"] == 3.0

    @pytest.mark.asyncio
    async def test_ema_update_formula(self, blackboard: Blackboard) -> None:
        """Test that EMA formula is applied correctly."""
        # Use alpha=0.5 for easy calculation
        fusion = EdgeFusion(blackboard=blackboard, alpha=0.5)

        # First record: initializes EMA to 1.0
        record1 = WindowedFusedRecord(
            timestamp_ns=1000000000,
            rgb_path="/test1.jpg",
            window_size_s=0.1,
            n_samples=1,
            sensor_window=[SensorReading(timestamp_ns=1000000000, force_x=1.0, force_y=2.0)],
        )
        await fusion.process_record(record1)

        # Second record: should apply EMA
        # new_ema = 0.5 * 3.0 + 0.5 * 1.0 = 2.0
        record2 = WindowedFusedRecord(
            timestamp_ns=2000000000,
            rgb_path="/test2.jpg",
            window_size_s=0.1,
            n_samples=1,
            sensor_window=[SensorReading(timestamp_ns=2000000000, force_x=3.0, force_y=4.0)],
        )
        await fusion.process_record(record2)

        ema_state = await fusion.get_ema_state()
        assert ema_state["force_x"] == 2.0
        assert ema_state["force_y"] == 3.0

    @pytest.mark.asyncio
    async def test_multiple_samples_in_window(
        self, edge_fusion: EdgeFusion, sample_record: WindowedFusedRecord
    ) -> None:
        """Test that all samples in window are processed sequentially."""
        await edge_fusion.process_record(sample_record)

        # With alpha=0.5, processing [1.0, 1.1, 1.2] sequentially:
        # After 1.0: ema = 1.0
        # After 1.1: ema = 0.5*1.1 + 0.5*1.0 = 1.05
        # After 1.2: ema = 0.5*1.2 + 0.5*1.05 = 1.125
        ema_state = await edge_fusion.get_ema_state()
        assert ema_state["force_x"] == pytest.approx(1.125)
        assert ema_state["pos_x"] == pytest.approx(10.125)


class TestNonNumericHandling:
    """Test handling of non-numeric and special fields."""

    @pytest.mark.asyncio
    async def test_timestamp_field_ignored(
        self, edge_fusion: EdgeFusion, blackboard: Blackboard
    ) -> None:
        """Test that timestamp_ns field is not added to EMA state."""
        record = WindowedFusedRecord(
            timestamp_ns=1000000000,
            rgb_path="/test.jpg",
            window_size_s=0.1,
            n_samples=1,
            sensor_window=[
                SensorReading(
                    timestamp_ns=1000000000,
                    force_x=1.0,
                )
            ],
        )

        await edge_fusion.process_record(record)

        ema_state = await edge_fusion.get_ema_state()
        assert "timestamp_ns" not in ema_state

    @pytest.mark.asyncio
    async def test_none_values_ignored(
        self, edge_fusion: EdgeFusion, blackboard: Blackboard
    ) -> None:
        """Test that None values are skipped."""
        record = WindowedFusedRecord(
            timestamp_ns=1000000000,
            rgb_path="/test.jpg",
            window_size_s=0.1,
            n_samples=1,
            sensor_window=[
                SensorReading(
                    timestamp_ns=1000000000,
                    force_x=1.0,
                    force_y=None,  # Should be ignored
                    force_z=3.0,
                )
            ],
        )

        await edge_fusion.process_record(record)

        ema_state = await edge_fusion.get_ema_state()
        assert "force_x" in ema_state
        assert "force_y" not in ema_state
        assert "force_z" in ema_state


class TestWindowStatistics:
    """Test window statistics computation."""

    @pytest.mark.asyncio
    async def test_window_stats_basic(
        self, edge_fusion: EdgeFusion, sample_record: WindowedFusedRecord
    ) -> None:
        """Test basic window statistics calculation."""
        snapshot = await edge_fusion.process_record(sample_record)
        assert snapshot is not None
        stats = snapshot.sensors["window_stats"]

        # force_x: [1.0, 1.1, 1.2]
        assert stats["force_x"]["mean"] == pytest.approx(1.1)
        assert stats["force_x"]["min"] == 1.0
        assert stats["force_x"]["max"] == 1.2
        assert stats["force_x"]["range"] == pytest.approx(0.2)
        assert stats["force_x"]["std"] > 0  # Should have some variance

    @pytest.mark.asyncio
    async def test_window_stats_single_sample(
        self, edge_fusion: EdgeFusion, blackboard: Blackboard
    ) -> None:
        """Test statistics with single sample (std should be 0)."""
        record = WindowedFusedRecord(
            timestamp_ns=1000000000,
            rgb_path="/test.jpg",
            window_size_s=0.1,
            n_samples=1,
            sensor_window=[
                SensorReading(
                    timestamp_ns=1000000000,
                    force_x=1.0,
                    force_y=2.0,
                )
            ],
        )

        snapshot = await edge_fusion.process_record(record)
        assert snapshot is not None
        stats = snapshot.sensors["window_stats"]

        assert stats["force_x"]["mean"] == 1.0
        assert stats["force_x"]["std"] == 0.0
        assert stats["force_x"]["min"] == 1.0
        assert stats["force_x"]["max"] == 1.0
        assert stats["force_x"]["range"] == 0.0

    @pytest.mark.asyncio
    async def test_window_stats_ignores_none(
        self, edge_fusion: EdgeFusion, blackboard: Blackboard
    ) -> None:
        """Test that None values don't appear in stats."""
        record = WindowedFusedRecord(
            timestamp_ns=1000000000,
            rgb_path="/test.jpg",
            window_size_s=0.1,
            n_samples=1,
            sensor_window=[
                SensorReading(
                    timestamp_ns=1000000000,
                    force_x=1.0,
                    force_y=None,  # Should be excluded
                )
            ],
        )

        snapshot = await edge_fusion.process_record(record)
        assert snapshot is not None
        stats = snapshot.sensors["window_stats"]

        assert "force_y" not in stats
        assert "timestamp_ns" not in stats


class TestSnapshotCreation:
    """Test FusionSnapshot creation and blackboard updates."""

    @pytest.mark.asyncio
    async def test_snapshot_structure(
        self, edge_fusion: EdgeFusion, sample_record: WindowedFusedRecord
    ) -> None:
        """Test that snapshot has correct structure."""
        snapshot = await edge_fusion.process_record(sample_record)

        assert isinstance(snapshot, FusionSnapshot)
        assert isinstance(snapshot.timestamp, datetime)
        assert "raw" in snapshot.sensors
        assert "ema_smoothed" in snapshot.sensors
        assert "window_stats" in snapshot.sensors
        assert snapshot.derived is not None

    @pytest.mark.asyncio
    async def test_snapshot_timestamp_conversion(
        self, edge_fusion: EdgeFusion, sample_record: WindowedFusedRecord
    ) -> None:
        """Test that nanosecond timestamp is correctly converted."""
        snapshot = await edge_fusion.process_record(sample_record)
        assert snapshot is not None

        # 1000000000 ns = 1 second = 1970-01-01 00:00:01
        expected_time = datetime.fromtimestamp(1.0)
        assert snapshot.timestamp == expected_time

    @pytest.mark.asyncio
    async def test_blackboard_updated(
        self, edge_fusion: EdgeFusion, blackboard: Blackboard, sample_record: WindowedFusedRecord
    ) -> None:
        """Test that blackboard is updated with snapshot."""
        await edge_fusion.process_record(sample_record)

        bb_snapshot = await blackboard.get_fusion_snapshot()
        assert bb_snapshot is not None
        assert bb_snapshot == edge_fusion._last_snapshot


class TestStateManagement:
    """Test state management and reset functionality."""

    @pytest.mark.asyncio
    async def test_metrics_tracking(
        self, edge_fusion: EdgeFusion, sample_record: WindowedFusedRecord
    ) -> None:
        """Test that metrics are tracked correctly."""
        await edge_fusion.process_record(sample_record)

        metrics = await edge_fusion.get_metrics()
        assert metrics["records_processed"] == 1
        assert metrics["samples_processed"] == 3  # 3 samples in window
        assert metrics["ema_state_size"] == 9  # 9 sensor fields (3 force, 3 torque, 3 pos)

    @pytest.mark.asyncio
    async def test_metrics_accumulate(
        self, edge_fusion: EdgeFusion, sample_record: WindowedFusedRecord
    ) -> None:
        """Test that metrics accumulate across multiple records."""
        await edge_fusion.process_record(sample_record)
        await edge_fusion.process_record(sample_record)

        metrics = await edge_fusion.get_metrics()
        assert metrics["records_processed"] == 2
        assert metrics["samples_processed"] == 6

    @pytest.mark.asyncio
    async def test_reset_ema_state(self, edge_fusion: EdgeFusion) -> None:
        """Test that reset clears all state."""
        # Manually set some state
        edge_fusion._ema_state = {"force_x": 1.0}
        edge_fusion._samples_processed = 10
        edge_fusion._records_processed = 5
        edge_fusion._last_snapshot = MagicMock()

        await edge_fusion.reset_ema_state()

        assert edge_fusion._ema_state == {}
        assert edge_fusion._samples_processed == 0
        assert edge_fusion._records_processed == 0
        assert edge_fusion._last_snapshot is None

    @pytest.mark.asyncio
    async def test_get_ema_state_returns_copy(self, edge_fusion: EdgeFusion) -> None:
        """Test that get_ema_state returns a copy, not reference."""
        edge_fusion._ema_state = {"force_x": 1.0}

        state1 = await edge_fusion.get_ema_state()
        state1["force_x"] = 999.0

        state2 = await edge_fusion.get_ema_state()
        assert state2["force_x"] == 1.0  # Original unchanged


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_empty_sensor_window(
        self, edge_fusion: EdgeFusion, blackboard: Blackboard
    ) -> None:
        """Test processing record with empty sensor window."""
        record = WindowedFusedRecord(
            timestamp_ns=1000000000,
            rgb_path="/test.jpg",
            window_size_s=0.0,
            n_samples=0,
            sensor_window=[],
        )

        snapshot = await edge_fusion.process_record(record)
        assert snapshot is not None

        # Should not crash, should produce empty stats
        assert snapshot.sensors["window_stats"] == {}
        assert await edge_fusion.get_ema_state() == {}

    @pytest.mark.asyncio
    async def test_all_none_values(self, edge_fusion: EdgeFusion, blackboard: Blackboard) -> None:
        """Test processing samples where all values are None."""
        record = WindowedFusedRecord(
            timestamp_ns=1000000000,
            rgb_path="/test.jpg",
            window_size_s=0.1,
            n_samples=1,
            sensor_window=[
                SensorReading(
                    timestamp_ns=1000000000,
                    force_x=None,
                    force_y=None,
                    force_z=None,
                )
            ],
        )

        snapshot = await edge_fusion.process_record(record)
        assert snapshot is not None

        # Should handle gracefully
        assert await edge_fusion.get_ema_state() == {}
        assert snapshot.sensors["window_stats"] == {}

    @pytest.mark.asyncio
    async def test_partial_none_values(
        self, edge_fusion: EdgeFusion, blackboard: Blackboard
    ) -> None:
        """Test processing samples with mix of None and valid values."""
        record = WindowedFusedRecord(
            timestamp_ns=1000000000,
            rgb_path="/test.jpg",
            window_size_s=0.1,
            n_samples=1,
            sensor_window=[
                SensorReading(
                    timestamp_ns=1000000000,
                    force_x=1.0,
                    force_y=None,
                    force_z=3.0,
                )
            ],
        )

        _ = await edge_fusion.process_record(record)

        ema_state = await edge_fusion.get_ema_state()
        assert "force_x" in ema_state
        assert "force_y" not in ema_state
        assert "force_z" in ema_state


class TestIntegration:
    """Integration tests with multiple records."""

    @pytest.mark.asyncio
    async def test_continuous_processing(
        self, edge_fusion: EdgeFusion, blackboard: Blackboard
    ) -> None:
        """Test processing multiple records sequentially."""
        records = [
            WindowedFusedRecord(
                timestamp_ns=i * 1000000000,
                rgb_path=f"/test{i}.jpg",
                window_size_s=0.1,
                n_samples=1,
                sensor_window=[
                    SensorReading(
                        timestamp_ns=i * 1000000000,
                        force_x=1.0 + i * 0.1,
                        force_y=2.0 + i * 0.1,
                    )
                ],
            )
            for i in range(5)
        ]

        snapshots: list[FusionSnapshot] = []
        for record in records:
            snapshot = await edge_fusion.process_record(record)
            assert snapshot is not None
            snapshots.append(snapshot)

        # Check that EMA converges toward recent values
        final_ema = await edge_fusion.get_ema_state()
        assert final_ema["force_x"] > 1.0  # Should increase

        # Check metrics
        metrics = await edge_fusion.get_metrics()
        assert metrics["records_processed"] == 5
        assert metrics["samples_processed"] == 5

    @pytest.mark.asyncio
    async def test_reset_during_processing(
        self, edge_fusion: EdgeFusion, sample_record: WindowedFusedRecord
    ) -> None:
        """Test that reset works correctly during processing."""
        # Process some records
        await edge_fusion.process_record(sample_record)
        await edge_fusion.process_record(sample_record)

        # Reset
        await edge_fusion.reset_ema_state()

        # Process again - should start fresh
        await edge_fusion.process_record(sample_record)

        metrics = await edge_fusion.get_metrics()
        assert metrics["records_processed"] == 1
        assert metrics["samples_processed"] == 3


# Pytest configuration
@pytest.fixture(autouse=True)
def reset_logging() -> Generator[None, None, None]:
    """Reset logging between tests."""
    logging.getLogger("cortexguard").handlers = []
    yield


@pytest.mark.asyncio
async def test_tensor_embedding_serialized_to_list(edge_fusion, blackboard, monkeypatch):
    tensor = torch.tensor([0.1, 0.2, 0.3])

    class DummyEmbedder:
        def embed(self, img):  # type: ignore[override]
            # ensure we received a PIL Image in the happy path
            assert isinstance(img, Image.Image)
            return tensor

    edge_fusion.embedder = DummyEmbedder()

    # Patch PIL.Image.open to return a tiny in-memory image so embedder gets an Image
    monkeypatch.setattr("PIL.Image.open", lambda path: Image.new("RGB", (1, 1)))

    record = WindowedFusedRecord(
        timestamp_ns=1_000_000_000,
        rgb_path="/tmp/nonexistent.jpg",  # path can be non-existent because we patched Image.open
        depth_path=None,
        window_size_s=0.1,
        n_samples=1,
        sensor_window=[SensorReading(timestamp_ns=1_000_000_000, force_x=1.0)],
    )

    snapshot = await edge_fusion.process_record(record)
    assert isinstance(snapshot.sensors["image_embedding"], list)
    assert snapshot.sensors["image_embedding"] == tensor.cpu().numpy().tolist()


@pytest.mark.asyncio
async def test_embedder_exception_does_not_crash(edge_fusion, blackboard):
    class BadEmbedder:
        def embed(self, img):
            raise RuntimeError("embedder boom")

    edge_fusion.embedder = BadEmbedder()
    record = WindowedFusedRecord(
        timestamp_ns=1_000_000_000,
        rgb_path="/tmp/test.jpg",
        depth_path=None,
        window_size_s=0.1,
        n_samples=1,
        sensor_window=[SensorReading(timestamp_ns=1_000_000_000, force_x=1.0)],
    )

    snapshot = await edge_fusion.process_record(record)
    # embedder failure should be handled and image_embedding should be None
    assert snapshot.sensors["image_embedding"] is None


# 3) Blackboard.set_scene_graph invocation
@pytest.mark.asyncio
async def test_set_scene_graph_called(edge_fusion, blackboard, monkeypatch):
    called: dict[str, Any] = {}

    async def fake_set_scene_graph(graph):
        called["graph"] = graph

    # Inject fake method onto the blackboard used by the fixture
    edge_fusion._blackboard.set_scene_graph = fake_set_scene_graph  # type: ignore[attr-defined]

    record = WindowedFusedRecord(
        timestamp_ns=1_000_000_000,
        rgb_path="",  # empty string is valid; field is required
        depth_path=None,
        window_size_s=0.1,
        n_samples=1,
        sensor_window=[SensorReading(timestamp_ns=1_000_000_000, force_x=1.0)],
    )

    await edge_fusion.process_record(record)
    assert "graph" in called
    graph = called["graph"]
    # basic sanity checks on the SceneGraph object
    assert hasattr(graph, "objects")
    assert hasattr(graph, "relationships")


@pytest.mark.asyncio
async def test_state_estimator_update_and_exception(edge_fusion, blackboard):
    called: dict[str, Any] = {}

    class GoodEstimator:
        async def update(self, snapshot):
            return {"state": "ok"}

    class BadEstimator:
        async def update(self, snapshot):
            raise RuntimeError("estimator fail")

    async def fake_update_state_estimate(est):
        called["est"] = est

    # success case
    edge_fusion._state_estimator = GoodEstimator()
    edge_fusion._blackboard.update_state_estimate = fake_update_state_estimate  # type: ignore[attr-defined]
    record = WindowedFusedRecord(
        timestamp_ns=1_000_000_000,
        rgb_path="",
        depth_path=None,
        window_size_s=0.1,
        n_samples=1,
        sensor_window=[SensorReading(timestamp_ns=1_000_000_000, force_x=1.0)],
    )
    await edge_fusion.process_record(record)
    assert called.get("est") == {"state": "ok"}

    # failure case: should not raise
    edge_fusion._state_estimator = BadEstimator()

    # ensure update_state_estimate still exists and is a no-op for this test
    async def noop_update_state_estimate(_):  # type: ignore[unused-def]
        return None

    edge_fusion._blackboard.update_state_estimate = noop_update_state_estimate  # type: ignore[attr-defined]
    await edge_fusion.process_record(record)  # should complete without raising


@pytest.mark.asyncio
async def test_scene_graph_built_and_summary_in_snapshot(edge_fusion, blackboard, monkeypatch):
    # Fake vision inference to return two nearby objects
    def fake_vision(record, embedding):
        return (
            [
                {
                    "id": "obj_a",
                    "label": "hand",
                    "distance_m": 0.2,
                    "confidence": 0.9,
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                },
                {
                    "id": "obj_b",
                    "label": "knife",
                    "distance_m": 0.25,
                    "confidence": 0.95,
                    "bbox": [0.15, 0.12, 0.25, 0.22],
                },
            ],
            None,
        )

    edge_fusion.vision_inference = fake_vision

    called: dict[str, Any] = {}

    async def fake_set_scene_graph(graph):
        called["graph"] = graph

    edge_fusion._blackboard.set_scene_graph = fake_set_scene_graph  # type: ignore[attr-defined]

    record = WindowedFusedRecord(
        timestamp_ns=1_000_000_000,
        rgb_path="",
        depth_path=None,
        window_size_s=0.1,
        n_samples=1,
        sensor_window=[SensorReading(timestamp_ns=1_000_000_000, force_x=1.0)],
    )

    snapshot = await edge_fusion.process_record(record)
    # snapshot should include the lightweight summary
    assert snapshot.sensors["scene_graph_summary"] == [
        {"id": "obj_a", "label": "hand"},
        {"id": "obj_b", "label": "knife"},
    ]

    # the published SceneGraph should contain a 'near' relationship
    assert "graph" in called
    relationships = getattr(called["graph"], "relationships", [])
    assert any(getattr(r, "relationship", "") == "near" for r in relationships)
