"""EdgeFusion unit tests that do not require torch/ML dependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cortexguard.edge.edge_fusion import EdgeFusion
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.simulation.models.windowed_fused_record import SensorReading, WindowedFusedRecord


def _make_fusion() -> EdgeFusion:
    bb = MagicMock(spec=Blackboard)
    bb.get_scene_graph = AsyncMock(return_value=None)
    bb.update_state_estimate = AsyncMock()
    return EdgeFusion(blackboard=bb)


def _make_record(timestamp_ns: int, smoke_ppm: float | None = None) -> WindowedFusedRecord:
    return WindowedFusedRecord(
        timestamp_ns=timestamp_ns,
        rgb_path="",
        window_size_s=0.1,
        n_samples=1,
        sensor_window=[SensorReading(timestamp_ns=timestamp_ns, smoke_ppm=smoke_ppm)],
    )


# ---------------------------------------------------------------------------
# Smoke hysteresis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_not_set_on_single_detection() -> None:
    """Single positive reading must not set smoke state — _SMOKE_SET_CONSECUTIVE=2 requires two consecutive."""
    fusion = _make_fusion()
    record = _make_record(1_000_000_000, smoke_ppm=100.0)  # above 50ppm threshold

    result = await fusion._detect_smoke(record, None)

    assert result is False, (
        "Smoke state should not be set after a single positive reading. "
        "_SMOKE_SET_CONSECUTIVE=2 requires two consecutive windows above threshold."
    )


@pytest.mark.asyncio
async def test_smoke_set_after_consecutive_detections() -> None:
    """Two consecutive smoke readings above threshold must set smoke state."""
    fusion = _make_fusion()
    record = _make_record(1_000_000_000, smoke_ppm=100.0)

    await fusion._detect_smoke(record, None)  # score: 1
    result = await fusion._detect_smoke(record, None)  # score: 2 >= _SMOKE_SET_CONSECUTIVE

    assert result is True, (
        "Smoke state should be set after two consecutive readings above threshold. "
        "Check _SMOKE_SET_CONSECUTIVE and score increment logic."
    )


@pytest.mark.asyncio
async def test_smoke_clears_only_when_score_reaches_zero() -> None:
    """Once set, smoke state only clears when the leaky-bucket score reaches zero, not on the first clean reading."""
    fusion = _make_fusion()
    smoky = _make_record(1_000_000_000, smoke_ppm=100.0)
    clear = _make_record(2_000_000_000, smoke_ppm=0.0)

    # Set smoke state: score reaches _SMOKE_SET_CONSECUTIVE
    await fusion._detect_smoke(smoky, None)
    await fusion._detect_smoke(smoky, None)
    assert fusion._smoke_state is True

    # One clean reading decrements score but should not yet clear
    result_after_one_clear = await fusion._detect_smoke(clear, None)

    assert result_after_one_clear is True, (
        "Smoke should still be active after a single clean reading. "
        "The leaky bucket must reach 0 before clearing — not on the first clean window."
    )


# ---------------------------------------------------------------------------
# Out-of-order record rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_out_of_order_record_returns_none() -> None:
    """_update_ema must return None for a record whose timestamp_ns is earlier than the last processed."""
    fusion = _make_fusion()

    later = _make_record(timestamp_ns=2_000_000_000)
    earlier = _make_record(timestamp_ns=1_000_000_000)

    await fusion._update_ema(later)  # establishes last_processed = 2s
    result = await fusion._update_ema(earlier)  # out-of-order: 1s < 2s

    assert result is None, (
        f"Expected None for out-of-order record but got {result!r}. "
        "_update_ema must reject records older than the last processed timestamp."
    )


@pytest.mark.asyncio
async def test_in_order_record_is_processed() -> None:
    """In-order records must not be rejected."""
    fusion = _make_fusion()

    first = _make_record(timestamp_ns=1_000_000_000)
    second = _make_record(timestamp_ns=2_000_000_000)

    await fusion._update_ema(first)
    result = await fusion._update_ema(second)

    assert result is not None, "In-order record was unexpectedly rejected by _update_ema."


# ---------------------------------------------------------------------------
# Dead fields: _smoke_consec_set / _smoke_consec_clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_consec_set_tracks_consecutive_positive_readings() -> None:
    """_smoke_consec_set must increment on each positive reading; currently initialised to 0 and never updated."""
    fusion = _make_fusion()
    smoky = _make_record(1_000_000_000, smoke_ppm=100.0)

    await fusion._detect_smoke(smoky, None)  # first positive
    await fusion._detect_smoke(smoky, None)  # second positive

    assert fusion._smoke_consec_set == 2


# ---------------------------------------------------------------------------
# _SMOKE_CLEAR_CONSECUTIVE is never referenced in _detect_smoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_clears_within_smoke_clear_consecutive_windows_from_max() -> None:
    """After _SMOKE_CLEAR_CONSECUTIVE=2 clean windows from max score, smoke state should clear."""
    fusion = _make_fusion()
    smoky = _make_record(1_000_000_000, smoke_ppm=100.0)
    clear = _make_record(2_000_000_000, smoke_ppm=0.0)

    # Drive score to _SMOKE_MAX_SCORE (5) via 5 consecutive positive windows
    for _ in range(5):
        await fusion._detect_smoke(smoky, None)

    assert fusion._smoke_state is True
    assert fusion._smoke_score == 5

    # Per _SMOKE_CLEAR_CONSECUTIVE=2, two clean windows should be enough to clear
    await fusion._detect_smoke(clear, None)  # score: 5 → 4
    result = await fusion._detect_smoke(clear, None)  # score: 4 → 3

    assert result is False
