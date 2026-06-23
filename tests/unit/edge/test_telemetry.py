from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from cortexguard.edge.models.fusion_snapshot import FusionSnapshot
from cortexguard.edge.telemetry import TelemetryClient


class RecordingHTTPClient:
    """Test double that records POST calls instead of making real HTTP requests."""

    def __init__(self) -> None:
        self.post_calls: list[dict[str, Any]] = []
        self.post_should_fail: bool = False
        self.aclose_called: bool = False

    async def post(self, url: str, **kwargs: Any) -> MagicMock:
        self.post_calls.append({"url": url, **kwargs})
        if self.post_should_fail:
            raise httpx.HTTPError("mock failure")
        return MagicMock(status_code=202)

    async def aclose(self) -> None:
        self.aclose_called = True


@pytest.fixture
def snapshot() -> FusionSnapshot:
    return FusionSnapshot(
        id="snap_1",
        timestamp=datetime.now(UTC),
        sensors={"force_N": 10.0, "temp_C": 45.0},
        derived={"ema_force": 9.8},
    )


@pytest.fixture
def http_client() -> RecordingHTTPClient:
    return RecordingHTTPClient()


@pytest.fixture
def telemetry(http_client: RecordingHTTPClient) -> TelemetryClient:
    return TelemetryClient(
        cloud_api_url="http://localhost:8001",
        cloud_api_key="test-key",
        batch_size=3,
        flush_interval_s=60.0,
        http_client=http_client,
    )


@pytest.mark.asyncio
async def test_send_appends_to_buffer(telemetry: TelemetryClient, snapshot: FusionSnapshot) -> None:
    await telemetry.send("dev_1", "step_1", "completed", snapshot)
    assert len(telemetry._buffer) == 1


@pytest.mark.asyncio
async def test_flush_on_batch_size(
    telemetry: TelemetryClient, http_client: RecordingHTTPClient, snapshot: FusionSnapshot
) -> None:
    for _ in range(3):
        await telemetry.send("dev_1", "step_1", "completed", snapshot)

    assert len(http_client.post_calls) == 1
    assert len(telemetry._buffer) == 0


@pytest.mark.asyncio
async def test_flush_on_stop(
    telemetry: TelemetryClient, http_client: RecordingHTTPClient, snapshot: FusionSnapshot
) -> None:
    await telemetry.send("dev_1", "step_1", "completed", snapshot)
    await telemetry.stop()

    assert len(http_client.post_calls) == 1


@pytest.mark.asyncio
async def test_network_error_swallowed(
    telemetry: TelemetryClient, http_client: RecordingHTTPClient, snapshot: FusionSnapshot
) -> None:
    http_client.post_should_fail = True

    for _ in range(3):
        await telemetry.send("dev_1", "step_1", "completed", snapshot)

    assert len(http_client.post_calls) == 1
    assert len(telemetry._buffer) == 0


@pytest.mark.asyncio
async def test_network_error_logged(
    telemetry: TelemetryClient,
    http_client: RecordingHTTPClient,
    snapshot: FusionSnapshot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    http_client.post_should_fail = True
    caplog.set_level(logging.WARNING)

    for _ in range(3):
        await telemetry.send("dev_1", "step_1", "completed", snapshot)

    assert any("Telemetry flush failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_start_creates_background_task(telemetry: TelemetryClient) -> None:
    await telemetry.start()
    assert telemetry._task is not None
    assert not telemetry._task.done()
    await telemetry.stop()


@pytest.mark.asyncio
async def test_auth_header_sent(snapshot: FusionSnapshot) -> None:
    client = RecordingHTTPClient()
    t = TelemetryClient(
        cloud_api_url="http://localhost:8001",
        cloud_api_key="sk-test",
        http_client=client,
    )
    await t.send("dev_1", "step_1", "completed", snapshot)
    await t.stop()

    headers: Any = client.post_calls[0].get("headers", {})
    assert headers.get("X-CortexGuard-Key") == "sk-test"


@pytest.mark.asyncio
async def test_payload_structure(
    telemetry: TelemetryClient, http_client: RecordingHTTPClient, snapshot: FusionSnapshot
) -> None:
    await telemetry.send("dev_1", "step_1", "completed", snapshot)
    await telemetry.stop()

    records = http_client.post_calls[0]["json"]["records"]
    assert len(records) == 1
    payload = records[0]
    assert payload["device_id"] == "dev_1"
    assert payload["key"] == "step_1"
    assert payload["outcome"] == "completed"
    assert isinstance(payload["timestamp"], str)
    assert payload["timestamp"].endswith("+00:00")
    assert payload["sensor_snapshot"]["id"] == "snap_1"
    assert payload["sensor_snapshot"]["sensors"]["force_N"] == 10.0


@pytest.mark.asyncio
async def test_empty_buffer_no_post(
    http_client: RecordingHTTPClient,
) -> None:
    t = TelemetryClient(
        cloud_api_url="http://localhost:8001",
        http_client=http_client,
    )
    await t.stop()
    assert len(http_client.post_calls) == 0


@pytest.mark.asyncio
async def test_no_auth_header_when_key_not_set(snapshot: FusionSnapshot) -> None:
    client = RecordingHTTPClient()
    t = TelemetryClient(
        cloud_api_url="http://localhost:8001",
        http_client=client,
    )
    await t.send("dev_1", "step_1", "completed", snapshot)
    await t.stop()

    headers: Any = client.post_calls[0].get("headers", {})
    assert "X-CortexGuard-Key" not in headers


@pytest.mark.asyncio
async def test_background_task_cancelled_on_stop(telemetry: TelemetryClient) -> None:
    await telemetry.start()
    await telemetry.stop()
    assert telemetry._task is None


@pytest.mark.asyncio
async def test_does_not_close_injected_client(
    telemetry: TelemetryClient, http_client: RecordingHTTPClient
) -> None:
    await telemetry.stop()
    assert not http_client.aclose_called


@pytest.mark.asyncio
async def test_flush_passes_timeout(snapshot: FusionSnapshot) -> None:
    client = RecordingHTTPClient()
    t = TelemetryClient(
        cloud_api_url="http://localhost:8001",
        timeout_s=5.0,
        http_client=client,
    )
    await t.send("dev_1", "step_1", "completed", snapshot)
    await t.stop()

    timeout = client.post_calls[0].get("timeout")
    assert timeout == 5.0
