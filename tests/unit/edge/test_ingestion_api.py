"""Tests for the ingestion API route — regression for union discriminator bug."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortexguard.edge.api.ingestion import get_ingestion_router
from cortexguard.edge.local_receiver import LocalReceiver


def _make_client() -> tuple[TestClient, AsyncMock]:
    ingest_mock = AsyncMock()
    receiver = AsyncMock(spec=LocalReceiver)
    receiver.ingest = ingest_mock
    receiver.received_count = 0

    app = FastAPI()
    app.include_router(get_ingestion_router(receiver))
    return TestClient(app, raise_server_exceptions=False), ingest_mock


# ---------------------------------------------------------------------------
# m3 — WindowedFusedRecord payload must not be silently coerced to FusedRecord
# ---------------------------------------------------------------------------


def test_windowed_fused_record_is_accepted_as_202() -> None:
    client, _ = _make_client()

    payload = {
        "timestamp_ns": 1_000_000,
        "rgb_path": "/data/frame_001.jpg",
        "window_size_s": 1.0,
        "n_samples": 10,
        "sensor_window": [],
    }

    response = client.post("/ingest", json=payload)

    assert response.status_code == 202, (
        f"Expected 202, got {response.status_code}. "
        "WindowedFusedRecord payload is likely being silently coerced to FusedRecord "
        "due to missing union discriminator."
    )


def test_windowed_fused_record_response_identifies_correct_type() -> None:
    client, _ = _make_client()

    payload = {
        "timestamp_ns": 1_000_000,
        "rgb_path": "/data/frame_001.jpg",
        "window_size_s": 1.0,
        "n_samples": 10,
        "sensor_window": [],
    }

    response = client.post("/ingest", json=payload)

    assert response.status_code == 202
    assert response.json()["record_type"] == "WindowedFusedRecord", (
        f"Got record_type={response.json().get('record_type')!r}. "
        "Parser resolved to wrong type — union needs a discriminator."
    )


def test_fused_record_payload_is_still_accepted() -> None:
    """Fixing the discriminator must not break plain FusedRecord ingestion."""
    client, _ = _make_client()

    payload = {
        "timestamp_ns": 2_000_000,
        "rgb_path": "/data/frame_002.jpg",
    }

    response = client.post("/ingest", json=payload)

    assert response.status_code == 202
    assert response.json()["record_type"] == "FusedRecord"
