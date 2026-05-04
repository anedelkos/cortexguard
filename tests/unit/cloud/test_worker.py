"""Unit tests for the SQS worker _process function and main() startup guards."""

from __future__ import annotations

import uuid
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortexguard.cloud.worker import _process, main
from cortexguard.edge.models.mayday_packet import MaydayPacket, SystemHealth


def _make_raw_packet(trace_id: str | None = None) -> dict[str, object]:
    packet = MaydayPacket(
        trace_id=trace_id or str(uuid.uuid4()),
        device_id="dev-01",
        timestamp=datetime.now(UTC),
        health=SystemHealth(),
    )
    return packet.model_dump(mode="json")  # type: ignore[return-value]


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    orch = MagicMock()
    orch.run_once = AsyncMock()
    return orch


@pytest.fixture
def mock_sqs() -> MagicMock:
    sqs = MagicMock()
    sqs.delete = AsyncMock()
    return sqs


@pytest.mark.asyncio
async def test_process_valid_message(mock_orchestrator: MagicMock, mock_sqs: MagicMock) -> None:
    trace_id = str(uuid.uuid4())
    body: dict[str, object] = {"trace_id": trace_id, "packet": _make_raw_packet(trace_id)}

    await _process(body, mock_orchestrator, mock_sqs, "rh-1")

    mock_orchestrator.run_once.assert_awaited_once()
    called_packet: MaydayPacket = mock_orchestrator.run_once.call_args[0][0]
    assert called_packet.trace_id == trace_id
    mock_sqs.delete.assert_awaited_once_with("rh-1")


@pytest.mark.asyncio
async def test_process_missing_packet_deletes_immediately(
    mock_orchestrator: MagicMock, mock_sqs: MagicMock
) -> None:
    body: dict[str, object] = {"trace_id": "bad-msg"}  # no "packet" key

    await _process(body, mock_orchestrator, mock_sqs, "rh-bad")

    mock_orchestrator.run_once.assert_not_awaited()
    mock_sqs.delete.assert_awaited_once_with("rh-bad")


@pytest.mark.asyncio
async def test_process_packet_not_dict_deletes_immediately(
    mock_orchestrator: MagicMock, mock_sqs: MagicMock
) -> None:
    body: dict[str, object] = {"trace_id": "bad-msg", "packet": "not-a-dict"}

    await _process(body, mock_orchestrator, mock_sqs, "rh-bad2")

    mock_orchestrator.run_once.assert_not_awaited()
    mock_sqs.delete.assert_awaited_once_with("rh-bad2")


@pytest.mark.asyncio
async def test_process_orchestrator_exception_propagates(
    mock_orchestrator: MagicMock, mock_sqs: MagicMock
) -> None:
    mock_orchestrator.run_once.side_effect = RuntimeError("planning failed")
    body: dict[str, object] = {"trace_id": "t1", "packet": _make_raw_packet()}

    with pytest.raises(RuntimeError, match="planning failed"):
        await _process(body, mock_orchestrator, mock_sqs, "rh-3")

    mock_sqs.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_invalid_packet_fields_deletes_immediately(
    mock_orchestrator: MagicMock, mock_sqs: MagicMock
) -> None:
    # dict but missing required MaydayPacket fields → ValidationError
    body: dict[str, object] = {"trace_id": "t2", "packet": {"not": "a valid packet"}}

    await _process(body, mock_orchestrator, mock_sqs, "rh-4")

    mock_orchestrator.run_once.assert_not_awaited()
    mock_sqs.delete.assert_awaited_once_with("rh-4")


@pytest.mark.asyncio
async def test_main_raises_without_sqs_url() -> None:
    with patch("cortexguard.cloud.worker.CloudConfig") as mock_cfg_cls:
        cfg = MagicMock()
        cfg.sqs_queue_url = None
        mock_cfg_cls.return_value = cfg

        with pytest.raises(RuntimeError, match="CLOUD_SQS_QUEUE_URL"):
            await main()


@pytest.mark.asyncio
async def test_main_raises_without_postgres() -> None:
    with patch("cortexguard.cloud.worker.CloudConfig") as mock_cfg_cls:
        cfg = MagicMock()
        cfg.sqs_queue_url = "https://sqs.us-east-1.amazonaws.com/123/q"
        cfg.incident_store = "sqlite"
        cfg.db_url = None
        mock_cfg_cls.return_value = cfg

        with pytest.raises(RuntimeError, match="CLOUD_INCIDENT_STORE=postgres"):
            await main()


def _infra_patches() -> list[Any]:
    """Return the list of patch targets needed to run main() without real I/O."""
    return [
        patch("cortexguard.cloud.worker.PostgresIncidentRepository"),
        patch("cortexguard.cloud.worker.get_embedder"),
        patch("cortexguard.cloud.worker.InMemoryVectorStore"),
        patch("cortexguard.cloud.worker.RetrievalStore"),
        patch("cortexguard.cloud.worker.get_llm_client", return_value=None),
        patch("cortexguard.cloud.worker.CapabilityAdapter"),
        patch("cortexguard.cloud.worker.PlanValidator"),
        patch("cortexguard.cloud.worker.CloudOrchestrator"),
        patch("cortexguard.cloud.worker.SQSQueue"),
        patch("cortexguard.cloud.worker.CloudConfig"),
    ]


def _make_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.sqs_queue_url = "https://sqs.us-east-1.amazonaws.com/123/q"
    cfg.incident_store = "postgres"
    cfg.db_url = "postgresql://user:pass@localhost/db"
    cfg.sqs_region = "us-east-1"
    cfg.vector_store_backend = "memory"
    return cfg


@pytest.mark.asyncio
async def test_main_polls_empty_queue_and_stops() -> None:
    """main() initialises all components, polls once (empty), then stops cleanly."""
    mock_stop = MagicMock()
    mock_stop.is_set.side_effect = [False, True]

    mock_repo = AsyncMock()
    mock_sqs_instance = MagicMock()
    mock_sqs_instance.receive = AsyncMock(return_value=[])

    patches = _infra_patches()
    with ExitStack() as stack:
        mocks = {p.attribute: stack.enter_context(p) for p in patches}
        stack.enter_context(patch("asyncio.Event", return_value=mock_stop))

        mocks["CloudConfig"].return_value = _make_cfg()
        mocks["PostgresIncidentRepository"].return_value = mock_repo
        mocks["SQSQueue"].return_value = mock_sqs_instance

        await main()

    mock_repo.initialize.assert_awaited_once()
    mock_sqs_instance.receive.assert_awaited_once()
    mock_repo.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_processes_one_message() -> None:
    """main() processes a message from the queue then stops cleanly."""
    from cortexguard.cloud.queue.sqs import SQSMessage

    mock_stop = MagicMock()
    mock_stop.is_set.side_effect = [False, False, True]  # while / if-inside-loop / while again

    mock_repo = AsyncMock()
    msg = SQSMessage(receipt_handle="rh-1", body={"trace_id": "t1", "packet": {}}, message_id="m1")
    mock_sqs_instance = MagicMock()
    mock_sqs_instance.receive = AsyncMock(return_value=[msg])

    patches = _infra_patches()
    with ExitStack() as stack:
        mocks = {p.attribute: stack.enter_context(p) for p in patches}
        stack.enter_context(patch("asyncio.Event", return_value=mock_stop))
        mock_process = stack.enter_context(
            patch("cortexguard.cloud.worker._process", new_callable=AsyncMock)
        )

        mocks["CloudConfig"].return_value = _make_cfg()
        mocks["PostgresIncidentRepository"].return_value = mock_repo
        mocks["SQSQueue"].return_value = mock_sqs_instance

        await main()

    mock_process.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_process_exception_is_logged_not_fatal() -> None:
    """main() logs _process errors but keeps running (message will reappear after visibility timeout)."""
    from cortexguard.cloud.queue.sqs import SQSMessage

    mock_stop = MagicMock()
    mock_stop.is_set.side_effect = [False, False, True]

    mock_repo = AsyncMock()
    msg = SQSMessage(receipt_handle="rh-err", body={}, message_id="m-err")
    mock_sqs_instance = MagicMock()
    mock_sqs_instance.receive = AsyncMock(return_value=[msg])

    patches = _infra_patches()
    with ExitStack() as stack:
        mocks = {p.attribute: stack.enter_context(p) for p in patches}
        stack.enter_context(patch("asyncio.Event", return_value=mock_stop))
        stack.enter_context(
            patch(
                "cortexguard.cloud.worker._process",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            )
        )

        mocks["CloudConfig"].return_value = _make_cfg()
        mocks["PostgresIncidentRepository"].return_value = mock_repo
        mocks["SQSQueue"].return_value = mock_sqs_instance

        # Exception inside _process must NOT propagate out of main()
        await main()

    mock_repo.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_wraps_llm_client_in_throttler() -> None:
    """main() wraps a non-None LLM client in LLMThrottler before passing it to the orchestrator."""
    mock_stop = MagicMock()
    mock_stop.is_set.side_effect = [False, True]

    mock_repo = AsyncMock()
    mock_sqs_instance = MagicMock()
    mock_sqs_instance.receive = AsyncMock(return_value=[])
    mock_llm = MagicMock()
    mock_throttler_instance = MagicMock()

    patches = _infra_patches()
    with ExitStack() as stack:
        mocks = {p.attribute: stack.enter_context(p) for p in patches}
        stack.enter_context(patch("asyncio.Event", return_value=mock_stop))
        mock_throttler = stack.enter_context(patch("cortexguard.cloud.worker.LLMThrottler"))
        mock_throttler.return_value = mock_throttler_instance

        mocks["CloudConfig"].return_value = _make_cfg()
        mocks["PostgresIncidentRepository"].return_value = mock_repo
        mocks["SQSQueue"].return_value = mock_sqs_instance
        mocks["get_llm_client"].return_value = mock_llm

        await main()

    mock_throttler.assert_called_once_with(mock_llm, mocks["CloudConfig"].return_value)


@pytest.mark.asyncio
async def test_main_retries_after_receive_error() -> None:
    """main() catches SQS receive errors, sleeps 5s, and retries."""
    mock_stop = MagicMock()
    mock_stop.is_set.side_effect = [False, False, True]

    mock_repo = AsyncMock()
    mock_sqs_instance = MagicMock()
    mock_sqs_instance.receive = AsyncMock(side_effect=[RuntimeError("SQS down"), []])

    patches = _infra_patches()
    with ExitStack() as stack:
        mocks = {p.attribute: stack.enter_context(p) for p in patches}
        stack.enter_context(patch("asyncio.Event", return_value=mock_stop))
        mock_sleep = stack.enter_context(patch("asyncio.sleep", new_callable=AsyncMock))

        mocks["CloudConfig"].return_value = _make_cfg()
        mocks["PostgresIncidentRepository"].return_value = mock_repo
        mocks["SQSQueue"].return_value = mock_sqs_instance

        await main()

    mock_sleep.assert_awaited_once_with(5)
    assert mock_sqs_instance.receive.await_count == 2
