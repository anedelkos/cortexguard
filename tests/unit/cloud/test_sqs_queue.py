"""Unit tests for SQSQueue using mocked boto3."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cortexguard.cloud.queue.sqs import SQSMessage, SQSQueue


def _make_sqs_response(messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"Messages": messages}


@pytest.fixture
def queue() -> SQSQueue:
    return SQSQueue(queue_url="https://sqs.us-east-1.amazonaws.com/123/test-queue")


@pytest.fixture
def mock_boto_client() -> MagicMock:
    return MagicMock()


@pytest.mark.asyncio
async def test_enqueue_sends_message(queue: SQSQueue, mock_boto_client: MagicMock) -> None:
    with patch("boto3.client", return_value=mock_boto_client):
        await queue.enqueue({"trace_id": "abc", "packet": {}})

    mock_boto_client.send_message.assert_called_once()
    call_kwargs = mock_boto_client.send_message.call_args[1]
    assert call_kwargs["QueueUrl"] == queue._queue_url
    body = json.loads(call_kwargs["MessageBody"])
    assert body["trace_id"] == "abc"


@pytest.mark.asyncio
async def test_receive_returns_parsed_messages(
    queue: SQSQueue, mock_boto_client: MagicMock
) -> None:
    raw_body = json.dumps({"trace_id": "xyz", "packet": {}})
    mock_boto_client.receive_message.return_value = _make_sqs_response(
        [{"ReceiptHandle": "rh-1", "Body": raw_body, "MessageId": "msg-1"}]
    )

    with patch("boto3.client", return_value=mock_boto_client):
        messages = await queue.receive(max_messages=1, wait_seconds=0)

    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, SQSMessage)
    assert msg.receipt_handle == "rh-1"
    assert msg.message_id == "msg-1"
    assert msg.body["trace_id"] == "xyz"


@pytest.mark.asyncio
async def test_receive_empty_queue(queue: SQSQueue, mock_boto_client: MagicMock) -> None:
    mock_boto_client.receive_message.return_value = {}

    with patch("boto3.client", return_value=mock_boto_client):
        messages = await queue.receive(max_messages=1, wait_seconds=0)

    assert messages == []


@pytest.mark.asyncio
async def test_delete_calls_sqs(queue: SQSQueue, mock_boto_client: MagicMock) -> None:
    with patch("boto3.client", return_value=mock_boto_client):
        await queue.delete("receipt-handle-abc")

    mock_boto_client.delete_message.assert_called_once_with(
        QueueUrl=queue._queue_url,
        ReceiptHandle="receipt-handle-abc",
    )
