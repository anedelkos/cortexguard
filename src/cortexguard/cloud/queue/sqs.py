"""Async SQS helpers using asyncio.to_thread — no aioboto3 dependency."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SQSMessage:
    receipt_handle: str
    body: dict[str, Any]
    message_id: str


class SQSQueue:
    """Thin async wrapper around the boto3 SQS client.

    The boto3 client is created once on first use and reused across calls.
    boto3 clients are thread-safe, so sharing across ``asyncio.to_thread``
    invocations is safe.
    """

    def __init__(self, queue_url: str, region: str = "us-east-1") -> None:
        self._queue_url = queue_url
        self._region = region
        self._cached_client: Any = None

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def enqueue(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload)
        await asyncio.to_thread(self._send_message, body)

    async def receive(
        self,
        max_messages: int = 1,
        wait_seconds: int = 20,
        visibility_timeout: int = 300,
    ) -> list[SQSMessage]:
        raw = await asyncio.to_thread(
            self._receive_messages, max_messages, wait_seconds, visibility_timeout
        )
        return [
            SQSMessage(
                receipt_handle=m["ReceiptHandle"],
                body=json.loads(m["Body"]),
                message_id=m["MessageId"],
            )
            for m in raw
        ]

    async def delete(self, receipt_handle: str) -> None:
        await asyncio.to_thread(self._delete_message, receipt_handle)

    # ------------------------------------------------------------------
    # Synchronous helpers executed inside threads
    # ------------------------------------------------------------------

    def _client(self) -> Any:  # type: ignore[return]
        if self._cached_client is None:
            import boto3  # type: ignore[import-untyped]

            self._cached_client = boto3.client("sqs", region_name=self._region)
        return self._cached_client

    def _send_message(self, body: str) -> None:
        self._client().send_message(QueueUrl=self._queue_url, MessageBody=body)

    def _receive_messages(
        self, max_messages: int, wait_seconds: int, visibility_timeout: int
    ) -> list[dict[str, Any]]:
        resp: dict[str, Any] = self._client().receive_message(
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_seconds,
            VisibilityTimeout=visibility_timeout,
        )
        messages: list[dict[str, Any]] = resp.get("Messages", [])
        return messages

    def _delete_message(self, receipt_handle: str) -> None:
        self._client().delete_message(QueueUrl=self._queue_url, ReceiptHandle=receipt_handle)
