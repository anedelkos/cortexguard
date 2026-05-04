"""SQS planning worker — long-polls SQS and runs the cloud planning workflow.

Start with:
    python -m cortexguard.cloud.worker

Reads the same CLOUD_* environment variables as the cloud API, plus
CLOUD_SQS_QUEUE_URL (required) and CLOUD_SQS_REGION (default: us-east-1).
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import signal

from pydantic import ValidationError as PydanticValidationError

from cortexguard.cloud.config import CloudConfig
from cortexguard.cloud.orchestrator import CloudOrchestrator
from cortexguard.cloud.persistence.postgres_repository import PostgresIncidentRepository
from cortexguard.cloud.planner.factory import get_llm_client
from cortexguard.cloud.planner.throttler import LLMThrottler
from cortexguard.cloud.queue.sqs import SQSQueue
from cortexguard.cloud.retrieval.embedder import get_embedder
from cortexguard.cloud.retrieval.store import RetrievalStore
from cortexguard.cloud.retrieval.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStoreProtocol,
)
from cortexguard.cloud.validation.capability_adapter import CapabilityAdapter
from cortexguard.cloud.validation.plan_validator import PlanValidator
from cortexguard.common.logging_config import setup_logging
from cortexguard.edge.models.mayday_packet import MaydayPacket

logger = logging.getLogger(__name__)

_POLL_WAIT_SECONDS = 20
_VISIBILITY_TIMEOUT = 300  # must be >= worst-case planning duration
_HEARTBEAT_FILE = pathlib.Path(
    "/tmp/worker-heartbeat"
)  # nosec B108 — fixed path for ECS container health check


async def _process(
    msg_body: dict[str, object],
    orchestrator: CloudOrchestrator,
    sqs: SQSQueue,
    receipt_handle: str,
) -> None:
    trace_id = str(msg_body.get("trace_id", ""))
    raw_packet = msg_body.get("packet")
    if not isinstance(raw_packet, dict):
        logger.error("worker: invalid SQS message — missing packet dict trace_id=%s", trace_id)
        # Unprocessable message — delete immediately rather than cycling through
        # the full visibility timeout before reaching the DLQ.
        await sqs.delete(receipt_handle)
        return

    try:
        packet = MaydayPacket.model_validate(raw_packet)
    except PydanticValidationError:
        logger.error("worker: packet failed validation trace_id=%s — deleting", trace_id)
        await sqs.delete(receipt_handle)
        return

    await orchestrator.run_once(packet)
    await sqs.delete(receipt_handle)
    logger.info("worker: processed and deleted trace_id=%s", trace_id)


async def main() -> None:
    config = CloudConfig()

    if not config.sqs_queue_url:
        raise RuntimeError("CLOUD_SQS_QUEUE_URL must be set for the worker")
    if config.incident_store != "postgres" or not config.db_url:
        raise RuntimeError(
            "Worker requires CLOUD_INCIDENT_STORE=postgres and CLOUD_DB_URL to be set"
        )

    repo = PostgresIncidentRepository(config.db_url)
    await repo.initialize()

    embedder = get_embedder(config.embedder_backend)
    _qdrant: QdrantVectorStore | None = None
    vector_store: VectorStoreProtocol
    if config.vector_store_backend == "qdrant":
        _qdrant = QdrantVectorStore(config.qdrant_url)
        await _qdrant.initialize()
        vector_store = _qdrant
    else:
        vector_store = InMemoryVectorStore()

    retrieval_store = RetrievalStore(
        embedder,
        vector_store,
        repo,
        outcome_boost=config.cloud_retrieval_outcome_boost,
        failure_penalty=config.cloud_retrieval_failure_penalty,
    )

    llm_client = get_llm_client(config.llm_backend, api_key=config.anthropic_api_key)
    if llm_client is not None:
        llm_client = LLMThrottler(llm_client, config)

    validator = PlanValidator(
        CapabilityAdapter.load_default(), min_confidence=config.min_confidence
    )
    orchestrator = CloudOrchestrator(
        repo=repo,
        retrieval_store=retrieval_store,
        llm_client=llm_client,
        validator=validator,
    )

    sqs = SQSQueue(config.sqs_queue_url, config.sqs_region)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    loop.add_signal_handler(signal.SIGINT, stop.set)

    logger.info("worker: started — queue=%s region=%s", config.sqs_queue_url, config.sqs_region)

    try:
        while not stop.is_set():
            try:
                messages = await sqs.receive(
                    max_messages=1,
                    wait_seconds=_POLL_WAIT_SECONDS,
                    visibility_timeout=_VISIBILITY_TIMEOUT,
                )
                _HEARTBEAT_FILE.touch()
            except Exception:
                logger.exception("worker: SQS receive error — retrying in 5s")
                await asyncio.sleep(5)
                continue

            for msg in messages:
                if stop.is_set():
                    break
                try:
                    await _process(msg.body, orchestrator, sqs, msg.receipt_handle)
                except Exception:
                    logger.exception(
                        "worker: unhandled error for message_id=%s — message will reappear",
                        msg.message_id,
                    )
    finally:
        await repo.close()
        logger.info("worker: shutdown complete")


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
