"""API route that proxies inference requests to the SageMaker endpoint.

The edge calls this endpoint to get a model-based step outcome prediction.
Runs the synchronous SageMaker call in a thread pool to avoid blocking
the FastAPI event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)


class ClassifyRequest(BaseModel):
    device_id: str
    step_key: str
    sensor_snapshot: dict[str, Any]


class ClassifyResponse(BaseModel):
    predicted_outcome: str
    confidence: float
    model_id: str
    model_version: str | int


# Lazy boto3 client: not created at import time so tests and CI don't need AWS creds.
_sm_client: Any | None = None


def _get_sm_client() -> Any:
    global _sm_client
    if _sm_client is None:
        import boto3  # noqa: PLC0415

        _sm_client = boto3.client("sagemaker-runtime")
    return _sm_client


def _do_classify(payload: dict[str, Any], endpoint_name: str) -> ClassifyResponse:
    client = _get_sm_client()
    resp = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    body = resp["Body"].read().decode()
    parsed = json.loads(body)
    return ClassifyResponse(
        predicted_outcome=parsed.get("predicted_outcome", "completed"),
        confidence=parsed.get("confidence", 0.0),
        model_id=parsed.get("model_id", "unknown"),
        model_version=parsed.get("model_version", 0),
    )


def get_classify_router(
    _limiter: Limiter | None = None,
) -> APIRouter:
    lim = _limiter if _limiter is not None else Limiter(key_func=get_remote_address)
    router = APIRouter()

    endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")

    @router.post("/classify", response_model=ClassifyResponse)
    @lim.limit("100/minute")
    async def classify_step(request: Request, body: ClassifyRequest) -> ClassifyResponse:
        if not endpoint_name:
            raise HTTPException(status_code=503, detail="SageMaker endpoint not configured")

        try:
            result = await asyncio.to_thread(_do_classify, body.model_dump(), endpoint_name)
            return result
        except Exception as e:
            logger.warning("SageMaker classify failed", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Inference call failed: {e}") from e

    return router
