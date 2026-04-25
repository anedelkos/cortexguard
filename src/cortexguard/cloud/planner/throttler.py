"""Concurrency-limiting and retry wrapper for outbound LLM client calls."""

from __future__ import annotations

import asyncio
import logging
import random

import pydantic

from cortexguard.cloud.config import CloudConfig
from cortexguard.cloud.planner.llm_client import LLMClientProtocol, PlannerRequest, PlannerResponse

logger = logging.getLogger(__name__)


class LLMThrottleError(Exception):
    """Raised when the throttler cannot fulfil the LLM request."""

    def __init__(self, outcome: str) -> None:
        """Initialise with a short outcome label (timeout, exhausted, parse_error)."""
        super().__init__(outcome)
        self.outcome = outcome


def _is_retryable(exc: BaseException) -> bool:
    """Return True for HTTP 429 / 5xx errors that warrant a retry."""
    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            return bool(exc.response.status_code == 429 or exc.response.status_code >= 500)
    except ImportError:
        pass

    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIStatusError):
            return bool(exc.status_code >= 500)
    except ImportError:
        pass

    return False


def _is_parse_error(exc: BaseException) -> bool:
    """Return True for instructor / pydantic parse failures that should not be retried."""
    if isinstance(exc, pydantic.ValidationError):
        return True
    try:
        from instructor.core import IncompleteOutputException as _IncompleteOutputException

        if isinstance(exc, _IncompleteOutputException):
            return True
    except (ImportError, AttributeError):
        pass
    return False


class LLMThrottler:
    """Wraps an LLMClientProtocol with concurrency limiting, timeout, and retry logic."""

    def __init__(self, client: LLMClientProtocol, config: CloudConfig) -> None:
        """Initialise the throttler from an LLM client and cloud config."""
        self._client = client
        self._timeout_s = config.cloud_llm_timeout_s
        self._max_retries = config.cloud_llm_max_retries
        self._base_backoff_ms = config.cloud_llm_base_backoff_ms
        self._semaphore = asyncio.Semaphore(config.cloud_llm_max_concurrency)
        self._provider: str = getattr(client, "provider_name", "unknown")
        self._inflight: int = 0
        self._inflight_lock = asyncio.Lock()

    def _inc(self, metric_name: str, **labels: str) -> None:
        try:
            from cortexguard.cloud import runtime as _rt

            metric = getattr(_rt, metric_name, None)
            if metric is not None:
                metric.labels(**labels).inc()
        except ImportError:
            logger.debug("Cloud runtime unavailable for metric %s", metric_name)

    def _set_gauge(self, metric_name: str, value: float, **labels: str) -> None:
        try:
            from cortexguard.cloud import runtime as _rt

            metric = getattr(_rt, metric_name, None)
            if metric is not None:
                metric.labels(**labels).set(value)
        except ImportError:
            logger.debug("Cloud runtime unavailable for gauge %s", metric_name)

    async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
        """Delegate to the wrapped client with concurrency cap, timeout, and retries."""
        provider = self._provider
        async with self._semaphore:
            async with self._inflight_lock:
                self._inflight += 1
                self._set_gauge("cloud_llm_inflight", float(self._inflight), provider=provider)
            try:
                return await self._call_with_retries(request, provider)
            finally:
                async with self._inflight_lock:
                    self._inflight -= 1
                    self._set_gauge("cloud_llm_inflight", float(self._inflight), provider=provider)

    async def _call_with_retries(self, request: PlannerRequest, provider: str) -> PlannerResponse:
        attempt = 0
        last_exc: BaseException | None = None

        while attempt <= self._max_retries:
            try:
                response = await asyncio.wait_for(
                    self._client.generate_structured_plan(request),
                    timeout=self._timeout_s,
                )
                self._inc("cloud_llm_requests_total", provider=provider, outcome="success")
                return response

            except TimeoutError as te:
                logger.warning("LLM call timed out (provider=%s attempt=%d)", provider, attempt)
                self._inc("cloud_llm_requests_total", provider=provider, outcome="timeout")
                raise LLMThrottleError("timeout") from te

            except BaseException as exc:
                if _is_parse_error(exc):
                    logger.warning(
                        "LLM parse error (provider=%s): %s", provider, exc, exc_info=True
                    )
                    self._inc("cloud_llm_requests_total", provider=provider, outcome="parse_error")
                    raise LLMThrottleError("parse_error") from exc

                if _is_retryable(exc):
                    last_exc = exc
                    if attempt < self._max_retries:
                        backoff_s = (
                            self._base_backoff_ms
                            * (2**attempt)
                            * (1 + random.random())  # nosec B311
                            / 1000.0
                        )
                        logger.warning(
                            "LLM retryable error (provider=%s attempt=%d), "
                            "retrying in %.2fs: %s",
                            provider,
                            attempt,
                            backoff_s,
                            exc,
                        )
                        self._inc("cloud_llm_retries_total", provider=provider)
                        await asyncio.sleep(backoff_s)
                        attempt += 1
                        continue

                    # Retries exhausted
                    logger.error(
                        "LLM retries exhausted (provider=%s max_retries=%d): %s",
                        provider,
                        self._max_retries,
                        exc,
                    )
                    self._inc("cloud_llm_requests_total", provider=provider, outcome="exhausted")
                    raise LLMThrottleError("exhausted") from last_exc

                # Non-retryable, non-parse error — propagate as provider_error
                logger.error("LLM provider error (provider=%s): %s", provider, exc, exc_info=True)
                self._inc("cloud_llm_requests_total", provider=provider, outcome="provider_error")
                raise LLMThrottleError("provider_error") from exc

        # Should be unreachable, but keeps mypy happy
        self._inc("cloud_llm_requests_total", provider=provider, outcome="exhausted")
        raise LLMThrottleError("exhausted") from last_exc
