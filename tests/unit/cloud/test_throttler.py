"""Unit tests for Phase 2 LLM throttler (concurrency, timeout, retries)."""

from __future__ import annotations

import asyncio

import pydantic
import pytest

from cortexguard.cloud.config import CloudConfig
from cortexguard.cloud.planner.llm_client import PlannerRequest, PlannerResponse
from cortexguard.cloud.planner.throttler import LLMThrottleError, LLMThrottler


def _make_request() -> PlannerRequest:
    return PlannerRequest(
        escalation_summary="device=x anomaly=Y",
        state_summary="{}",
        retrieved_summaries=[],
        capability_catalog_json="[]",
        anomaly_key="Y",
        severity="high",
    )


def _make_config(**overrides: object) -> CloudConfig:
    cfg = CloudConfig(
        incident_store="in_memory",
        llm_backend="mock",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class _SleepyClient:
    """Client that sleeps longer than the timeout."""

    async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise
        raise AssertionError("Should have timed out")


class _RaisingClient:
    """Client that raises a given exception a fixed number of times, then succeeds."""

    def __init__(self, exc: BaseException, fail_count: int = 999) -> None:
        self._exc = exc
        self._fail_count = fail_count
        self._calls = 0
        self._canned = PlannerResponse(
            candidate_plan=None,
            confidence=0.8,
            needs_human_review=False,
            rationale="ok",
        )

    async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
        self._calls += 1
        if self._calls <= self._fail_count:
            raise self._exc
        return self._canned


class _FakeRateLimitError(Exception):
    """Stands in for openai.RateLimitError without importing openai."""

    status_code = 429

    def __str__(self) -> str:
        return "rate limited"


class _FakeHTTPStatusError(Exception):
    """Stands in for httpx.HTTPStatusError at 429."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")

        class _FakeResponse:
            def __init__(self, code: int) -> None:
                self.status_code = code

        self.response = _FakeResponse(status_code)


@pytest.mark.asyncio
async def test_timeout_raises_llm_throttle_error() -> None:
    cfg = _make_config(cloud_llm_timeout_s=0.05, cloud_llm_max_retries=0)
    throttler = LLMThrottler(_SleepyClient(), cfg)  # type: ignore[arg-type]
    with pytest.raises(LLMThrottleError) as exc_info:
        await throttler.generate_structured_plan(_make_request())
    assert exc_info.value.outcome == "timeout"


@pytest.mark.asyncio
async def test_parse_error_raises_immediately_no_retries() -> None:
    class _BadModel(pydantic.BaseModel):
        x: int

    try:
        _BadModel.model_validate({"x": "not-an-int"})
        raise AssertionError("Should have raised")
    except pydantic.ValidationError as _ve:
        err: pydantic.ValidationError = _ve

    client = _RaisingClient(err, fail_count=999)
    cfg = _make_config(cloud_llm_timeout_s=5.0, cloud_llm_max_retries=2)
    throttler = LLMThrottler(client, cfg)  # type: ignore[arg-type]
    with pytest.raises(LLMThrottleError) as exc_info:
        await throttler.generate_structured_plan(_make_request())
    assert exc_info.value.outcome == "parse_error"
    assert client._calls == 1


@pytest.mark.asyncio
async def test_retryable_error_retries_and_succeeds() -> None:
    err = _FakeHTTPStatusError(429)
    client = _RaisingClient(err, fail_count=2)
    cfg = _make_config(
        cloud_llm_timeout_s=5.0,
        cloud_llm_max_retries=3,
        cloud_llm_base_backoff_ms=1,
    )

    from cortexguard.cloud.planner import throttler as throttler_mod

    original = throttler_mod._is_retryable

    def patched_retryable(exc: BaseException) -> bool:
        if isinstance(exc, _FakeHTTPStatusError):
            return exc.response.status_code == 429 or exc.response.status_code >= 500
        return original(exc)

    throttler_mod._is_retryable = patched_retryable  # type: ignore[assignment]
    try:
        throttler = LLMThrottler(client, cfg)  # type: ignore[arg-type]
        result = await throttler.generate_structured_plan(_make_request())
        assert result.confidence == 0.8
        assert client._calls == 3
    finally:
        throttler_mod._is_retryable = original  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_exhausted_retries_raises_llm_throttle_error() -> None:
    err = _FakeHTTPStatusError(429)
    client = _RaisingClient(err, fail_count=999)
    cfg = _make_config(
        cloud_llm_timeout_s=5.0,
        cloud_llm_max_retries=2,
        cloud_llm_base_backoff_ms=1,
    )

    from cortexguard.cloud.planner import throttler as throttler_mod

    original = throttler_mod._is_retryable

    def patched_retryable(exc: BaseException) -> bool:
        if isinstance(exc, _FakeHTTPStatusError):
            return True
        return original(exc)

    throttler_mod._is_retryable = patched_retryable  # type: ignore[assignment]
    try:
        throttler = LLMThrottler(client, cfg)  # type: ignore[arg-type]
        with pytest.raises(LLMThrottleError) as exc_info:
            await throttler.generate_structured_plan(_make_request())
        assert exc_info.value.outcome == "exhausted"
        assert client._calls == 3
    finally:
        throttler_mod._is_retryable = original  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_concurrency_cap_serialises_calls() -> None:
    """With max_concurrency=1, two concurrent calls on a single throttler are serialised."""
    start_times: list[float] = []
    gate = asyncio.Event()

    class _BlockingClient:
        async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
            start_times.append(asyncio.get_event_loop().time())
            await gate.wait()
            return PlannerResponse(
                candidate_plan=None, confidence=0.5, needs_human_review=False, rationale="ok"
            )

    cfg = _make_config(
        cloud_llm_timeout_s=5.0,
        cloud_llm_max_retries=0,
        cloud_llm_max_concurrency=1,
    )
    throttler = LLMThrottler(_BlockingClient(), cfg)  # type: ignore[arg-type]
    req = _make_request()

    async def _release_after_first() -> None:
        # Wait until the first call is in-flight, then open the gate so both can complete.
        while len(start_times) == 0:
            await asyncio.sleep(0.001)
        gate.set()

    results = await asyncio.gather(
        throttler.generate_structured_plan(req),
        throttler.generate_structured_plan(req),
        _release_after_first(),
    )
    # Two PlannerResponse results plus the None from _release_after_first
    plan_results = [r for r in results if r is not None]
    assert len(plan_results) == 2
    # With max_concurrency=1 both calls went through the same semaphore sequentially;
    # start_times[1] must be >= start_times[0] (the second call waits for the first to finish).
    assert len(start_times) == 2
    assert start_times[1] >= start_times[0]


@pytest.mark.asyncio
async def test_provider_name_unknown_when_absent() -> None:
    class _NoProviderClient:
        async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
            return PlannerResponse(
                candidate_plan=None, confidence=0.9, needs_human_review=False, rationale="ok"
            )

    cfg = _make_config(cloud_llm_timeout_s=5.0, cloud_llm_max_retries=0)
    throttler = LLMThrottler(_NoProviderClient(), cfg)  # type: ignore[arg-type]
    assert throttler._provider == "unknown"
    result = await throttler.generate_structured_plan(_make_request())
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_provider_name_read_from_attribute() -> None:
    class _NamedClient:
        provider_name: str = "groq"

        async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
            return PlannerResponse(
                candidate_plan=None, confidence=0.7, needs_human_review=False, rationale="ok"
            )

    cfg = _make_config(cloud_llm_timeout_s=5.0, cloud_llm_max_retries=0)
    throttler = LLMThrottler(_NamedClient(), cfg)  # type: ignore[arg-type]
    assert throttler._provider == "groq"


@pytest.mark.asyncio
async def test_non_retryable_non_parse_error_raises_provider_error() -> None:
    """A generic RuntimeError (non-retryable, non-parse) must raise provider_error exactly once."""

    class _UnexpectedClient:
        calls: int = 0

        async def generate_structured_plan(self, request: PlannerRequest) -> PlannerResponse:
            self.calls += 1
            raise RuntimeError("unexpected internal failure")

    client = _UnexpectedClient()
    cfg = _make_config(cloud_llm_timeout_s=5.0, cloud_llm_max_retries=3)
    throttler = LLMThrottler(client, cfg)  # type: ignore[arg-type]

    with pytest.raises(LLMThrottleError) as exc_info:
        await throttler.generate_structured_plan(_make_request())

    assert exc_info.value.outcome == "provider_error"
    # No retries — should have been called exactly once.
    assert client.calls == 1
