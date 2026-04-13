from __future__ import annotations

import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider

from cortexguard.common.logging_config import OtelTraceContextFilter, setup_logging


class TestOtelTraceContextFilter:
    def test_injects_zeros_when_no_active_span(self) -> None:
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        OtelTraceContextFilter().filter(record)
        assert record.trace_id == "0" * 32  # type: ignore[attr-defined]
        assert record.span_id == "0" * 16  # type: ignore[attr-defined]

    def test_injects_trace_ids_when_span_is_active(self) -> None:
        provider = TracerProvider()
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("test-span") as span:
            ctx = span.get_span_context()
            record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
            OtelTraceContextFilter().filter(record)

        assert record.trace_id == format(ctx.trace_id, "032x")  # type: ignore[attr-defined]
        assert record.span_id == format(ctx.span_id, "016x")  # type: ignore[attr-defined]
        assert len(record.trace_id) == 32  # type: ignore[attr-defined]
        assert len(record.span_id) == 16  # type: ignore[attr-defined]

    def test_filter_always_returns_true(self) -> None:
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert OtelTraceContextFilter().filter(record) is True


class TestSetupLogging:
    def test_handler_has_otel_filter(self) -> None:
        setup_logging(level="WARNING")
        root = logging.getLogger()
        filters = [f for h in root.handlers for f in h.filters]
        assert any(isinstance(f, OtelTraceContextFilter) for f in filters)

    def test_log_record_has_trace_id_attribute(self, caplog: pytest.LogCaptureFixture) -> None:
        setup_logging(level="DEBUG")
        with caplog.at_level(logging.DEBUG, logger="cortexguard.test"):
            logging.getLogger("cortexguard.test").info("hello")
        record = logging.LogRecord("cortexguard.test", logging.INFO, "", 0, "hello", (), None)
        OtelTraceContextFilter().filter(record)
        assert hasattr(record, "trace_id")
        assert hasattr(record, "span_id")
