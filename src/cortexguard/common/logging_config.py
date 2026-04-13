from __future__ import annotations

import logging
import os
import sys

from opentelemetry import trace

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | trace_id=%(trace_id)s | %(message)s"
JSON_LOG_FORMAT = (
    '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s",'
    ' "trace_id": "%(trace_id)s", "span_id": "%(span_id)s", "msg": "%(message)s"}'
)


class OtelTraceContextFilter(logging.Filter):
    """Injects the active OTEL trace_id and span_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            record.trace_id = format(ctx.trace_id, "032x")
            record.span_id = format(ctx.span_id, "016x")
        else:
            record.trace_id = "0" * 32
            record.span_id = "0" * 16
        return True


def setup_logging(level: str | None = None, json: bool | None = None) -> None:
    """
    Unified logging setup for all CortexGuard components.
    - Level defaults to INFO
    - JSON output if running in Docker or explicitly requested
    - OTEL trace_id / span_id injected into every log record for log/trace correlation
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    use_json = json if json is not None else os.getenv("LOG_JSON", "false").lower() == "true"

    # Clear existing handlers (important when uvicorn reloads)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(JSON_LOG_FORMAT if use_json else LOG_FORMAT))
    handler.addFilter(OtelTraceContextFilter())

    logging.basicConfig(level=log_level, handlers=[handler], force=True)

    # Tweak noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    logging.getLogger("cortexguard").info(
        f"✅ Logging initialized (level={log_level}, json={use_json})"
    )
