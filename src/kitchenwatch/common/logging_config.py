import logging
import os
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
JSON_LOG_FORMAT = (
    '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "msg": "%(message)s"}'
)


def setup_logging(level: str | None = None, json: bool | None = None) -> None:
    """
    Unified logging setup for all KitchenWatch components.
    - Level defaults to INFO
    - JSON output if running in Docker or explicitly requested
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    use_json = json if json is not None else os.getenv("LOG_JSON", "false").lower() == "true"

    # Clear existing handlers (important when uvicorn reloads)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(JSON_LOG_FORMAT if use_json else LOG_FORMAT))

    logging.basicConfig(level=log_level, handlers=[handler], force=True)

    # Tweak noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    logging.getLogger("kitchenwatch").info(
        f"✅ Logging initialized (level={log_level}, json={use_json})"
    )
