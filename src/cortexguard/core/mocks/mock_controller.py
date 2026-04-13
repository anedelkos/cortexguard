from __future__ import annotations

import asyncio
import logging
from typing import Any

from cortexguard.core.interfaces.base_controller import BaseController

logger = logging.getLogger(__name__)


class MockController(BaseController):
    """A mock hardware controller for development and testing."""

    def __init__(
        self,
        fail_on: set[str] | None = None,
        delay: float = 0.05,
        custom_logger: logging.Logger = logger,
    ) -> None:
        self._fail_on = fail_on or set()
        self._delay = delay
        self._logger = custom_logger

    async def execute(self, primitive_name: str, parameters: dict[str, Any]) -> None:
        if primitive_name in self._fail_on:
            self._logger.warning(f"[MOCK] Simulated failure for {primitive_name}")
            raise RuntimeError(f"Simulated failure for {primitive_name}")

        self._logger.info(
            f"[MOCK] Executing primitive '{primitive_name}' with parameters {parameters}"
        )
        await asyncio.sleep(self._delay)  # simulate execution time
