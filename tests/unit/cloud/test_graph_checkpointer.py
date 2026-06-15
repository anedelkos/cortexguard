from __future__ import annotations

from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from cortexguard.cloud.graph.workflow import create_checkpointer


@pytest.mark.asyncio
async def test_create_checkpointer_memory() -> None:
    saver = await create_checkpointer("memory")
    assert isinstance(saver, MemorySaver)


@pytest.mark.asyncio
async def test_create_checkpointer_unknown_fallback_to_memory() -> None:
    saver = await create_checkpointer("unknown-backend")
    assert isinstance(saver, MemorySaver)


@pytest.mark.asyncio
async def test_create_checkpointer_sqlite_import_error_fallback() -> None:
    with patch("cortexguard.cloud.graph.workflow.MemorySaver") as mock_ms:
        instance = MemorySaver()
        mock_ms.return_value = instance
        import aiosqlite  # noqa: F401

        with patch.dict("sys.modules", {"aiosqlite": None}):
            saver = await create_checkpointer("sqlite")
    assert isinstance(saver, MemorySaver)


@pytest.mark.asyncio
async def test_create_checkpointer_postgres_import_error_fallback() -> None:
    with patch.dict("sys.modules", {"asyncpg": None}):
        saver = await create_checkpointer("postgres")
    assert isinstance(saver, MemorySaver)
