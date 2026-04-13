"""Background service that periodically snapshots the Blackboard to a JSON file."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from cortexguard.edge.persistence.blackboard_snapshot import (
    CURRENT_SCHEMA_VERSION,
    BlackboardSnapshot,
)

if TYPE_CHECKING:
    from cortexguard.edge.models.blackboard import Blackboard

logger = logging.getLogger(__name__)


class PersistenceManager:
    """Manages periodic JSON snapshots of the Blackboard to disk.

    Write strategy: serialize to ``<path>.tmp``, then ``os.replace()`` for
    atomic crash-safe replacement on Linux.

    Args:
        blackboard: The shared Blackboard instance to snapshot.
        file_path: Destination file path for the JSON snapshot.
        snapshot_interval: Seconds between automatic snapshots (default 5.0).
    """

    def __init__(
        self,
        blackboard: Blackboard,
        file_path: Path,
        snapshot_interval: float = 5.0,
    ) -> None:
        self._blackboard = blackboard
        self._file_path = file_path
        self._tmp_path = Path(str(file_path) + ".tmp")
        self._snapshot_interval = snapshot_interval
        self._loop_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Create parent directories and launch the background snapshot loop."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._loop_task = asyncio.ensure_future(self._snapshot_loop())
        logger.info(
            "PersistenceManager started; snapshot interval=%.1fs path=%s",
            self._snapshot_interval,
            self._file_path,
        )

    async def stop(self) -> None:
        """Cancel the snapshot loop and flush a final snapshot."""
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        await self._persist_snapshot()
        logger.info("PersistenceManager stopped; final snapshot flushed to %s", self._file_path)

    async def restore(self) -> bool:
        """Load the JSON snapshot and hydrate the Blackboard.

        Returns:
            True if the snapshot was loaded successfully, False if the file is
            missing, unreadable, or has an incompatible schema version.
        """
        if not self._file_path.exists():
            logger.info("No persistence file found at %s; starting fresh", self._file_path)
            return False

        try:
            raw = self._file_path.read_text(encoding="utf-8")
            snapshot = BlackboardSnapshot.model_validate_json(raw)
        except Exception as exc:
            logger.warning("Failed to read/parse persistence file: %s", exc)
            return False

        if snapshot.schema_version != CURRENT_SCHEMA_VERSION:
            logger.warning(
                "Persistence file schema version %d does not match current version %d; "
                "ignoring stale snapshot",
                snapshot.schema_version,
                CURRENT_SCHEMA_VERSION,
            )
            return False

        await self._blackboard.restore_from_snapshot(snapshot)
        logger.info(
            "Blackboard restored from snapshot captured at %s", snapshot.captured_at.isoformat()
        )
        return True

    async def _snapshot_loop(self) -> None:
        """Background task: persist a snapshot every ``_snapshot_interval`` seconds."""
        try:
            while True:
                await asyncio.sleep(self._snapshot_interval)
                await self._persist_snapshot()
        except asyncio.CancelledError:
            pass

    async def _persist_snapshot(self) -> None:
        """Capture and atomically write a Blackboard snapshot to disk."""
        try:
            snapshot = await self._blackboard.capture_snapshot()
            json_bytes = snapshot.model_dump_json().encode("utf-8")
            self._tmp_path.write_bytes(json_bytes)
            os.replace(self._tmp_path, self._file_path)
            logger.debug("Snapshot persisted to %s", self._file_path)
        except Exception as exc:
            logger.error("Failed to persist Blackboard snapshot: %s", exc)
