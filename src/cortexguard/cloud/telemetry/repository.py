"""Repository implementations for step-telemetry record persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import aiosqlite

from cortexguard.cloud.telemetry.models import StoredTelemetryRecord


@runtime_checkable
class TelemetryRepositoryProtocol(Protocol):
    async def save_records(self, records: list[StoredTelemetryRecord]) -> None: ...

    async def list_recent_records(self, limit: int) -> list[StoredTelemetryRecord]: ...


_CREATE_TELEMETRY = """
CREATE TABLE IF NOT EXISTS step_telemetry (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    key TEXT NOT NULL,
    outcome TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    sensor_snapshot_json TEXT NOT NULL
)
"""


class SQLiteTelemetryRepository:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_TELEMETRY)
            await db.commit()

    async def save_records(self, records: list[StoredTelemetryRecord]) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(
                "INSERT INTO step_telemetry (device_id, key, outcome, timestamp, sensor_snapshot_json) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        r.device_id,
                        r.key,
                        r.outcome,
                        r.timestamp.isoformat(),
                        r.sensor_snapshot_json,
                    )
                    for r in records
                ],
            )
            await db.commit()

    async def list_recent_records(self, limit: int) -> list[StoredTelemetryRecord]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT device_id, key, outcome, timestamp, sensor_snapshot_json "
                "FROM step_telemetry ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [_row_to_record(r) for r in rows]


def _row_to_record(row: aiosqlite.Row) -> StoredTelemetryRecord:
    from datetime import datetime

    return StoredTelemetryRecord(
        device_id=row[0],
        key=row[1],
        outcome=row[2],
        timestamp=datetime.fromisoformat(row[3]),
        sensor_snapshot_json=row[4],
    )


class InMemoryTelemetryRepository:
    def __init__(self) -> None:
        self._records: list[StoredTelemetryRecord] = []

    async def save_records(self, records: list[StoredTelemetryRecord]) -> None:
        self._records.extend(records)

    async def list_recent_records(self, limit: int) -> list[StoredTelemetryRecord]:
        return list(reversed(self._records))[:limit]
