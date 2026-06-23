"""Postgres-backed step-telemetry repository using asyncpg connection pool."""

from __future__ import annotations

import asyncpg  # type: ignore[import-untyped]

from cortexguard.cloud.telemetry.models import StoredTelemetryRecord

_CREATE_TELEMETRY = """
CREATE TABLE IF NOT EXISTS step_telemetry (
    rowid                BIGSERIAL PRIMARY KEY,
    device_id            TEXT NOT NULL,
    key                  TEXT NOT NULL,
    outcome              TEXT NOT NULL,
    timestamp            TEXT NOT NULL,
    sensor_snapshot_json TEXT NOT NULL
)
"""

_INSERT_TELEMETRY = """
INSERT INTO step_telemetry (device_id, key, outcome, timestamp, sensor_snapshot_json)
VALUES ($1, $2, $3, $4, $5)
"""

_SELECT_RECENT = """
SELECT device_id, key, outcome, timestamp, sensor_snapshot_json
FROM step_telemetry
ORDER BY rowid DESC
LIMIT $1
"""


def _row_to_record(row: asyncpg.Record) -> StoredTelemetryRecord:
    from datetime import datetime

    return StoredTelemetryRecord(
        device_id=row["device_id"],
        key=row["key"],
        outcome=row["outcome"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        sensor_snapshot_json=row["sensor_snapshot_json"],
    )


class PostgresTelemetryRepository:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TELEMETRY)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "PostgresTelemetryRepository not initialized — call initialize() first"
            )
        return self._pool

    async def save_records(self, records: list[StoredTelemetryRecord]) -> None:
        async with self._get_pool().acquire() as conn:
            await conn.executemany(
                _INSERT_TELEMETRY,
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

    async def list_recent_records(self, limit: int) -> list[StoredTelemetryRecord]:
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(_SELECT_RECENT, limit)
            return [_row_to_record(r) for r in rows]
