"""Postgres-backed incident and outcome repository using asyncpg connection pool."""

from __future__ import annotations

import asyncpg  # type: ignore[import-untyped]

from cortexguard.cloud.persistence.models import IncidentRecord, OutcomeRecord

_CREATE_INCIDENTS = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id            TEXT PRIMARY KEY,
    escalation_id          TEXT NOT NULL,
    trace_id               TEXT NOT NULL,
    device_id              TEXT NOT NULL,
    anomaly_key            TEXT NOT NULL,
    anomaly_type           TEXT NOT NULL,
    severity               TEXT NOT NULL,
    summary                TEXT NOT NULL,
    raw_packet_json        TEXT NOT NULL,
    retrieved_incident_ids_json TEXT NOT NULL,
    candidate_plan_json    TEXT,
    validation_errors_json TEXT NOT NULL,
    decision               TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    rationale              TEXT,
    confidence             DOUBLE PRECISION,
    parent_incident_id     TEXT,
    source                 TEXT NOT NULL DEFAULT 'edge',
    operator_resolution_json TEXT,
    retrieved_incidents_json TEXT
)
"""

_CREATE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id    TEXT PRIMARY KEY,
    escalation_id TEXT NOT NULL,
    decision_id   TEXT NOT NULL,
    device_id     TEXT NOT NULL,
    status        TEXT NOT NULL,
    completed_at  TEXT NOT NULL,
    notes         TEXT,
    failure_reason TEXT,
    linked_at     TEXT NOT NULL
)
"""

_CREATE_INCIDENTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_incidents_trace_id ON incidents (trace_id)
"""

_UPSERT_INCIDENT = """
INSERT INTO incidents VALUES (
    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20
)
ON CONFLICT (incident_id) DO UPDATE SET
    escalation_id              = EXCLUDED.escalation_id,
    trace_id                   = EXCLUDED.trace_id,
    device_id                  = EXCLUDED.device_id,
    anomaly_key                = EXCLUDED.anomaly_key,
    anomaly_type               = EXCLUDED.anomaly_type,
    severity                   = EXCLUDED.severity,
    summary                    = EXCLUDED.summary,
    raw_packet_json            = EXCLUDED.raw_packet_json,
    retrieved_incident_ids_json = EXCLUDED.retrieved_incident_ids_json,
    candidate_plan_json        = EXCLUDED.candidate_plan_json,
    validation_errors_json     = EXCLUDED.validation_errors_json,
    decision                   = EXCLUDED.decision,
    created_at                 = EXCLUDED.created_at,
    rationale                  = EXCLUDED.rationale,
    confidence                 = EXCLUDED.confidence,
    parent_incident_id         = EXCLUDED.parent_incident_id,
    source                     = EXCLUDED.source,
    operator_resolution_json   = EXCLUDED.operator_resolution_json,
    retrieved_incidents_json   = EXCLUDED.retrieved_incidents_json
"""

_UPSERT_OUTCOME = """
INSERT INTO outcomes VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
ON CONFLICT (outcome_id) DO UPDATE SET
    escalation_id  = EXCLUDED.escalation_id,
    decision_id    = EXCLUDED.decision_id,
    device_id      = EXCLUDED.device_id,
    status         = EXCLUDED.status,
    completed_at   = EXCLUDED.completed_at,
    notes          = EXCLUDED.notes,
    failure_reason = EXCLUDED.failure_reason,
    linked_at      = EXCLUDED.linked_at
"""


def _row_to_incident(row: asyncpg.Record) -> IncidentRecord:
    from datetime import datetime

    return IncidentRecord(
        incident_id=row["incident_id"],
        escalation_id=row["escalation_id"],
        trace_id=row["trace_id"],
        device_id=row["device_id"],
        anomaly_key=row["anomaly_key"],
        anomaly_type=row["anomaly_type"],
        severity=row["severity"],
        summary=row["summary"],
        raw_packet_json=row["raw_packet_json"],
        retrieved_incident_ids_json=row["retrieved_incident_ids_json"],
        candidate_plan_json=row["candidate_plan_json"],
        validation_errors_json=row["validation_errors_json"],
        decision=row["decision"],
        created_at=datetime.fromisoformat(row["created_at"]),
        rationale=row["rationale"],
        confidence=float(row["confidence"]) if row["confidence"] is not None else None,
        parent_incident_id=row["parent_incident_id"],
        source=row["source"] or "edge",
        operator_resolution_json=row["operator_resolution_json"],
        retrieved_incidents_json=row["retrieved_incidents_json"],
    )


def _row_to_outcome(row: asyncpg.Record) -> OutcomeRecord:
    from datetime import datetime

    return OutcomeRecord(
        outcome_id=row["outcome_id"],
        escalation_id=row["escalation_id"],
        decision_id=row["decision_id"],
        device_id=row["device_id"],
        status=row["status"],
        completed_at=datetime.fromisoformat(row["completed_at"]),
        notes=row["notes"],
        failure_reason=row["failure_reason"],
        linked_at=datetime.fromisoformat(row["linked_at"]),
    )


class PostgresIncidentRepository:
    """Postgres-backed repository — production replacement for SQLiteIncidentRepository.

    Uses a connection pool created on ``initialize()``. Call ``close()`` on shutdown.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_INCIDENTS)
            await conn.execute(_CREATE_INCIDENTS_INDEX)
            await conn.execute(_CREATE_OUTCOMES)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "PostgresIncidentRepository not initialized — call initialize() first"
            )
        return self._pool

    async def save_incident(self, record: IncidentRecord) -> None:
        async with self._get_pool().acquire() as conn:
            await conn.execute(
                _UPSERT_INCIDENT,
                record.incident_id,
                record.escalation_id,
                record.trace_id,
                record.device_id,
                record.anomaly_key,
                record.anomaly_type,
                record.severity,
                record.summary,
                record.raw_packet_json,
                record.retrieved_incident_ids_json,
                record.candidate_plan_json,
                record.validation_errors_json,
                record.decision,
                record.created_at.isoformat(),
                record.rationale,
                record.confidence,
                record.parent_incident_id,
                record.source,
                record.operator_resolution_json,
                record.retrieved_incidents_json,
            )

    async def get_incident(self, incident_id: str) -> IncidentRecord | None:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM incidents WHERE incident_id = $1", incident_id)
            return _row_to_incident(row) if row is not None else None

    async def get_incident_by_trace_id(self, trace_id: str) -> IncidentRecord | None:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM incidents WHERE trace_id = $1 ORDER BY created_at DESC LIMIT 1",
                trace_id,
            )
            return _row_to_incident(row) if row is not None else None

    async def save_outcome(self, record: OutcomeRecord) -> None:
        async with self._get_pool().acquire() as conn:
            await conn.execute(
                _UPSERT_OUTCOME,
                record.outcome_id,
                record.escalation_id,
                record.decision_id,
                record.device_id,
                record.status,
                record.completed_at.isoformat(),
                record.notes,
                record.failure_reason,
                record.linked_at.isoformat(),
            )

    async def get_outcome_by_escalation(self, escalation_id: str) -> OutcomeRecord | None:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM outcomes WHERE escalation_id = $1", escalation_id
            )
            return _row_to_outcome(row) if row is not None else None

    async def list_recent_outcomes(self, limit: int) -> list[OutcomeRecord]:
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM outcomes ORDER BY completed_at DESC LIMIT $1", limit
            )
            return [_row_to_outcome(r) for r in rows]

    async def list_recent_incidents(self, limit: int) -> list[IncidentRecord]:
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM incidents ORDER BY created_at DESC LIMIT $1", limit
            )
            return [_row_to_incident(r) for r in rows]

    async def update_operator_resolution(self, incident_id: str, resolution_json: str) -> bool:
        async with self._get_pool().acquire() as conn:
            result = await conn.execute(
                "UPDATE incidents SET operator_resolution_json = $1 WHERE incident_id = $2",
                resolution_json,
                incident_id,
            )
            return str(result) == "UPDATE 1"
