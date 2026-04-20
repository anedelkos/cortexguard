"""Repository implementations for cloud incident and outcome persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import aiosqlite

from cortexguard.cloud.persistence.models import IncidentRecord, OutcomeRecord


@runtime_checkable
class IncidentRepositoryProtocol(Protocol):
    async def save_incident(self, record: IncidentRecord) -> None: ...

    async def get_incident(self, incident_id: str) -> IncidentRecord | None: ...

    async def save_outcome(self, record: OutcomeRecord) -> None: ...

    async def get_outcome_by_escalation(self, escalation_id: str) -> OutcomeRecord | None: ...

    async def list_recent_incidents(self, limit: int) -> list[IncidentRecord]: ...


_CREATE_INCIDENTS = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    escalation_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    anomaly_key TEXT NOT NULL,
    anomaly_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    summary TEXT NOT NULL,
    raw_packet_json TEXT NOT NULL,
    retrieved_incident_ids_json TEXT NOT NULL,
    candidate_plan_json TEXT,
    validation_errors_json TEXT NOT NULL,
    decision TEXT NOT NULL,
    created_at TEXT NOT NULL,
    rationale TEXT,
    confidence REAL
)
"""

_CREATE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id TEXT PRIMARY KEY,
    escalation_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    status TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    notes TEXT,
    failure_reason TEXT,
    linked_at TEXT NOT NULL
)
"""


def _row_to_incident(row: aiosqlite.Row) -> IncidentRecord:
    from datetime import datetime

    return IncidentRecord(
        incident_id=row[0],
        escalation_id=row[1],
        trace_id=row[2],
        device_id=row[3],
        anomaly_key=row[4],
        anomaly_type=row[5],
        severity=row[6],
        summary=row[7],
        raw_packet_json=row[8],
        retrieved_incident_ids_json=row[9],
        candidate_plan_json=row[10],
        validation_errors_json=row[11],
        decision=row[12],
        created_at=datetime.fromisoformat(row[13]),
        rationale=row[14],
        confidence=float(row[15]) if row[15] is not None else None,
    )


def _row_to_outcome(row: aiosqlite.Row) -> OutcomeRecord:
    from datetime import datetime

    return OutcomeRecord(
        outcome_id=row[0],
        escalation_id=row[1],
        decision_id=row[2],
        device_id=row[3],
        status=row[4],
        completed_at=datetime.fromisoformat(row[5]),
        notes=row[6],
        failure_reason=row[7],
        linked_at=datetime.fromisoformat(row[8]),
    )


class SQLiteIncidentRepository:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_INCIDENTS)
            await db.execute(_CREATE_OUTCOMES)
            # Migrate existing DBs that predate rationale/confidence columns
            for col, typedef in (("rationale", "TEXT"), ("confidence", "REAL")):
                try:
                    await db.execute(f"ALTER TABLE incidents ADD COLUMN {col} {typedef}")
                except aiosqlite.OperationalError as exc:
                    # SQLite raises "duplicate column name" when the migration has
                    # already been applied. Re-raise anything else.
                    if "duplicate column name" not in str(exc).lower():
                        raise
            await db.commit()

    async def save_incident(self, record: IncidentRecord) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO incidents VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
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
                ),
            )
            await db.commit()

    async def get_incident(self, incident_id: str) -> IncidentRecord | None:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return _row_to_incident(row) if row is not None else None

    async def save_outcome(self, record: OutcomeRecord) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO outcomes VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    record.outcome_id,
                    record.escalation_id,
                    record.decision_id,
                    record.device_id,
                    record.status,
                    record.completed_at.isoformat(),
                    record.notes,
                    record.failure_reason,
                    record.linked_at.isoformat(),
                ),
            )
            await db.commit()

    async def get_outcome_by_escalation(self, escalation_id: str) -> OutcomeRecord | None:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT * FROM outcomes WHERE escalation_id = ?", (escalation_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return _row_to_outcome(row) if row is not None else None

    async def list_recent_incidents(self, limit: int) -> list[IncidentRecord]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT * FROM incidents ORDER BY rowid DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [_row_to_incident(r) for r in rows]


class InMemoryIncidentRepository:
    def __init__(self) -> None:
        self._incidents: dict[str, IncidentRecord] = {}
        self._outcomes: dict[str, OutcomeRecord] = {}
        self._incident_order: list[str] = []

    async def save_incident(self, record: IncidentRecord) -> None:
        if record.incident_id not in self._incidents:
            self._incident_order.append(record.incident_id)
        self._incidents[record.incident_id] = record

    async def get_incident(self, incident_id: str) -> IncidentRecord | None:
        return self._incidents.get(incident_id)

    async def save_outcome(self, record: OutcomeRecord) -> None:
        self._outcomes[record.outcome_id] = record

    async def get_outcome_by_escalation(self, escalation_id: str) -> OutcomeRecord | None:
        for record in self._outcomes.values():
            if record.escalation_id == escalation_id:
                return record
        return None

    async def list_recent_incidents(self, limit: int) -> list[IncidentRecord]:
        return [self._incidents[iid] for iid in reversed(self._incident_order)][:limit]
