from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cortexguard.cloud.schemas.responses import (
    ExecutionOutcome,
    ExecutionOutcomeStatus,
    MaydayAcceptedResponse,
)


def _make_outcome(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "escalation_id": str(uuid.uuid4()),
        "decision_id": str(uuid.uuid4()),
        "device_id": "robot-arm-01",
        "status": ExecutionOutcomeStatus.completed,
        "completed_at": datetime.now(UTC),
    }
    base.update(overrides)
    return base


class TestMaydayAcceptedResponse:
    def test_serializes_to_trace_id_only(self) -> None:
        trace = str(uuid.uuid4())
        response = MaydayAcceptedResponse(trace_id=trace)
        assert response.model_dump() == {"trace_id": trace}

    def test_json_roundtrip(self) -> None:
        trace = str(uuid.uuid4())
        response = MaydayAcceptedResponse(trace_id=trace)
        assert MaydayAcceptedResponse.model_validate_json(response.model_dump_json()) == response


class TestExecutionOutcomeStatus:
    def test_all_four_values_accepted(self) -> None:
        for status in (
            ExecutionOutcomeStatus.completed,
            ExecutionOutcomeStatus.failed,
            ExecutionOutcomeStatus.aborted,
            ExecutionOutcomeStatus.rejected_by_operator,
        ):
            outcome = ExecutionOutcome.model_validate(_make_outcome(status=status))
            assert outcome.status is status

    def test_string_value_coercion(self) -> None:
        for raw in ("completed", "failed", "aborted", "rejected_by_operator"):
            outcome = ExecutionOutcome.model_validate(_make_outcome(status=raw))
            assert outcome.status == ExecutionOutcomeStatus(raw)


class TestExecutionOutcomeRequiredFields:
    @pytest.mark.parametrize(
        "missing_field",
        ["escalation_id", "decision_id", "device_id", "status", "completed_at"],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        data = _make_outcome()
        del data[missing_field]
        with pytest.raises(ValidationError):
            ExecutionOutcome.model_validate(data)

    def test_optional_fields_default_to_none(self) -> None:
        outcome = ExecutionOutcome.model_validate(_make_outcome())
        assert outcome.notes is None
        assert outcome.failure_reason is None

    def test_optional_fields_accepted_when_provided(self) -> None:
        outcome = ExecutionOutcome.model_validate(
            _make_outcome(
                notes="recovered after retry",
                failure_reason=None,
                status=ExecutionOutcomeStatus.completed,
            )
        )
        assert outcome.notes == "recovered after retry"
