import json
from typing import Any

import pytest
from jsonschema import ValidationError as SchemaValidationError
from jsonschema import validate
from pydantic import ValidationError

from cortexguard.common.constants import FUSED_EVENT_SCHEMA, SAMPLE_DATA_DIR
from cortexguard.simulation.models.fused_record import FusedRecord

# --- Fixtures -----------------------------------------------------------


@pytest.fixture
def record() -> FusedRecord:
    """Return a valid FusedRecord instance."""
    return FusedRecord(
        timestamp_ns=1234567890,
        rgb_path=str(SAMPLE_DATA_DIR / "subject_1_bagel/rgb_images/1/image_001.png"),
        depth_path=None,
        force_x=1.0,
        force_y=2.0,
        force_z=3.0,
        torque_x=0.1,
        torque_y=0.2,
        torque_z=0.3,
        pos_x=0.0,
        pos_y=0.0,
        pos_z=0.0,
    )


@pytest.fixture
def fused_schema() -> Any:
    """Load the JSON schema for fused events dynamically."""
    with FUSED_EVENT_SCHEMA.open() as f:
        return json.load(f)


# --- Tests --------------------------------------------------------------


def test_fused_record_valid_construction(record: FusedRecord) -> None:
    """Check that fields are correctly initialized."""
    assert record.rgb_path.endswith(".png")
    assert record.force_z == 3.0
    assert record.depth_path is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("force_x", "not-a-float"),
        ("timestamp_ns", "not-an-int"),
        ("pos_y", "not-a-float"),
    ],
)
def test_fused_record_invalid_field(field: str, value: Any) -> None:
    """Ensure invalid field types raise ValidationError."""
    data = {
        "timestamp_ns": 1234567890,
        "rgb_path": str(SAMPLE_DATA_DIR / "subject_1_bagel/rgb_images/1/image_001.png"),
        "depth_path": None,
        "force_x": 1.0,
        "force_y": 2.0,
        "force_z": 3.0,
        "torque_x": 0.1,
        "torque_y": 0.2,
        "torque_z": 0.3,
        "pos_x": 0.0,
        "pos_y": 0.0,
        "pos_z": 0.0,
    }
    data[field] = value
    with pytest.raises(ValidationError) as exception_info:
        FusedRecord(**data)  # type: ignore[arg-type]
    assert field in str(exception_info.value)


def test_fused_record_serialization(record: FusedRecord) -> None:
    """Test model_dump and model_validate round-trip."""
    dumped: dict[str, Any] = record.model_dump(mode="python")
    reloaded = FusedRecord.model_validate(dumped)
    assert reloaded == record


def test_sample_fused_record_schema(fused_schema: dict[str, Any]) -> None:
    """Validate a sample fused record against the JSON schema."""
    record = FusedRecord(
        timestamp_ns=1637705108501665831,
        rgb_path=str(SAMPLE_DATA_DIR / "subject_1_bagel/rgb_images/1/1637705108501665831_rgb.jpg"),
        depth_path=str(
            SAMPLE_DATA_DIR / "subject_1_bagel/depth_images/1/1637705108501665831_depth.png"
        ),
        force_x=1.23,
        force_y=0.45,
        force_z=-0.67,
        torque_x=0.12,
        torque_y=0.34,
        torque_z=0.56,
        pos_x=0.01,
        pos_y=0.02,
        pos_z=0.03,
    )

    try:
        validate(instance=record.model_dump(), schema=fused_schema)
    except SchemaValidationError as e:
        pytest.fail(f"Schema validation failed: {e}")
