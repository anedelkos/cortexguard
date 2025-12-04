from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SceneObject(BaseModel):
    id: str = Field(..., description="Unique ID for the object")
    label: str = Field(..., description="Semantic label")
    location_2d: list[float] | None = Field(
        None, description="Normalized bbox [xmin,ymin,xmax,ymax]"
    )
    pose_3d: list[float] | None = Field(None, description="3D pose [x,y,z,roll,pitch,yaw]")
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("location_2d")
    def _check_bbox(cls: "SceneObject", v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError("location_2d must have 4 elements")
        return v

    @field_validator("pose_3d")
    def _check_pose(cls: "SceneObject", v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if len(v) != 6:
            raise ValueError("pose_3d must have 6 elements")
        return v


class SceneRelationship(BaseModel):
    source_id: str
    relationship: str
    target_id: str


class SceneGraph(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    objects: list[SceneObject] = Field(default_factory=list)
    relationships: list[SceneRelationship] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
