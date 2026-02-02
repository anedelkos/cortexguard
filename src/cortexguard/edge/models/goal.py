from enum import Enum

from pydantic import BaseModel, Field


class GoalStatus(str, Enum):
    """Enumeration for the state of a Goal."""

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class GoalContext(BaseModel):
    """
    The static context linking the plan back to the user's initial goal.
    This is used in the Plan blueprint validation.
    """

    goal_id: str = Field(description="The unique ID of the original high-level goal.")
    user_prompt: str = Field(description="The user's original request text.")
    intent: str = Field(
        description="The derived high-level purpose or objective of the plan (e.g., 'Make pasta', 'Remediate Anomaly X')."
    )
    priority: int = Field(default=5, description="Priority level (1=highest, 10=lowest).")
