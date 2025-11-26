from typing import Any

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Defines a capability the agent can execute."""

    id: str = Field(description="Unique ID for the tool (matches Action.tool_id).")
    name: str = Field(description="Human-readable name for the tool.")
    description: str = Field(description="Detailed explanation for the LLM on what the tool does.")
    # Schema defining the expected arguments for the tool's function
    parameter_schema: dict[str, Any] = Field(
        description="JSON Schema defining the tool's input arguments."
    )
