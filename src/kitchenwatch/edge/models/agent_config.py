from typing import Any

from pydantic import BaseModel, Field

from .tool import Tool  # The runtime resource instance
from .tool_definition import ToolDefinition  # The LLM's callable function schema


class AgentConfig(BaseModel):
    """
    The central, static configuration manifest for the edge agent.
    This file defines all available resources, initial parameters, and system settings.
    """

    agent_id: str = Field(description="Unique identifier for this specific edge agent deployment.")

    # --- Tooling & Capabilities ---

    # The actual physical and software resources available to the agent.
    # The Execution Manager uses this list to know which devices to connect to.
    available_tools: list[Tool] = Field(
        description="A list of all initialized Tool instances (devices) controlled by this agent."
    )

    # The list of function schemas exposed to the LLM Planner.
    # The LLM uses this to determine what Actions it can generate.
    tool_definitions: list[ToolDefinition] = Field(
        description="A list of all callable capabilities (ToolDefinition schemas) available to the LLM Planner."
    )

    # --- Environment Parameters ---

    environment_name: str = Field(
        description="A human-readable name for the deployment location (e.g., 'Main Production Kitchen', 'Test Lab A')."
    )

    # Key operational settings that influence the agent's goals and safety
    operational_parameters: dict[str, Any] = Field(
        default={
            "default_safe_temp_c": 4.0,
            "max_allowed_temp_c": 10.0,
            "min_logging_interval_s": 10.0,
        },
        description="Global parameters governing safety thresholds and behavior.",
    )

    # --- External Interfaces ---

    logging_endpoint: str | None = Field(
        default=None, description="URL or address for centralized data logging and persistence."
    )

    telemetry_frequency_s: float = Field(
        default=5.0,
        description="How often (in seconds) the agent should report high-level state/telemetry.",
    )
