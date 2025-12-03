import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FunctionSchema(BaseModel):
    """Represents the JSON Schema definition of a single LLM-callable function."""

    description: str = Field(
        ...,
        description="A clear, concise description for the LLM to understand the function's purpose.",
    )
    parameters: dict[str, Any] = Field(
        ...,
        description="The JSON Schema definition for the function's arguments (properties, required).",
    )
    risk_level: str = Field(
        "LOW",
        description="The operational risk associated with using this tool (e.g., LOW, MEDIUM, HIGH, E-STOP). This helps the LLM choose safer paths.",
    )
    pre_conditions: list[str] = Field(
        default_factory=list,
        description="List of simple environmental state conditions that must be true before calling this tool (e.g., 'Power is ON', 'Arm is HOME').",
    )
    post_effects: list[str] = Field(
        default_factory=list,
        description="List of state changes expected after successful execution (e.g., 'Item is now sliced', 'Device is powered off').",
    )


class CapabilityRegistry(BaseModel):
    """
    The central catalog for all available system capabilities (functions/tools).
    It loads, validates, and stores the capability definitions from the configuration file.
    """

    # Store the capabilities as a dictionary mapping function name (str) to its schema (FunctionSchema)
    capabilities: dict[str, FunctionSchema] = Field(
        default_factory=dict,
        description="A dictionary mapping function names (e.g., 'GRILL_ITEM') to their schemas.",
    )

    def get_function_schema(self, function_name: str) -> FunctionSchema:
        """
        Retrieves the FunctionSchema for a named capability.
        Raises KeyError if the function is not registered.
        """
        if function_name not in self.capabilities:
            raise KeyError(f"Capability '{function_name}' is not registered in the system.")
        return self.capabilities[function_name]

    def get_llm_tool_catalog(self) -> str:
        """
        Returns the list of all registered capability schemas, formatted as a JSON string
        suitable for an LLM to use as a tool catalog. This replaces get_primitive_tool_catalog.
        """
        schemas = []
        for name, schema in self.capabilities.items():
            schemas.append(
                {
                    "name": name,
                    "description": schema.description,
                    "parameters": schema.parameters,
                    "risk_level": schema.risk_level,
                    "pre_conditions": schema.pre_conditions,
                    "post_effects": schema.post_effects,
                }
            )
        return json.dumps(schemas, indent=2)

    def validate_call(self, function_name: str, arguments: dict[str, Any]) -> None:
        """
        Validates the arguments of a function call against the registered tool schema.
        Raises an error (e.g., ValueError or TypeError) if validation fails.
        """
        pass

    @classmethod
    def load_from_yaml(cls, config_path: Path | None = None) -> "CapabilityRegistry":
        """
        Loads the capabilities from the provided YAML configuration file.
        If config_path is None, it defaults to 'capability_registry.yaml' in the
        same directory as this module file.
        """
        import yaml

        # Resolve the path if not explicitly provided
        if config_path is None:
            # Assumes the configuration file is named 'capability_registry.yaml'
            # and resides next to this file.
            config_path = Path(__file__).parent / "capability_registry.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path) as f:
            raw_data = yaml.safe_load(f)

        validated_capabilities = {
            name: FunctionSchema(**schema_data) for name, schema_data in raw_data.items()
        }

        return cls(capabilities=validated_capabilities)
