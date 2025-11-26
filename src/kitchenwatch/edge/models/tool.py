import uuid
from enum import Enum, auto
from typing import Any


class ToolType(Enum):
    """
    Defines the general category or function of a physical tool or capability.
    This helps the planner filter by function.
    """

    SENSOR = auto()  # Gathers data (e.g., temperature, light)
    ACTUATOR = auto()  # Executes a physical change (e.g., switch, valve)
    UTILITY = auto()  # Executes a logical or software operation (e.g., network call)
    LOGGING = auto()  # Used specifically for internal state recording


class ToolStatus(Enum):
    """
    The current operational status of the tool/resource.
    """

    ONLINE = auto()
    OFFLINE = auto()
    ERROR = auto()
    MAINTENANCE = auto()


class Tool:
    """
    A static model representing an available physical device or software capability
    that the agent can interact with. This defines the resources in the external world.

    Uses Python 3.9+ type annotations (list, dict).
    """

    tool_id: str
    name: str
    tool_type: ToolType
    description: str
    capabilities: list[str]
    config: dict[str, Any]
    status: ToolStatus

    def __init__(
        self,
        tool_id: str,
        name: str,
        tool_type: ToolType,
        description: str,
        capabilities: list[str],
        config: dict[str, Any],
        initial_status: ToolStatus = ToolStatus.ONLINE,
    ):
        """
        Initializes a Tool instance.

        Args:
            tool_id: Unique identifier for the tool (e.g., 'freezer_001').
            name: Human-readable name (e.g., 'Walk-in Freezer 1').
            tool_type: The functional category of the tool (ToolType).
            description: Detailed description of the tool's purpose.
            capabilities: A list of abstract commands this tool can execute
                          (e.g., ['SET_TEMP', 'READ_TEMP', 'SHUTDOWN']).
            config: A dictionary of static, low-level connection parameters
                    (e.g., IP address, API key, serial port).
            initial_status: The status when the tool is initialized.
        """
        self.tool_id = tool_id
        self.name = name
        self.tool_type = tool_type
        self.description = description
        self.capabilities = capabilities
        self.config = config
        self.status = initial_status

    def __repr__(self) -> str:
        return (
            f"Tool(id='{self.tool_id}', name='{self.name}', "
            f"type={self.tool_type.name}, status={self.status.name})"
        )

    @classmethod
    def create_new(
        cls,
        name: str,
        tool_type: ToolType,
        description: str,
        capabilities: list[str],
        config: dict[str, Any],
    ) -> "Tool":
        """Helper to create a new tool with a generated UUID."""
        return cls(
            tool_id=str(uuid.uuid4()),
            name=name,
            tool_type=tool_type,
            description=description,
            capabilities=capabilities,
            config=config,
        )
