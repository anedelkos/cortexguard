from typing import Any, Protocol


class BaseController(Protocol):
    async def execute(self, primitive_name: str, parameters: dict[str, Any]) -> None:
        """
        Execute a high-level primitive.
        """
