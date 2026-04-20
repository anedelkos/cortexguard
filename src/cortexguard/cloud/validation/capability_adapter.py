"""Adapters for validating cloud-generated actions against the shared capability registry."""

from __future__ import annotations

import logging
from typing import Protocol

from cortexguard.edge.models.capability_registry import CapabilityRegistry

logger = logging.getLogger(__name__)


class CapabilityAdapterProtocol(Protocol):
    def is_known_capability(self, name: str) -> bool: ...
    def get_risk_level(self, name: str) -> str | None: ...
    def validate_arguments(self, name: str, arguments: dict[str, object]) -> list[str]: ...


class CapabilityAdapter:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    @classmethod
    def load_default(cls) -> CapabilityAdapter:
        try:
            return cls(CapabilityRegistry.load_from_yaml())
        except Exception:
            logger.warning("Failed to load capability registry; using empty registry")
            return cls(CapabilityRegistry())

    def is_known_capability(self, name: str) -> bool:
        return name in self._registry.capabilities

    def get_risk_level(self, name: str) -> str | None:
        schema = self._registry.capabilities.get(name)
        if schema is None:
            return None
        return schema.risk_level.value

    def validate_arguments(self, name: str, arguments: dict[str, object]) -> list[str]:
        if name not in self._registry.capabilities:
            return []  # unknown capability handled separately
        valid, _ = self._registry.validate_call(name, arguments)  # type: ignore[arg-type]
        if not valid:
            return [f"{name}: arguments failed schema validation"]
        return []
