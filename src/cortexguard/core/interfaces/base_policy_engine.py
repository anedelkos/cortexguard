from __future__ import annotations

from typing import Protocol

from cortexguard.edge.models.anomaly_event import AnomalyEvent
from cortexguard.edge.models.remediation_policy import RemediationPolicy
from cortexguard.edge.models.state_estimate import StateEstimate


class BasePolicyEngine(Protocol):
    """
    The Protocol interface for any component responsible for translating an
    AnomalyEvent (your existing model) into an actionable RemediationPolicy (your existing model).

    Any class that implements the methods defined below automatically satisfies this Protocol.
    """

    async def generate_policy(
        self,
        event: AnomalyEvent,
        context: StateEstimate,
        action_catalog_json: str,
        active_plan_context: str,
        vision_context: str,
    ) -> RemediationPolicy:
        """
        Takes a system anomaly event and context and generates a deterministic, prioritized
        remediation policy containing a sequence of corrective steps, reasoning, and risk assessment.

        This method encapsulates the 'reasoning' layer.
        """
        ...

    def model_name(self) -> str:
        """Returns the name of the underlying model or mechanism."""
        ...
