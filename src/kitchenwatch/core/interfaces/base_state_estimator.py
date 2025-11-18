from typing import Protocol

from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.state_estimate import StateEstimate


class StateEstimator(Protocol):
    """
    Interface for any state estimator used on the edge.

    Implementations may include:
      - rule-based estimators
      - online learners
      - predictive models
      - anomaly-driven estimators
    """

    async def update(self, snapshot: FusionSnapshot) -> StateEstimate:
        """
        Compute a semantic state estimate from a fusion snapshot.
        Must be non-blocking (async).

        Args:
            snapshot: The latest fused and preprocessed sensor reading.

        Returns:
            A StateEstimate containing semantic labels, flags,
            residuals (optional), and confidence.
        """
        ...
