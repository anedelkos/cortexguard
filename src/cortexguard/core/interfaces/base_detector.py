from typing import Any, Protocol

from cortexguard.edge.models.fusion_snapshot import FusionSnapshot


class BaseDetector(Protocol):
    """
    Protocol for anomaly detectors.

    Each detector analyzes a FusionSnapshot and returns detection results with:
    - anomaly_score: float (0.0–1.0, where 1.0 is most anomalous)
    - severity: str ('low', 'medium', 'high')
    - key: str (identifier for this anomaly type)
    - metadata: optional dict with additional context

    Example Implementation:
        class StatisticalDetector:
            async def detect(self, snapshot: FusionSnapshot) -> dict[str, Any]:
                # Analyze snapshot
                return {
                    "key": "statistical_outlier",
                    "anomaly_score": 0.85,
                    "severity": "high",
                    "metadata": {"threshold": 3.0, "z_score": 4.2}
                }
    """

    async def detect(
        self,
        snapshot: FusionSnapshot,
    ) -> dict[str, Any]:
        """
        Detect anomalies in the given snapshot.

        Args:
            snapshot: Fused sensor snapshot to analyze

        Returns:
            Dictionary containing detection results with required keys:
            - key: str (anomaly identifier)
            - anomaly_score: float (0.0-1.0)
            - severity: str ('low', 'medium', 'high')
            - metadata: dict (optional additional information)
        """
        ...
