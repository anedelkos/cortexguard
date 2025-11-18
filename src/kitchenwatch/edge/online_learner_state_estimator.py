from collections import deque

from kitchenwatch.core.interfaces.base_online_learner import BaseOnlineLearner
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.state_estimate import StateEstimate


class OnlineLearnerStateEstimator:
    """
    State estimator that wraps an online learner and computes simple confidence
    and uncertainty metrics based on running residuals.
    """

    def __init__(self, learner: BaseOnlineLearner, window_size: int = 50) -> None:
        self._learner = learner
        # Store running residuals per feature
        self._residuals: dict[str, deque[float]] = {}
        self._window_size = window_size

    async def update(self, snapshot: FusionSnapshot) -> StateEstimate:
        now = snapshot.timestamp
        features = snapshot.derived.copy()  # consume only processed/fused features
        residuals: dict[str, float] = {}
        uncertainty: dict[str, float] = {}

        # predict current expected values
        expected = self._learner.predict(features)

        # compute residuals and track running stats
        for key, obs in features.items():
            exp = expected.get(key, obs)
            res = obs - exp
            residuals[key] = res

            if key not in self._residuals:
                self._residuals[key] = deque(maxlen=self._window_size)
            self._residuals[key].append(res)

            # simple uncertainty = stddev of residuals
            vals = self._residuals[key]
            if len(vals) > 1:
                mean = sum(vals) / len(vals)
                variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                uncertainty[key] = variance**0.5
            else:
                uncertainty[key] = 0.0

        # update learner with current observation
        self._learner.update(features)

        # compute simple confidence: inverse of mean residual magnitude
        if residuals:
            avg_residual = sum(abs(r) for r in residuals.values()) / len(residuals)
            confidence = max(0.0, min(1.0, 1.0 - avg_residual / 100.0))  # normalize, tweak scale
        else:
            confidence = 0.5

        # default label (could be extended to rules)
        label = "unknown"

        return StateEstimate(
            timestamp=now,
            label=label,
            confidence=confidence,
            residuals=residuals,
            uncertainty=uncertainty,
            ttd=None,
            ttf=None,
            flags={},
            source_intent=snapshot.intent,
        )
