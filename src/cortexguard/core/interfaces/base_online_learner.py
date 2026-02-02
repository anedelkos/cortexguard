from typing import Protocol


class BaseOnlineLearner(Protocol):
    """
    A minimal contract for any online learner that maintains
    a notion of expected sensor values and updates incrementally.
    """

    def update(self, features: dict[str, float]) -> None:
        """
        Incrementally update the model with a new observation.
        Non-blocking, no return value.
        """
        ...

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        """
        Given the current fused features, return the expected values
        for the same keys. If the learner doesn't know, it may return
        an empty dict or partial predictions.
        """
        ...

    def anomaly_score(self, features: dict[str, float]) -> float:
        """
        Return a scalar score: how unlikely the current features are
        under the learned model. Purely optional for state estimator.
        """
        ...
