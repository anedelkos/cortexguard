import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kitchenwatch.core.interfaces.base_online_learner import BaseOnlineLearner
from kitchenwatch.edge.models.state_estimate import StateEstimate
from kitchenwatch.edge.online_learner_state_estimator import OnlineLearnerStateEstimator


class DummyLearner(BaseOnlineLearner):
    def __init__(
        self,
        predict_map: (
            Callable[[dict[str, float]], dict[str, float]] | dict[str, float] | None
        ) = None,
    ) -> None:
        # predict_map can be a callable(features)->dict or a static dict
        self.predict_map = predict_map
        self.update_calls: int = 0
        self.last_updated: dict[str, float] | None = None

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        if callable(self.predict_map):
            return self.predict_map(features)  # type: ignore[no-any-return]
        if isinstance(self.predict_map, dict):
            # return mapping for keys present, leave others unchanged
            return {k: self.predict_map.get(k, features[k]) for k in features}
        # default: echo features (no residuals)
        return {k: features[k] for k in features}

    def update(self, features: dict[str, float]) -> None:
        self.update_calls += 1
        self.last_updated = dict(features)

    def anomaly_score(self, features: dict[str, float]) -> float:
        return 0.0


def run_update(estimator: OnlineLearnerStateEstimator, snapshot: Any) -> StateEstimate:
    """Helper to run the async update synchronously."""
    return asyncio.run(estimator.update(snapshot))


def approx_stddev_sample(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean: float = sum(values) / n
    variance: float = sum((v - mean) ** 2 for v in values) / (n - 1)
    return float(variance**0.5)


def test_update_computes_residuals_uncertainty_and_confidence() -> None:
    # Prepare dummy learner that yields a residual for 'a' but not for 'b'
    def pred_map(features: dict[str, float]) -> dict[str, float]:
        return {"a": features["a"] - 2.0, "b": features["b"]}  # expected slightly lower for 'a'

    learner = DummyLearner(predict_map=pred_map)

    blackboard_mock = AsyncMock()
    blackboard_mock.get_current_step.return_value = SimpleNamespace(description="mock_intent")

    estimator = OnlineLearnerStateEstimator(learner, blackboard=blackboard_mock)

    snapshot = SimpleNamespace(timestamp=12345, derived={"a": 10.0, "b": 20.0})

    state: StateEstimate = run_update(estimator, snapshot)

    # residuals: a -> obs - exp = 10 - (10-2) = 2.0 ; b -> 20 - 20 = 0.0
    assert "a" in state.residuals and "b" in state.residuals
    assert state.residuals["a"] == pytest.approx(2.0)
    assert state.residuals["b"] == pytest.approx(0.0)

    # first observation -> uncertainty should be 0.0 for both features
    assert state.uncertainty is not None
    assert state.uncertainty["a"] == pytest.approx(0.0)
    assert state.uncertainty["b"] == pytest.approx(0.0)

    assert state.confidence == pytest.approx(1.0)

    assert state.label == "nominal"
    # Assertion now correctly relies on the blackboard mock's description value
    assert state.source_intent == "mock_intent"

    # ensure learner.update was invoked once and received the features dict
    assert learner.update_calls == 1
    assert learner.last_updated == {"a": 10.0, "b": 20.0}


def test_running_uncertainty_and_window_size() -> None:
    # Predictor returns zero expected value so residual == observed value
    learner = DummyLearner(predict_map=lambda features: {k: 0.0 for k in features})

    blackboard_mock = AsyncMock()
    blackboard_mock.get_current_step.return_value = SimpleNamespace(description="mock_intent")

    estimator = OnlineLearnerStateEstimator(learner, blackboard=blackboard_mock, window_size=3)

    # Override _min_history for testing the initial uncertainty ---
    # The default 10 samples is too high for this test. Set it low (e.g., 1)
    # to enable uncertainty calculation immediately after the second sample.
    estimator._min_history = 1

    # Three updates: residuals for 'x' are 1.0, 2.0, 4.0
    vals: list[float] = [1.0, 2.0, 4.0]
    snapshots: list[SimpleNamespace] = [
        # Removed intent=None
        SimpleNamespace(timestamp=i, derived={"x": v})
        for i, v in enumerate(vals, start=1)
    ]

    # Apply updates sequentially and capture state after each
    states: list[StateEstimate] = [run_update(estimator, s) for s in snapshots]

    # After first update uncertainty == 0.0
    assert states[0].uncertainty is not None
    assert states[0].uncertainty["x"] == pytest.approx(0.0)

    # After second update, uncertainty should be sample stddev of [1.0, 2.0]
    expected_std2 = approx_stddev_sample([1.0, 2.0])
    assert states[1].uncertainty is not None
    # Now this passes because _min_history = 1 allows calculation at N=2
    assert states[1].uncertainty["x"] == pytest.approx(expected_std2)

    # After third update, uncertainty should be sample stddev of [1.0,2.0,4.0]
    expected_std3 = approx_stddev_sample([1.0, 2.0, 4.0])
    assert states[2].uncertainty is not None
    assert states[2].uncertainty["x"] == pytest.approx(expected_std3)

    # Now push a fourth value; oldest (1.0) should be dropped due to window_size=3
    fourth_snapshot = SimpleNamespace(timestamp=99, derived={"x": 8.0})  # Removed intent=None
    state4: StateEstimate = run_update(estimator, fourth_snapshot)

    # The deque for 'x' should now contain last three residuals: [2.0, 4.0, 8.0]
    expected_std_after_drop = approx_stddev_sample([2.0, 4.0, 8.0])
    assert state4.uncertainty is not None
    assert state4.uncertainty["x"] == pytest.approx(expected_std_after_drop)

    # ensure learner.update called total of 4 times
    assert learner.update_calls == 4
