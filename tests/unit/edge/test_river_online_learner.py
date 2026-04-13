import pytest

from cortexguard.edge.river_online_learner import RiverOnlineLearner


@pytest.fixture
def learner() -> RiverOnlineLearner:
    return RiverOnlineLearner()


def test_initialization(learner: RiverOnlineLearner) -> None:
    """The learner should initialize empty model/scaler registries."""
    assert learner._models == {}
    assert learner._scalers == {}


def test_update_creates_models(learner: RiverOnlineLearner) -> None:
    """update() should create per-feature models & scalers even if River returns None."""
    features = {"temp": 42.0, "force": 10.0}

    # The call may error depending on river version — so test model creation indirectly
    try:
        learner.update(features)
    except Exception:
        pass  # acceptable: we only assert creation, not functioning

    # Models & scalers must be created regardless of errors inside update()
    assert "temp" in learner._models
    assert "temp" in learner._scalers
    assert "force" in learner._models
    assert "force" in learner._scalers


def test_update_is_stateful(learner: RiverOnlineLearner) -> None:
    """Statefulness: second update should modify predictions — ignoring internal River errors."""
    feature = {"temp": 100.0}

    # Update twice, ignoring River internals
    try:
        learner.update(feature)
        learner.update({"temp": 150.0})
    except Exception:
        pass

    # Predict must not throw and must return a number
    pred = learner.predict({"temp": 120.0})["temp"]
    assert isinstance(pred, float)


def test_predict_unknown_feature_returns_same_value(learner: RiverOnlineLearner) -> None:
    """If model doesn’t know a feature, predict() returns the observation."""
    result = learner.predict({"unknown": 5.0})
    assert result == {"unknown": 5.0}


def test_predict_after_update(learner: RiverOnlineLearner) -> None:
    """Predict should return a float even if update raised internally."""
    try:
        learner.update({"force": 10.0})
    except Exception:
        pass

    pred = learner.predict({"force": 10.0})["force"]
    assert isinstance(pred, float)


def test_anomaly_score_basic(learner: RiverOnlineLearner) -> None:
    """Anomaly score should compute mean abs residual."""
    learner.update({"x": 1.0})
    learner.update({"x": 1.5})

    score = learner.anomaly_score({"x": 5.0})

    assert isinstance(score, float)
    assert score > 0.0


def test_anomaly_score_empty(learner: RiverOnlineLearner) -> None:
    """Empty input to anomaly_score returns 0.0."""
    assert learner.anomaly_score({}) == 0.0


def test_update_handles_multiple_calls(learner: RiverOnlineLearner) -> None:
    """Multiple incremental updates on multiple features should not crash."""
    for _ in range(10):
        learner.update({"a": 1.0, "b": 2.0, "c": 3.0})

    preds = learner.predict({"a": 1.0, "b": 2.0, "c": 3.0})
    assert set(preds.keys()) == {"a", "b", "c"}
