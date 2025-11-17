import logging

from river import linear_model, preprocessing

from kitchenwatch.core.interfaces.base_online_learner import BaseOnlineLearner


class RiverOnlineLearner(BaseOnlineLearner):
    """River-based online learner adhering to BaseOnlineLearner protocol."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._models: dict[str, linear_model.LinearRegression] = {}
        self._scalers: dict[str, preprocessing.StandardScaler] = {}
        self._logger.info("RiverOnlineLearner initialized")

    def update(self, features: dict[str, float]) -> None:
        """Incrementally learn each feature as a univariate model predicting itself."""
        for key, value in features.items():
            x = {key: value}

            # Create scaler if missing
            if key not in self._scalers:
                self._scalers[key] = preprocessing.StandardScaler()
                self._logger.debug(f"Created scaler for feature '{key}'")

            # Update scaler first, *then* transform
            self._scalers[key].learn_one(x)
            x_scaled = self._scalers[key].transform_one(x)

            # Create model if missing
            if key not in self._models:
                self._models[key] = linear_model.LinearRegression()
                self._logger.debug(f"Created model for feature '{key}'")

            # Update model with scaled value
            self._models[key].learn_one(x_scaled, value)
            self._logger.debug(f"Updated '{key}' with value={value}")

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        """Return expected values for all features provided."""
        predictions: dict[str, float] = {}
        for key, value in features.items():
            if key not in self._models or key not in self._scalers:
                # tests expect missing model → fallback
                predictions[key] = value
                continue

            x = {key: value}
            x_scaled = self._scalers[key].transform_one(x)
            y_pred = self._models[key].predict_one(x_scaled)

            predictions[key] = y_pred if y_pred is not None else value

        return predictions

    def anomaly_score(self, features: dict[str, float]) -> float:
        """Compute simple anomaly score: mean absolute residual across features."""
        preds = self.predict(features)
        if not preds:
            return 0.0

        total = 0.0
        n = 0
        for k, obs in features.items():
            if k in preds:
                total += abs(obs - preds[k])
                n += 1

        return total / n if n else 0.0
