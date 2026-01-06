import logging

from river import linear_model, preprocessing

from kitchenwatch.core.interfaces.base_online_learner import BaseOnlineLearner

logger = logging.getLogger(__name__)


class RiverOnlineLearner(BaseOnlineLearner):
    """
    River-based online learner adhering to BaseOnlineLearner protocol.

    Uses incremental linear regression with online standardization.
    Each feature is modeled independently as a univariate predictor.

    Design Choice: Univariate Self-Prediction
    - Each feature predicts its own next value (autoregressive approach)
    - Simple baseline for anomaly detection
    - Production: Consider multivariate models or LSTM for better predictions
    """

    def __init__(self) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._models: dict[str, linear_model.LinearRegression] = {}
        self._scalers: dict[str, preprocessing.StandardScaler] = {}

    def update(self, features: dict[str, float]) -> None:
        """
        Incrementally learn from new feature observations.

        Each feature is modeled independently using online linear regression
        with standardization. The model learns to predict each feature's
        next value based on its current value (univariate autoregression).

        Args:
            features: Dictionary mapping feature names to observed values
        """
        for key, value in features.items():
            x = {key: value}

            # Create scaler if missing
            if key not in self._scalers:
                self._scalers[key] = preprocessing.StandardScaler()
                self._logger.debug(f"Created scaler for feature '{key}'")

            # Update scaler first, *then* transform
            self._scalers[key].learn_one(x)  # type: ignore[no-untyped-call]
            x_scaled = self._scalers[key].transform_one(x)  # type: ignore[no-untyped-call]

            # Create model if missing
            if key not in self._models:
                self._models[key] = linear_model.LinearRegression()
                self._logger.debug(f"Created model for feature '{key}'")

            # Update model with scaled value
            self._models[key].learn_one(x_scaled, value)
            self._logger.debug(f"Updated '{key}' with value={value}")

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        """
        Predict expected values for all features.

        Returns predictions for features with trained models.
        For unseen features, returns the observed value as fallback.

        Args:
            features: Dictionary mapping feature names to current values

        Returns:
            Dictionary mapping feature names to predicted values
        """
        predictions: dict[str, float] = {}
        for key, value in features.items():
            if value is None or not isinstance(value, (int, float, bool)):
                predictions[key] = 0.0
                continue

            if isinstance(value, bool):
                value = float(value)

            if key not in self._models or key not in self._scalers:
                # Fallback for untrained features
                predictions[key] = value
                continue

            x = {key: float(value)}
            try:
                x_scaled = self._scalers[key].transform_one(x)  # type: ignore[no-untyped-call]
                y_pred = self._models[key].predict_one(x_scaled)  # type: ignore[no-untyped-call]
            except Exception:  # Any error from scaler/model -> fallback to observed value
                predictions[key] = float(value)
                continue

            predictions[key] = float(y_pred) if y_pred is not None else float(value)

        return predictions

    def anomaly_score(self, features: dict[str, float]) -> float:
        """
        Compute anomaly score based on prediction errors.

        Uses Mean Absolute Error (MAE) across all features as anomaly score.
        Higher scores indicate observed values deviate more from predictions.

        Args:
            features: Dictionary mapping feature names to observed values

        Returns:
            Anomaly score (0.0 = perfect match, higher = more anomalous)
        """
        preds = self.predict(features)
        if not preds:
            return 0.0

        residuals = [abs(features[k] - preds[k]) for k in features if k in preds]

        if not residuals:
            return 0.0

        return sum(residuals) / len(residuals)
