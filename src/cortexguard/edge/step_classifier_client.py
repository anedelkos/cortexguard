"""HTTP client that calls the cloud classify API for model-based step classification.

Replaces MockStepClassifier when the edge is connected to the cloud.
"""

from __future__ import annotations

import logging

import httpx

from cortexguard.core.interfaces.base_step_classifier import BaseStepClassifier
from cortexguard.edge.models.plan import PlanStep, StepStatus

logger = logging.getLogger(__name__)


class StepClassifierClient(BaseStepClassifier):
    def __init__(
        self,
        cloud_api_url: str,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        fallback_status: StepStatus = StepStatus.COMPLETED,
    ) -> None:
        self._base_url = cloud_api_url.rstrip("/")
        self._api_key = api_key
        self._http_client = http_client
        self._fallback_status = fallback_status

    def _auth_headers(self) -> dict[str, str]:
        if self._api_key is not None:
            return {"X-CortexGuard-Key": self._api_key}
        return {}

    def classify_completion_status(self, step: PlanStep) -> StepStatus:
        """Synchronously classify step outcome via the cloud SageMaker proxy.

        Falls back to the default status on network or API errors.
        """
        payload: dict[str, object] = {
            "device_id": getattr(step, "device_id", ""),
            "step_key": step.id,
            "sensor_snapshot": {},
        }

        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self._base_url}/api/v1/classify",
                    json=payload,
                    headers=self._auth_headers(),
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    outcome = data.get("predicted_outcome", "completed")
                    logger.debug(
                        "Classifier returned %s (confidence=%.3f, model=%s v%s)",
                        outcome,
                        data.get("confidence", 0.0),
                        data.get("model_id", "?"),
                        data.get("model_version", "?"),
                    )
                    return StepStatus.COMPLETED if outcome == "completed" else StepStatus.FAILED

                logger.warning("Classifier returned %d; falling back", resp.status_code)
        except Exception:
            logger.debug("Classifier call failed; using fallback", exc_info=True)

        return self._fallback_status
