from __future__ import annotations

import logging
import random

from cortexguard.core.interfaces.base_step_classifier import BaseStepClassifier
from cortexguard.edge.models.plan import PlanStep, StepStatus

logger = logging.getLogger(__name__)


class MockStepClassifier(BaseStepClassifier):
    """
    Mock implementation of a step classifier that returns random outcomes.
    Useful for testing StepExecutor behavior.
    """

    def __init__(
        self,
        fail_rate: float = 0.2,
        running_rate: float = 0.1,
        custom_logger: logging.Logger = logger,
    ):
        """
        Args:
            fail_rate: probability that a step fails
            running_rate: probability that a step is still running
        """
        if fail_rate + running_rate > 1.0:
            raise ValueError("fail_rate + running_rate must be <= 1.0")
        self._fail_rate = fail_rate
        self._running_rate = running_rate
        self._logger = custom_logger

    def classify_completion_status(self, step: PlanStep) -> StepStatus:
        r = random.random()  # nosec B311: non-cryptographic randomness acceptable in mock
        if r < self._fail_rate:
            self._logger.debug(f"[MOCK] Step {step.id}.{step.description} -> FAILED")
            return StepStatus.FAILED
        elif r < self._fail_rate + self._running_rate:
            self._logger.debug(f"[MOCK] Step {step.id}.{step.description} -> RUNNING")
            return StepStatus.RUNNING
        else:
            self._logger.debug(f"[MOCK] Step {step.id}.{step.description} -> COMPLETED")
            return StepStatus.COMPLETED
