from __future__ import annotations

from typing import Protocol

from cortexguard.edge.models.plan import PlanStep, StepStatus


class BaseStepClassifier(Protocol):
    """
    Defines the interface for a step classifier that determines
    whether a step execution succeeded, failed, or is still running.
    """

    def classify_completion_status(self, step: PlanStep) -> StepStatus:
        """
        Classify the current outcome of a step based on sensor data,
        logs, or environment signals.

        Should return:
          - StepStatus.COMPLETED if the step succeeded,
          - StepStatus.FAILED if it failed,
          - StepStatus.RUNNING if it's still ongoing.
        """
        ...
