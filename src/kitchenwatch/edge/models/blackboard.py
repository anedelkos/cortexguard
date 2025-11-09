from pydantic import BaseModel

from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.plan import IntentContext, Plan, PlanStep


class Blackboard(BaseModel):
    """Shared memory and context store for the KitchenWatch Edge agent.

    The Blackboard acts as the central hub for inter-subsystem communication
    within the edge runtime. Subsystems read from and write to the Blackboard
    to share the current state, intent, and sensor-derived information.

    Responsibilities:
        - Store the current intent context (plan, step, action, parameters).
        - Keep the latest FusionSnapshot(s) or aggregated sensor summaries.
        - Track plan execution states (running, paused, completed).
        - Maintain anomaly flags and recovery status for other agents.
        - Provide thread-safe or async-safe access to shared data.
    """

    current_intent: IntentContext | None = None
    current_step: PlanStep | None = None
    current_plan: Plan | None = None
    latest_snapshot: FusionSnapshot | None = None
    anomaly_flags: dict[str, bool] = {}
    recovery_status: dict[str, str] = {}
    safety_flags: dict[str, bool] = {}

    def update_intent(self, intent: IntentContext) -> None:
        self.current_intent = intent

    def update_snapshot(self, snapshot: FusionSnapshot) -> None:
        self.latest_snapshot = snapshot

    def get_intent_action(self) -> str | None:
        return self.current_intent.action if self.current_intent else None

    def get_snapshot(self) -> FusionSnapshot | None:
        return self.latest_snapshot
