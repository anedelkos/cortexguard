import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.scene_graph import SceneGraph
from kitchenwatch.edge.models.state_estimate import StateEstimate

logger = logging.getLogger(__name__)


# Structured command instead of raw strings
@dataclass
class SafetyCommand:
    action: Literal["E-STOP", "PAUSE", "NOMINAL"]
    reason: str | None = None


type SafetyRule = Callable[[StateEstimate, SceneGraph | None], SafetyCommand]


class SafetyAgent:
    """
    The Safety Agent monitors system state and scene graph to detect hazards.
    It evaluates a set of safety rules and emits structured SafetyCommand objects.
    """

    def __init__(self, blackboard: Blackboard, rules: list[SafetyRule] | None = None) -> None:
        self.rules: list[SafetyRule] = rules if rules is not None else []
        self._load_default_rules()
        self._blackboard: Blackboard = blackboard
        logger.info("SafetyAgent initialized with %d rules", len(self.rules))

    def _load_default_rules(self) -> None:
        """Load critical, hard-coded safety rules."""
        self.rules.append(self._rule_critical_system_state)
        self.rules.append(self._rule_human_hand_near_hazard)

    async def execute_safety_check(self, state_estimate: StateEstimate) -> SafetyCommand:
        # pull scene graph from blackboard
        scene_graph = await self._blackboard.get_scene_graph()

        for rule in self.rules:
            cmd = rule(state_estimate, scene_graph)
            if cmd.action != "NOMINAL":
                logger.warning("Safety breach detected by %s: %s", rule.__name__, cmd.reason)
                return cmd

        return SafetyCommand(action="NOMINAL")

    # --- Default rules ---

    def _rule_critical_system_state(
        self, state: StateEstimate, scene: SceneGraph | None
    ) -> SafetyCommand:
        """Rule 1: Check for critical low-level system failures (no SceneGraph needed)."""
        for key, value in state.symbolic_system_state.items():
            if "CRITICAL" in str(value).upper():
                return SafetyCommand(
                    action="E-STOP", reason=f"Critical system component failure ({key} is {value})"
                )
        return SafetyCommand(action="NOMINAL")

    def _rule_human_hand_near_hazard(
        self, state: StateEstimate, scene: SceneGraph | None
    ) -> SafetyCommand:
        """Rule 2: Check for human-robot hazards using SceneGraph proximity."""
        if not scene:
            return SafetyCommand(action="NOMINAL")

        hazard_ids = {
            obj.id
            for obj in scene.objects
            if obj.label in ["Blade", "Grill"] or obj.properties.get("safety_critical", False)
        }
        hand_ids = {obj.id for obj in scene.objects if obj.label == "Human_Hand"}

        for rel in scene.relationships:
            if rel.relationship in ("near", "touching"):
                if rel.source_id in hand_ids and rel.target_id in hazard_ids:
                    return SafetyCommand(
                        action="E-STOP",
                        reason=f"Human Hand ({rel.source_id}) {rel.relationship} hazard {rel.target_id}",
                    )
                if rel.target_id in hand_ids and rel.source_id in hazard_ids:
                    return SafetyCommand(
                        action="E-STOP",
                        reason=f"Human Hand ({rel.target_id}) {rel.relationship} hazard {rel.source_id}",
                    )

        return SafetyCommand(action="NOMINAL")
