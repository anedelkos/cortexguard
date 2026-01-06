import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from kitchenwatch.edge.models.anomaly_event import AnomalyEvent
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.scene_graph import SceneGraph
from kitchenwatch.edge.models.state_estimate import StateEstimate

logger = logging.getLogger(__name__)

SAFETY_RADIUS_M = 0.5


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

    # canonical keys that always mean immediate stop (upper-case for normalization)
    EXPLICIT_STOP_KEYS = {"HUMAN_PROXIMITY_VIOLATION", "OVERHEAT_SMOKE_COMBO", "OVERHEAT_SMOKE"}

    def __init__(self, blackboard: Blackboard, rules: list[SafetyRule] | None = None) -> None:
        self.rules: list[SafetyRule] = rules if rules is not None else []
        self._load_default_rules()
        self._blackboard: Blackboard = blackboard
        self._anomalies: dict[str, AnomalyEvent] = {}
        logger.info("SafetyAgent initialized with %d rules", len(self.rules))

    def _load_default_rules(self) -> None:
        """Load critical, hard-coded safety rules."""
        self.rules.append(self._rule_critical_system_state)
        self.rules.append(self._rule_human_hand_near_hazard)
        self.rules.append(self._rule_immediate_human_proximity)
        self.rules.append(self._rule_detector_short_circuit)

    async def execute_safety_check(self, state_estimate: StateEstimate) -> SafetyCommand:
        # pull scene graph from blackboard
        scene_graph = await self._blackboard.get_scene_graph()
        self._anomalies = await self._blackboard.get_active_anomalies()

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

    def _rule_immediate_human_proximity(
        self, state: StateEstimate, scene: SceneGraph | None
    ) -> SafetyCommand:
        """Immediate E-STOP if an observed nearest human distance is within the safety radius."""
        # Prefer residuals/derived observed value if available
        # residuals may contain the observed feature or you may read snapshot.derived earlier
        nearest = None

        # 1) check numeric residual/derived in state if present
        if "vision_nearest_human_m" in state.residuals:
            # residuals may be observed - expected; if you store observed elsewhere, use that instead
            nearest = state.residuals.get("vision_nearest_human_m")
        # 2) fallback to z_scores or symbolic if you store observed in flags or symbolic names
        # 3) fallback to scene graph if provided
        if nearest is None and scene is not None:
            for o in scene.objects:
                if (o.label or "").lower() in ("person", "human"):
                    d = o.properties.get("distance_m")
                    if isinstance(d, (int, float)):
                        nearest = d if nearest is None else min(nearest, float(d))

        if isinstance(nearest, (int, float)) and nearest <= SAFETY_RADIUS_M:
            return SafetyCommand(
                action="E-STOP",
                reason=f"Immediate human proximity: {nearest:.2f}m <= {SAFETY_RADIUS_M:.2f}m",
            )

        return SafetyCommand(action="NOMINAL")

    def _rule_detector_short_circuit(
        self, state: StateEstimate, scene: SceneGraph | None
    ) -> SafetyCommand:
        """
        Short-circuit rule: immediate E-STOP for explicit anomaly keys or string severities.
        Assumes detectors publish a canonical `key` on each anomaly. Does not access .name.
        """
        if not self._anomalies:
            return SafetyCommand(action="NOMINAL")

        for aid, raw in self._anomalies.items():
            # normalize to a simple object with .key, .severity, .score, .metadata
            if isinstance(raw, dict):
                key = (raw.get("key") or raw.get("anomaly_key") or str(aid)).upper()
                severity = raw.get("severity")
                score = raw.get("score") or raw.get("anomaly_score")
            else:
                # assume AnomalyEvent-like object with attributes
                key = (getattr(raw, "key", None) or str(aid)).upper()
                severity = getattr(raw, "severity", None)
                score = getattr(raw, "score", None) or getattr(raw, "anomaly_score", None)

            # 1) explicit key check (fast, auditable)
            if key in self.EXPLICIT_STOP_KEYS:
                reason = f"Detector explicit key {key} (score={score})"
                return SafetyCommand(action="E-STOP", reason=reason)

            # 2) severity check if severity is a string (e.g., "HIGH", "CRITICAL")
            if isinstance(severity, str):
                if severity.upper() in ("HIGH", "CRITICAL"):
                    reason = (
                        f"Detector high-severity anomaly {key} (severity={severity}, score={score})"
                    )
                    return SafetyCommand(action="E-STOP", reason=reason)

        return SafetyCommand(action="NOMINAL")
