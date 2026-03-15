from __future__ import annotations

import asyncio
import copy
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from cortexguard.edge.models.anomaly_event import SEVERITY_RANKING, AnomalyEvent, AnomalySeverity
from cortexguard.edge.models.fusion_snapshot import FusionSnapshot
from cortexguard.edge.models.plan import Plan, PlanStatus, PlanStep
from cortexguard.edge.models.reasoning_trace_entry import ReasoningTraceEntry, TraceSeverity
from cortexguard.edge.models.remediation_policy import RemediationPolicy
from cortexguard.edge.models.scene_graph import SceneGraph
from cortexguard.edge.models.state_estimate import StateEstimate

logger = logging.getLogger(__name__)


@dataclass
class Blackboard:
    """
    Async-safe shared memory store for CortexGuard Edge agent.

    Implements the Blackboard architectural pattern for inter-subsystem
    communication. Acts as the central hub for plan execution state,
    sensor data, and system flags.
    """

    # Core execution state
    current_step: PlanStep | None = None
    current_plan: Plan | None = None
    paused_plan: Plan | None = None

    active_remediation_policy: RemediationPolicy | None = None

    _plan_step_indices: dict[str, int] = field(default_factory=dict)

    # Sensor state
    latest_snapshot: FusionSnapshot | None = None

    # System state & Agent Memory
    # Stores active AnomalyEvent objects, keyed by their semantic key (AnomalyEvent.key).
    active_anomalies: dict[str, AnomalyEvent] = field(default_factory=dict)
    _max_anomaly_severity: AnomalySeverity = AnomalySeverity.LOW

    # The Reasoning Trace (Agent Scratchpad) - Stores a history of all significant events.
    reasoning_traces: deque[ReasoningTraceEntry] = field(default_factory=lambda: deque(maxlen=1000))

    recovery_status: dict[str, str] = field(default_factory=dict)
    safety_flags: dict[str, bool] = field(default_factory=dict)

    # Failed plans for analysis
    failed_plans: deque[Plan] = field(default_factory=lambda: deque(maxlen=100))

    # Custom extensible state
    _custom_state: dict[str, Any] = field(default_factory=dict)

    # Async primitives (created per-instance)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    latest_state_estimate: StateEstimate | None = None

    _scene_graph: SceneGraph | None = None

    # ---------------------
    # Snapshot Methods
    # ---------------------
    async def update_fusion_snapshot(self, snapshot: FusionSnapshot) -> None:
        async with self._lock:
            self.latest_snapshot = snapshot.model_copy(deep=True)

    async def get_fusion_snapshot(self) -> FusionSnapshot | None:
        async with self._lock:
            return self.latest_snapshot.model_copy(deep=True) if self.latest_snapshot else None

    # ---------------------
    # Plan Methods
    # ---------------------
    async def set_current_plan(self, plan: Plan | None) -> None:
        async with self._lock:
            self.current_plan = plan.model_copy(deep=True) if plan else None

    async def set_paused_plan(self, plan: Plan | None) -> None:
        async with self._lock:
            self.paused_plan = plan.model_copy(deep=True) if plan else None

    async def set_failed_plan(self, plan: Plan) -> None:
        """Store failed plan for analysis and potential retry."""
        async with self._lock:
            self.failed_plans.append(plan.model_copy(deep=True))

    async def get_current_plan(self) -> Plan | None:
        async with self._lock:
            if self.current_plan is None:
                return None
            try:
                return self.current_plan.model_copy(deep=True)
            except AttributeError:
                logger.error("current_plan missing model_copy() method")
                return copy.deepcopy(self.current_plan)

    async def get_paused_plan(self) -> Plan | None:
        async with self._lock:
            return self.paused_plan.model_copy(deep=True) if self.paused_plan else None

    async def set_current_step(self, step: PlanStep | None) -> None:
        async with self._lock:
            self.current_step = step.model_copy(deep=True) if step else None

    async def get_current_step(self) -> PlanStep | None:
        async with self._lock:
            return self.current_step.model_copy(deep=True) if self.current_step else None

    async def clear_current_plan(self) -> None:
        """
        Clear active plan and step.

        For COMPLETED plans, also clear the step index.
        For PREEMPTED/FAILED plans, preserve the index so they can resume.
        """
        async with self._lock:
            if self.current_plan:
                # Only clear index if plan completed successfully
                if self.current_plan.status == PlanStatus.COMPLETED:
                    self._plan_step_indices.pop(self.current_plan.plan_id, None)
                # For PREEMPTED/FAILED, preserve index for resumption

            self.current_plan = None
            self.current_step = None

    async def set_step_index_for_plan(self, plan_id: str, index: int) -> None:
        """Saves the current step index for a specific plan ID."""
        async with self._lock:
            self._plan_step_indices[plan_id] = index

    async def get_step_index_for_plan(self, plan_id: str) -> int | None:
        """Retrieves the last saved step index for a plan, or None if new."""
        async with self._lock:
            return self._plan_step_indices.get(plan_id)

    async def clear_step_index_for_plan(self, plan_id: str) -> None:
        """Removes the stored step index when a plan execution is finalized."""
        async with self._lock:
            if plan_id in self._plan_step_indices:
                del self._plan_step_indices[plan_id]

    # ---------------------
    # Anomaly & Recovery Flags
    # ---------------------
    async def add_trace_entry(self, entry: ReasoningTraceEntry) -> None:
        """
        Adds a generic event entry to the Reasoning Trace (Agent Scratchpad).
        """
        async with self._lock:
            self.reasoning_traces.append(entry.model_copy(deep=True))

    async def set_anomaly(self, event: AnomalyEvent) -> None:
        """
        Adds a new AnomalyEvent to the Reasoning Trace and registers it
        as an active anomaly.
        """
        trace_sev = (
            TraceSeverity.CRITICAL
            if event.severity == AnomalySeverity.HIGH
            else (
                TraceSeverity.HIGH
                if event.severity == AnomalySeverity.MEDIUM
                else TraceSeverity.WARN
            )
        )

        trace_entry = ReasoningTraceEntry(
            timestamp=event.timestamp,
            source=event.key,
            event_type=f"ANOMALY_{event.severity.name}",
            reasoning_text=(
                f"Anomaly {event.key} detected. Severity: {event.severity.name}, "
                f"Score: {event.score:.2f}. Detectors: {', '.join(event.contributing_detectors)}."
            ),
            metadata={
                "anomaly_id": event.id,
                "severity_score": event.score,
                "contributing_detectors": event.contributing_detectors,
                "original_metadata": event.metadata,
            },
            severity=trace_sev,
        )

        async with self._lock:
            self.reasoning_traces.append(trace_entry.model_copy(deep=True))
            self.active_anomalies[event.key] = event.model_copy(deep=True)
            if SEVERITY_RANKING[event.severity] > SEVERITY_RANKING[self._max_anomaly_severity]:
                self._max_anomaly_severity = event.severity

    async def clear_anomaly(self, key: str) -> None:
        """
        Clears/removes an active anomaly flag based on its semantic key.
        """
        async with self._lock:
            if key in self.active_anomalies:
                removed = self.active_anomalies.pop(key)

                # Recompute max severity if we removed the current max
                if removed.severity == self._max_anomaly_severity:
                    if self.active_anomalies:
                        # Use SEVERITY_RANKING for comparison (Enums aren't directly comparable)
                        self._max_anomaly_severity = max(
                            self.active_anomalies.values(),
                            key=lambda e: SEVERITY_RANKING[e.severity],
                        ).severity
                    else:
                        self._max_anomaly_severity = AnomalySeverity.LOW

    async def get_active_anomalies(self) -> dict[str, AnomalyEvent]:
        """Retrieves all currently active anomaly events."""
        async with self._lock:
            # Return a copy to prevent external modification outside the lock
            return {k: v.model_copy(deep=True) for k, v in self.active_anomalies.items()}

    async def get_active_anomaly(self, key: str) -> AnomalyEvent | None:
        """
        Retrieve a specific active AnomalyEvent object.
        """
        async with self._lock:
            event = self.active_anomalies.get(key)
            return event.model_copy(deep=True) if event else None

    async def get_highest_anomaly_severity(self) -> AnomalySeverity | None:
        async with self._lock:
            return None if not self.active_anomalies else self._max_anomaly_severity

    async def is_anomaly_present(
        self, severity_min: AnomalySeverity = AnomalySeverity.MEDIUM
    ) -> bool:
        async with self._lock:
            if not self.active_anomalies:
                return False

            return SEVERITY_RANKING[self._max_anomaly_severity] >= SEVERITY_RANKING[severity_min]

    async def set_recovery_status(self, key: str, status: str) -> None:
        async with self._lock:
            self.recovery_status[key] = status

    async def get_recovery_status(self, key: str) -> str | None:
        async with self._lock:
            return self.recovery_status.get(key)

    # ---------------------
    # Remediation Policy
    # ---------------------
    async def set_remediation_policy(self, policy: RemediationPolicy) -> None:
        """Sets an active remediation policy."""
        async with self._lock:
            self.active_remediation_policy = policy.model_copy(deep=True)

    async def get_remediation_policy(self) -> RemediationPolicy | None:
        """Retrieves the active remediation policy."""
        async with self._lock:
            return (
                self.active_remediation_policy.model_copy(deep=True)
                if self.active_remediation_policy
                else None
            )

    async def clear_remediation_policy(self) -> None:
        """Removes the active remediation policy."""
        async with self._lock:
            self.active_remediation_policy = None

    # ---------------------
    # Safety Flags
    # ---------------------
    async def set_safety_flag(self, key: str, value: bool) -> None:
        async with self._lock:
            self.safety_flags[key] = value

    async def get_safety_flag(self, key: str) -> bool | None:
        async with self._lock:
            return self.safety_flags.get(key)

    # ---------------------
    # State estimates
    # ---------------------
    async def update_state_estimate(self, estimate: StateEstimate) -> None:
        async with self._lock:
            self.latest_state_estimate = estimate.model_copy(deep=True)

    async def get_latest_state_estimate(self) -> StateEstimate | None:
        async with self._lock:
            return (
                self.latest_state_estimate.model_copy(deep=True)
                if self.latest_state_estimate
                else None
            )

    # ---------------------
    # Scene graph
    # ---------------------
    async def set_scene_graph(self, graph: SceneGraph) -> None:
        """Store the latest scene graph for consumers."""
        async with self._lock:
            self._scene_graph = graph.model_copy(deep=True)

    async def get_scene_graph(self) -> SceneGraph | None:
        async with self._lock:
            return self._scene_graph.model_copy(deep=True) if self._scene_graph else None

    # ---------------------
    # Extensibility
    # ---------------------
    async def set(self, key: str, value: Any) -> None:
        """
        Store custom state for experimental features.
        Uses separate namespace to avoid corrupting core state.
        """
        async with self._lock:
            self._custom_state[key] = copy.deepcopy(value)

    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve custom state."""
        async with self._lock:
            custom_state = self._custom_state.get(key, default)
            return copy.deepcopy(custom_state)

    async def get_metrics(self) -> dict[str, int | bool]:
        """Get Blackboard metrics for monitoring."""
        async with self._lock:
            return {
                "active_anomalies_count": len(self.active_anomalies),
                "reasoning_traces_count": len(self.reasoning_traces),
                "failed_plans_count": len(self.failed_plans),
                "custom_state_keys": len(self._custom_state),
                "has_current_plan": self.current_plan is not None,
            }
