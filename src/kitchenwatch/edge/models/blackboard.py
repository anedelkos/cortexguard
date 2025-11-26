import asyncio
from dataclasses import dataclass, field
from typing import Any

# Assuming these imports are available in your environment
from kitchenwatch.edge.models.anomaly_event import AnomalyEvent
from kitchenwatch.edge.models.anomaly_severity import SEVERITY_RANKING, AnomalySeverity
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.plan import Plan, PlanStatus, PlanStep
from kitchenwatch.edge.models.state_estimate import StateEstimate


@dataclass
class Blackboard:
    """
    Async-safe shared memory store for KitchenWatch Edge agent.

    Implements the Blackboard architectural pattern for inter-subsystem
    communication. Acts as the central hub for plan execution state,
    sensor data, and system flags.
    """

    # Core execution state
    current_step: PlanStep | None = None
    current_plan: Plan | None = None
    paused_plan: Plan | None = None

    _plan_step_indices: dict[str, int] = field(default_factory=dict)

    # Sensor state
    latest_snapshot: FusionSnapshot | None = None

    # System state & Agent Memory
    # Stores active AnomalyEvent objects, keyed by their semantic key (AnomalyEvent.key).
    active_anomalies: dict[str, AnomalyEvent] = field(default_factory=dict)

    # The Reasoning Trace (Agent Scratchpad) - Stores a history of all significant events.
    reasoning_trace: list[AnomalyEvent] = field(default_factory=list)

    recovery_status: dict[str, str] = field(default_factory=dict)
    safety_flags: dict[str, bool] = field(default_factory=dict)

    # Failed plans for analysis
    failed_plans: list[Plan] = field(default_factory=list)

    # Custom extensible state
    _custom_state: dict[str, Any] = field(default_factory=dict)

    # Async primitives (created per-instance)
    _lock: asyncio.Lock = asyncio.Lock()
    snapshot_updated: asyncio.Event = asyncio.Event()
    latest_state_estimate: StateEstimate | None = None
    state_estimate_updated: asyncio.Event = asyncio.Event()
    anomaly_updated: asyncio.Event = asyncio.Event()

    # ---------------------
    # Snapshot Methods
    # ---------------------
    async def update_fusion_snapshot(self, snapshot: FusionSnapshot) -> None:
        async with self._lock:
            self.latest_snapshot = snapshot
        self.snapshot_updated.set()

    async def get_fusion_snapshot(self) -> FusionSnapshot | None:
        async with self._lock:
            return self.latest_snapshot

    # ---------------------
    # Plan Methods
    # ---------------------
    async def set_current_plan(self, plan: Plan | None) -> None:
        async with self._lock:
            self.current_plan = plan

    async def set_paused_plan(self, plan: Plan | None) -> None:
        async with self._lock:
            self.paused_plan = plan

    async def set_failed_plan(self, plan: Plan) -> None:
        """Store failed plan for analysis and potential retry."""
        async with self._lock:
            self.failed_plans.append(plan)

    async def get_current_plan(self) -> Plan | None:
        async with self._lock:
            return self.current_plan

    async def get_paused_plan(self) -> Plan | None:
        async with self._lock:
            return self.paused_plan

    async def set_current_step(self, step: PlanStep | None) -> None:
        async with self._lock:
            self.current_step = step

    async def get_current_step(self) -> PlanStep | None:
        async with self._lock:
            return self.current_step

    async def clear_current_plan(self) -> None:
        """
        Clear active plan and step.
        Typically called by Orchestrator during plan transitions.
        """
        async with self._lock:
            # If the plan is running, assume it completed and clear its index
            if (
                self.current_plan
                and self.current_plan.status != PlanStatus.PREEMPTED
                and self.current_plan.status != PlanStatus.FAILED
            ):
                if self.current_plan.plan_id in self._plan_step_indices:
                    del self._plan_step_indices[self.current_plan.plan_id]

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
    # Anomaly & Recovery Flags (Cleaned to only use AnomalyEvent objects)
    # ---------------------
    async def set_anomaly(self, event: AnomalyEvent) -> None:
        """
        Adds a new AnomalyEvent to the Reasoning Trace and registers it
        as an active anomaly.
        """
        async with self._lock:
            # 1. Add to the reasoning trace (Scratchpad)
            self.reasoning_trace.append(event)

            # 2. Register as active anomaly (keyed by semantic key)
            self.active_anomalies[event.key] = event

        self.anomaly_updated.set()

    async def clear_anomaly(self, key: str) -> None:
        """
        Clears/removes an active anomaly flag based on its semantic key.
        """
        async with self._lock:
            if key in self.active_anomalies:
                del self.active_anomalies[key]
        self.anomaly_updated.set()

    async def get_active_anomaly(self, key: str) -> AnomalyEvent | None:
        """
        Retrieve a specific active AnomalyEvent object.
        """
        async with self._lock:
            return self.active_anomalies.get(key)

    async def get_highest_anomaly_severity(self) -> AnomalySeverity | None:
        """
        Scans all active anomalies to return the highest current severity level.
        """
        max_severity = AnomalySeverity.LOW
        # FIX 1: Use the Enum object itself as the key
        max_level = SEVERITY_RANKING[max_severity]

        async with self._lock:
            for event in self.active_anomalies.values():
                # FIX 2: Use the Enum object itself as the key
                event_level = SEVERITY_RANKING.get(event.severity, -1)

                if event_level > max_level:
                    max_level = event_level
                    max_severity = event.severity

            return max_severity if self.active_anomalies else None

    async def is_anomaly_present(
        self, severity_min: AnomalySeverity = AnomalySeverity.MEDIUM
    ) -> bool:
        """
        Checks if any active anomaly meets or exceeds a minimum severity level.
        This is the critical 'Safety Gate' check for the StepExecutor.
        """
        # FIX 3: Use the Enum object itself as the key
        min_level = SEVERITY_RANKING.get(severity_min)

        if min_level is None:
            # This should ideally never happen if severity_min is a valid Enum member
            raise ValueError(
                f"Invalid severity_min: {severity_min}. Must be a valid AnomalySeverity member."
            )

        async with self._lock:
            for event in self.active_anomalies.values():
                # FIX 4: Use the Enum object itself as the key
                flag_level = SEVERITY_RANKING.get(event.severity)

                if flag_level is None:
                    print(
                        f"ERROR: Blackboard has invalid severity stored for {event.key}: {event.severity}"
                    )
                    continue

                if flag_level >= min_level:
                    return True  # Found an active, critical anomaly

            return False  # No critical anomalies found

    async def set_recovery_status(self, key: str, status: str) -> None:
        async with self._lock:
            self.recovery_status[key] = status

    async def get_recovery_status(self, key: str) -> str | None:
        async with self._lock:
            return self.recovery_status.get(key)

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
            self.latest_state_estimate = estimate
            self.state_estimate_updated.set()

    async def get_state_estimate(self) -> StateEstimate | None:
        async with self._lock:
            return self.latest_state_estimate

    async def wait_for_state_estimate(self) -> StateEstimate | None:
        await self.state_estimate_updated.wait()
        return await self.get_state_estimate()

    # ---------------------
    # Extensibility
    # ---------------------
    async def set(self, key: str, value: Any) -> None:
        """
        Store custom state for experimental features.
        Uses separate namespace to avoid corrupting core state.
        """
        async with self._lock:
            self._custom_state[key] = value

    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve custom state."""
        async with self._lock:
            return self._custom_state.get(key, default)
