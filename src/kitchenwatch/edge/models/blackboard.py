import asyncio
from dataclasses import dataclass, field
from typing import Any

from kitchenwatch.edge.models.anomaly_severity import AnomalySeverity
from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.plan import IntentContext, Plan, PlanStep
from kitchenwatch.edge.models.state_estimate import StateEstimate

# --- SEVERITY ORDER MAPPING ---
# Maps the AnomalySeverity Enum values to an integer for comparison checks.
# This must match the defined order in the Enum (HIGH > MEDIUM > LOW).
SEVERITY_ORDER: dict[str, int] = {
    AnomalySeverity.LOW.value: 0,
    AnomalySeverity.MEDIUM.value: 1,
    AnomalySeverity.HIGH.value: 2,
}


@dataclass
class Blackboard:
    """
    Async-safe shared memory store for KitchenWatch Edge agent.

    Implements the Blackboard architectural pattern for inter-subsystem
    communication. Acts as the central hub for plan execution state,
    sensor data, and system flags.
    """

    # Core execution state
    current_intent: IntentContext | None = None
    current_step: PlanStep | None = None
    current_plan: Plan | None = None
    paused_plan: Plan | None = None

    # Sensor state
    latest_snapshot: FusionSnapshot | None = None

    # System flags
    # anomaly_flags now stores the severity string (e.g., 'high') for active anomalies.
    # If an anomaly is inactive, its key should be removed from the dict.
    anomaly_flags: dict[str, str] = field(default_factory=dict)
    recovery_status: dict[str, str] = field(default_factory=dict)
    safety_flags: dict[str, bool] = field(default_factory=dict)

    # Failed plans for analysis
    failed_plans: list[Plan] = field(default_factory=list)

    # Custom extensible state
    _custom_state: dict[str, Any] = field(default_factory=dict)

    # Async primitives (created per-instance)
    _lock: asyncio.Lock = asyncio.Lock()
    intent_updated: asyncio.Event = asyncio.Event()
    snapshot_updated: asyncio.Event = asyncio.Event()
    latest_state_estimate: StateEstimate | None = None
    state_estimate_updated: asyncio.Event = asyncio.Event()
    anomaly_updated: asyncio.Event = asyncio.Event()

    # ---------------------
    # Intent Methods
    # ---------------------
    async def update_intent(self, intent: IntentContext) -> None:
        async with self._lock:
            self.current_intent = intent
        self.intent_updated.set()  # Signal outside lock

    async def get_intent(self) -> IntentContext | None:
        async with self._lock:
            return self.current_intent

    async def get_intent_action(self) -> str | None:
        async with self._lock:
            return self.current_intent.action if self.current_intent else None

    async def wait_for_intent_change(self) -> IntentContext | None:
        """Block until intent changes. Caller should clear event after handling."""
        await self.intent_updated.wait()
        return await self.get_intent()

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
            self.current_plan = None
            self.current_step = None

    # ---------------------
    # Anomaly & Recovery Flags
    # ---------------------
    async def set_anomaly_flag(self, key: str, severity: AnomalySeverity, is_active: bool) -> None:
        """
        Set or clear an anomaly flag based on its severity.

        Args:
            key: Unique identifier for the anomaly (e.g., 'repeated_misgrasp').
            severity: The severity level string ('low', 'medium', 'high').
            is_active: If True, set the flag; if False, clear/remove the flag.
        """
        # Validate severity input using the imported enum's values
        if severity.lower() not in SEVERITY_ORDER:
            # Note: We don't raise here, as the Anomaly Detector should be validating it,
            # but we ensure the input is recognizable before storage.
            print(f"WARNING: Invalid severity provided to Blackboard: {severity}")
            return

        async with self._lock:
            if is_active:
                # Store the severity string for later querying
                self.anomaly_flags[key] = severity.lower()
            elif key in self.anomaly_flags:
                # Clear the flag
                del self.anomaly_flags[key]

        self.anomaly_updated.set()

    async def get_anomaly_flag(self, key: str) -> str | None:
        """
        Retrieve the severity string of an active anomaly flag, or None if inactive.
        """
        async with self._lock:
            return self.anomaly_flags.get(key)

    async def is_anomaly_present(
        self, key_prefix: str | None = None, severity_min: AnomalySeverity = AnomalySeverity.MEDIUM
    ) -> bool:
        """
        Checks if any active anomaly flag meets or exceeds a minimum severity level.
        This is the critical 'Safety Gate' check for the StepExecutor.

        Args:
            key_prefix: Optional prefix to filter anomaly keys (e.g., 'system_health').
            severity_min: The minimum AnomalySeverity required to return True.
                          Defaults to MEDIUM to trigger a pause on critical issues.

        Returns:
            True if one or more matching anomaly flags are active at or above severity_min.
        """
        min_level_str = severity_min.value.lower()
        min_level = SEVERITY_ORDER.get(min_level_str)

        # This should theoretically never happen if the Enum is used correctly,
        # but serves as a safety check against improper enum usage.
        if min_level is None:
            raise ValueError(
                f"Invalid severity_min: {severity_min}. Must be a valid AnomalySeverity member."
            )

        async with self._lock:
            for key, severity_str in self.anomaly_flags.items():
                # 1. Filter by key prefix if provided
                if key_prefix and not key.startswith(key_prefix):
                    continue

                # 2. Check severity level
                flag_level = SEVERITY_ORDER.get(severity_str)

                if flag_level is None:
                    # Log internal error if a severity string stored in the dict is invalid
                    print(
                        f"ERROR: Blackboard has invalid severity stored for {key}: {severity_str}"
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
