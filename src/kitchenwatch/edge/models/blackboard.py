import asyncio
from dataclasses import dataclass, field
from typing import Any

from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.plan import IntentContext, Plan, PlanStep
from kitchenwatch.edge.models.state_estimate import StateEstimate


@dataclass
class Blackboard:
    """
    Async-safe shared memory store for KitchenWatch Edge agent.

    Implements the Blackboard architectural pattern for inter-subsystem
    communication. Acts as the central hub for plan execution state,
    sensor data, and system flags.

    Design Philosophy:
    - Single lock for simplicity (adequate for edge workload)
    - Events for non-blocking state change notifications
    - Type-safe accessors for core state
    - Extensible via custom_state for experimentation

    Production Enhancements (not implemented):
    - Fine-grained locking per state category
    - State snapshots for debugging/replay
    - Metrics/observability hooks
    """

    # Core execution state
    current_intent: IntentContext | None = None
    current_step: PlanStep | None = None
    current_plan: Plan | None = None
    paused_plan: Plan | None = None

    # Sensor state
    latest_snapshot: FusionSnapshot | None = None

    # System flags
    anomaly_flags: dict[str, bool] = field(default_factory=dict)
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
    async def set_anomaly_flag(self, key: str, value: bool) -> None:
        async with self._lock:
            self.anomaly_flags[key] = value
        self.anomaly_updated.set()

    async def get_anomaly_flag(self, key: str) -> bool | None:
        async with self._lock:
            return self.anomaly_flags.get(key)

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
