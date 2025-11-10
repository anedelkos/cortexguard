import asyncio

from pydantic import BaseModel

from kitchenwatch.edge.models.fusion_snapshot import FusionSnapshot
from kitchenwatch.edge.models.plan import IntentContext, Plan, PlanStep


class Blackboard(BaseModel):
    """
    Async-safe shared memory and context store for the KitchenWatch Edge agent.

    Acts as the central hub for inter-subsystem communication within the edge runtime.
    Subsystems read from and write to the Blackboard to share the current state, intent,
    and sensor-derived information.

    Responsibilities:
        - Store the current intent context (plan, step, action, parameters)
        - Keep the latest FusionSnapshot(s) or aggregated sensor summaries
        - Track plan execution states (running, paused, completed)
        - Maintain anomaly flags and recovery status for other agents
        - Maintain safety flags for high-priority overrides
        - Provide async-safe access to all shared data
    """

    model_config = {"arbitrary_types_allowed": True}

    current_intent: IntentContext | None = None
    current_step: PlanStep | None = None
    current_plan: Plan | None = None
    paused_plan: Plan | None = None
    latest_snapshot: FusionSnapshot | None = None

    anomaly_flags: dict[str, bool] = {}
    recovery_status: dict[str, str] = {}
    safety_flags: dict[str, bool] = {}

    # Internal asyncio lock for async-safe access
    _lock: asyncio.Lock = asyncio.Lock()

    # Async Events to signal important changes (optional)
    intent_updated: asyncio.Event = asyncio.Event()
    snapshot_updated: asyncio.Event = asyncio.Event()
    anomaly_updated: asyncio.Event = asyncio.Event()

    # ---------------------
    # Intent Methods
    # ---------------------
    async def update_intent(self, intent: IntentContext) -> None:
        async with self._lock:
            self.current_intent = intent
            self.intent_updated.set()
            self.intent_updated.clear()

    async def get_intent(self) -> IntentContext | None:
        async with self._lock:
            return self.current_intent

    async def get_intent_action(self) -> str | None:
        async with self._lock:
            return self.current_intent.action if self.current_intent else None

    # ---------------------
    # Snapshot Methods
    # ---------------------
    async def update_fusion_snapshot(self, snapshot: FusionSnapshot) -> None:
        async with self._lock:
            self.latest_snapshot = snapshot
            self.snapshot_updated.set()
            self.snapshot_updated.clear()

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

    # ---------------------
    # Anomaly & Recovery Flags
    # ---------------------
    async def set_anomaly_flag(self, key: str, value: bool) -> None:
        async with self._lock:
            self.anomaly_flags[key] = value
            self.anomaly_updated.set()
            self.anomaly_updated.clear()

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

    async def set(self, key: str, value: object) -> None:
        """
        Generic setter for arbitrary blackboard attributes.
        Only sets attributes that already exist on the blackboard.
        """
        async with self._lock:
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Blackboard has no attribute '{key}'")
