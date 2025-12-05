"""
Edge Runtime - Composition Root for KitchenWatch Edge Agent

This module serves as the composition root (Dependency Injection container)
for the edge agent. It instantiates all subsystems and wires them together
with shared dependencies like the Blackboard singleton. It also contains the
FastAPI factory function used by Uvicorn.

Architectural Pattern: Composition Root
- Single place where all dependencies are created and wired
- Makes dependency graph explicit and auditable
- Enables easier testing (can create isolated runtime instances)
- Follows Dependency Inversion Principle (DIP)
"""

import asyncio
import logging
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

from fastapi import FastAPI

from kitchenwatch.common.logging_config import setup_logging
from kitchenwatch.core.interfaces.base_online_learner import BaseOnlineLearner
from kitchenwatch.core.interfaces.base_policy_engine import BasePolicyEngine
from kitchenwatch.core.mocks.mock_controller import MockController
from kitchenwatch.core.mocks.mock_step_classifier import MockStepClassifier
from kitchenwatch.edge.api import health
from kitchenwatch.edge.api.ingestion import get_ingestion_router
from kitchenwatch.edge.arbiter import Arbiter
from kitchenwatch.edge.detectors.anomaly_detector import AnomalyDetector
from kitchenwatch.edge.detectors.numeric import StatisticalImpulseDetector
from kitchenwatch.edge.detectors.rule_based import HardLimitDetector, LogicalRuleDetector
from kitchenwatch.edge.detectors.vision.vision_safety_detector import (
    VisionSafetyDetector,
    VisionSafetyDetectorConfig,
)
from kitchenwatch.edge.edge_fusion import EdgeFusion, VisionEmbedder
from kitchenwatch.edge.local_receiver import LocalReceiver
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.capability_registry import CapabilityRegistry
from kitchenwatch.edge.online_learner_state_estimator import OnlineLearnerStateEstimator
from kitchenwatch.edge.orchestrator import Orchestrator
from kitchenwatch.edge.policy.mistral_policy_engine import MistralLLMPolicyEngine
from kitchenwatch.edge.policy.policy_agent import PolicyAgent
from kitchenwatch.edge.river_online_learner import RiverOnlineLearner
from kitchenwatch.edge.safety_agent import SafetyAgent
from kitchenwatch.edge.step_executor import StepExecutor

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    """
    Configuration for edge runtime.
    """

    # Orchestrator settings
    orchestrator_tick_interval: float = 0.1

    # Executor settings
    executor_max_retries: int = 3
    executor_retry_delay: float = 0.5

    # Anomaly detection settings
    anomaly_check_interval: float = 1.0
    anomaly_threshold: float = 0.8

    # Sensor fusion settings
    sensor_fusion_rate: float = 0.1  # 10 Hz

    # LLM settings (for plan generation)
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    # Logging
    log_level: str = "INFO"

    # Graceful shutdown timeout
    shutdown_timeout: float = 10.0


class EdgeRuntime:
    """
    Main edge runtime that composes and manages all subsystems.
    """

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig()
        setup_logging()

        # Singleton Blackboard - shared across all subsystems
        self.blackboard = Blackboard()

        # --- CORE AGENT SUBSYSTEMS ---
        # Create controller & registry
        self.controller = MockController()  # or real robot controller
        self.capability_registry = CapabilityRegistry()
        self.step_classifier = MockStepClassifier()

        self.arbiter = Arbiter(
            blackboard=self.blackboard,
            capability_registry=self.capability_registry,
            controller=self.controller,
        )
        self.safety_agent = SafetyAgent(self.blackboard)
        self.orchestrator = Orchestrator(
            blackboard=self.blackboard,
            arbiter=self.arbiter,
            safety_agent=self.safety_agent,
        )

        # Step executor
        self.executor = StepExecutor(
            blackboard=self.blackboard,
            step_classifier=self.step_classifier,
            capability_registry=self.capability_registry,
            default_max_retries=self.config.executor_max_retries,
            default_retry_delay=self.config.executor_retry_delay,
            controller=self.controller,
        )

        # --- PERCEPTION SUBSYSTEM (Fusion and Learning) ---
        # 1. Instantiate Vision Embedder
        self.vision_embedder = VisionEmbedder()

        # 2. Instantiate the learning dependency
        self.online_learner: BaseOnlineLearner = RiverOnlineLearner()

        # 3. Instantiate the State Estimator (Translates residuals to Z-scores)
        self.state_estimator = OnlineLearnerStateEstimator(self.online_learner, self.blackboard)

        # 4. Instantiate Edge Fusion
        self.edge_fusion = EdgeFusion(
            blackboard=self.blackboard,
            state_estimator=self.state_estimator,
            embedder=self.vision_embedder,
        )

        # 5. Instantiate Local Receiver (The API ingestion endpoint dependency)
        self.receiver = LocalReceiver(verbose=False, edge_fusion=self.edge_fusion)

        # --- ANOMALY DETECTION SUBSYSTEM (Ensemble) ---
        # 1. Instantiate the Anomaly Detector (The Manager/Coordinator)
        self.anomaly_detector = AnomalyDetector(
            blackboard=self.blackboard,
            tick_interval=self.config.anomaly_check_interval,
        )

        # ... Register Sub-Detectors ...
        # S0.3 Impact Detector (Statistical)
        self.statistical_impulse_detector = StatisticalImpulseDetector(
            state_estimator=self.state_estimator,
            z_score_threshold=self.config.anomaly_threshold,
        )
        self.anomaly_detector.register_detector(self.statistical_impulse_detector)
        # S0.2 Overheat Detector
        self.hard_limit_detector = HardLimitDetector()
        self.anomaly_detector.register_detector(self.hard_limit_detector)
        # S1.1: Repeated system failures, S2.3: Sensor/Blackboard data freeze
        self.logical_rule_detector = LogicalRuleDetector()
        self.anomaly_detector.register_detector(self.logical_rule_detector)

        # S1.x Vision reflex detector (human proximity, occlusion)
        self.vision_safety_detector = VisionSafetyDetector(
            VisionSafetyDetectorConfig(safety_radius_m=0.5, min_confidence=0.6)
        )
        self.anomaly_detector.register_detector(self.vision_safety_detector)

        # --- REASONING SUBSYSTEM (Policy Agent) ---
        # 1. Instantiate the Policy Engine (LLM)
        self.policy_engine: BasePolicyEngine = MistralLLMPolicyEngine(
            use_mock=True  # Use mock for safe/fast edge deployment
        )

        # 2. Instantiate the Policy Agent (The anomaly-to-policy rules engine)
        self.policy_agent = PolicyAgent(
            blackboard=self.blackboard,
            capability_registry=self.capability_registry,
            policy_engine=self.policy_engine,
        )

        # Runtime state
        self._running = False
        self._stop_event = asyncio.Event()
        self._subsystems_started = False

        logger.info("EdgeRuntime initialized with config: %s", self.config)

    async def start(self) -> None:
        """
        Initialize and start all subsystems.
        """
        if self._subsystems_started:
            logger.warning("EdgeRuntime already started")
            return

        logger.info("Starting EdgeRuntime...")

        try:
            # Start all subsystems
            logger.info("Starting subsystems...")
            await self._start_subsystems()

            self._subsystems_started = True
            self._running = True
            logger.info("✅ EdgeRuntime started successfully")

        except Exception as e:
            logger.exception("Failed to start EdgeRuntime: %s", e)
            await self.stop()  # Cleanup on failure
            raise

    async def _start_subsystems(self) -> None:
        """Start all subsystems in parallel."""
        start_tasks = []

        if self.orchestrator:
            start_tasks.append(
                self.orchestrator.start(tick_interval=self.config.orchestrator_tick_interval)
            )
        if self.executor:
            start_tasks.append(self.executor.start())
        if self.anomaly_detector:
            start_tasks.append(self.anomaly_detector.start())
        if self.policy_agent:
            start_tasks.append(self.policy_agent.start())

        # Start all subsystems concurrently and check for failures
        results = await asyncio.gather(*start_tasks, return_exceptions=True)

        # Check for startup failures
        failed = False
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Subsystem {i} failed to start: {result}", exc_info=result)
                failed = True

        if failed:
            raise RuntimeError("One or more subsystems failed to start")

    async def stop(self) -> None:
        """
        Stop all subsystems gracefully.
        """
        if not self._subsystems_started:
            logger.warning("EdgeRuntime not started, nothing to stop")
            return

        logger.info("Stopping EdgeRuntime...")
        self._running = False
        self._stop_event.set()

        try:
            # Stop subsystems with timeout
            stop_tasks = []

            # 1. Stop execution layer first (prevent new work)
            if self.executor:
                stop_tasks.append(self.executor.stop())
            if self.orchestrator:
                stop_tasks.append(self.orchestrator.stop())

            # 2. Stop Reasoning/Detection layer
            if self.anomaly_detector:
                stop_tasks.append(self.anomaly_detector.stop())
            if self.policy_agent:
                stop_tasks.append(self.policy_agent.stop())

            # Wait for graceful shutdown with timeout
            await asyncio.wait_for(
                asyncio.gather(*stop_tasks, return_exceptions=True),
                timeout=self.config.shutdown_timeout,
            )

            logger.info("✅ EdgeRuntime stopped successfully")

        except TimeoutError:
            logger.error("Shutdown timeout exceeded, forcing stop")
        except Exception as e:
            logger.exception("Error during shutdown: %s", e)
        finally:
            self._subsystems_started = False

    async def run_until_stopped(self) -> None:
        """
        Run the edge runtime until stop signal received.
        """
        if not self._subsystems_started:
            await self.start()

        loop = asyncio.get_running_loop()
        stop_event = self._stop_event

        def handle_signal(sig: signal.Signals) -> None:
            logger.info(f"Received signal {sig.name}, initiating shutdown...")
            stop_event.set()

        # Register signal handlers using functools.partial
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, partial(handle_signal, sig))

        logger.info("EdgeRuntime running. Press Ctrl+C to stop.")

        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            # Expected during shutdown
            pass
        finally:
            # Explicitly check that loop is still open before removing handlers
            if loop.is_running():
                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop.remove_signal_handler(sig)
                    except Exception as e:
                        logger.debug(f"Failed to remove signal handler for {sig}: {e}")

            logger.info("Signal handlers removed, runtime stopping.")

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all subsystems.
        """
        health = {
            "runtime": self._running,
            "blackboard": self.blackboard is not None,
            "orchestrator": self.orchestrator is not None
            and getattr(self.orchestrator, "_loop_running", False),
            "executor": self.executor is not None
            and getattr(self.executor, "_loop_running", False),
            "anomaly_detector": self.anomaly_detector is not None
            and getattr(self.anomaly_detector, "_loop_running", False),
        }

        return health

    @asynccontextmanager
    async def managed(self) -> AsyncIterator["EdgeRuntime"]:
        """
        Context manager for automatic lifecycle management.
        """
        try:
            await self.start()
            yield self
        finally:
            await self.stop()

    async def get_metrics(self) -> dict[str, int | str | None]:
        """
        Get runtime metrics for monitoring.
        """
        if not self.blackboard:
            return {
                "uptime": "TODO",
                "plans_executed": "TODO",
                "anomalies_detected": 0,
                "failed_plans": 0,
                "current_plan_id": None,
            }

        # 1. Get the current plan via its getter
        current_plan = await self.blackboard.get_current_plan()

        # 2. Access the metrics counts under the Blackboard's internal lock.
        async with self.blackboard._lock:
            anomaly_count = len(self.blackboard.active_anomalies)
            failed_plan_count = len(self.blackboard.failed_plans)

        return {
            "uptime": "TODO",  # Track start time
            "plans_executed": "TODO",  # Track in orchestrator
            "anomalies_detected": anomaly_count,
            "failed_plans": failed_plan_count,
            "current_plan_id": current_plan.plan_id if current_plan else None,
        }


# Factory function for creating runtime with different profiles
def create_runtime(profile: str = "default", **overrides: Any) -> EdgeRuntime:
    """
    Factory for creating runtime with preset configurations.
    """
    configs: dict[str, RuntimeConfig] = {
        "default": RuntimeConfig(),
        "development": RuntimeConfig(
            log_level="DEBUG",
            orchestrator_tick_interval=0.05,  # Faster for dev
            anomaly_threshold=0.7,  # Lower threshold for testing
        ),
        "production": RuntimeConfig(
            log_level="INFO",
            orchestrator_tick_interval=0.1,
            executor_max_retries=5,
            shutdown_timeout=30.0,
        ),
        "simulation": RuntimeConfig(
            log_level="DEBUG",
            orchestrator_tick_interval=0.01,  # Fast simulation
            sensor_fusion_rate=0.01,
            anomaly_check_interval=0.5,
        ),
    }

    config = configs.get(profile, configs["default"])

    # Apply overrides with validation
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config override ignored: {key}={value}")

    return EdgeRuntime(config)


# --- FASTAPI INTEGRATION / UVICORN ENTRY POINT ---


def get_api_app(profile: str = "default") -> FastAPI:
    """
    The main factory function for the FastAPI application.

    This is the entry point used by Uvicorn:
    `uvicorn kitchenwatch.edge.runtime:get_api_app`
    """
    import os

    # Ensure logging is set up before anything else
    setup_logging()

    # Determine the runtime profile
    runtime_profile = os.getenv("RUNTIME_PROFILE", profile)
    logger.info(f"Initializing EdgeRuntime and FastAPI with profile: {runtime_profile}")

    # Create the runtime instance. Its lifecycle is managed by the lifespan context.
    runtime = create_runtime(profile=runtime_profile)

    # Use the runtime's built-in managed context as the FastAPI lifespan
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # Added return type annotation
        """FastAPI Lifespan: Calls start/stop on the EdgeRuntime."""
        async with runtime.managed() as started_runtime:
            app.state.runtime = started_runtime
            yield

    app = FastAPI(
        title="KitchenWatch Edge API",
        description="Edge ingestion endpoint for fused records from the simulator.",
        lifespan=lifespan,  # Set the custom lifespan handler
    )

    # --- Router Wiring and Dependency Injection ---

    # 1. Inject the LocalReceiver instance into the ingestion router factory
    receiver_instance = runtime.receiver
    ingestion_router = get_ingestion_router(receiver=receiver_instance)

    # 2. Include the routers
    app.include_router(health.router)
    app.include_router(ingestion_router, prefix="/api/v1")

    # 3. Expose Health and Metrics (using the runtime instance created above)
    @app.get("/healthz")
    async def get_healthz() -> dict[str, bool]:
        """Exposes detailed health checks for the EdgeRuntime subsystems."""
        # Delegating to EdgeRuntime's method which returns dict[str, bool]
        return await runtime.health_check()

    @app.get("/metrics")
    async def get_metrics() -> dict[str, int | str | None]:
        """Exposes key runtime metrics (anomalies, plans, etc.)."""
        # Delegating to EdgeRuntime's method which returns dict[str, int | str | None]
        return await runtime.get_metrics()

    logger.info("FastAPI Application configured and wired to EdgeRuntime.")
    return app


# Main entry point for standalone execution (for debugging/non-API use)
async def main() -> None:
    """
    Main entry point for edge agent (standalone, non-API use).
    """
    import os

    profile = os.getenv("RUNTIME_PROFILE", "default")
    logger.info(f"Starting EdgeRuntime with profile: {profile} (Standalone Mode)")

    async with create_runtime(profile).managed() as runtime:
        await runtime.run_until_stopped()


if __name__ == "__main__":
    # If running directly, default to standalone mode
    asyncio.run(main())
