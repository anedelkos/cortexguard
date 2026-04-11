"""
Edge Runtime - Composition Root for CortexGuard Edge Agent

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

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from slowapi.errors import RateLimitExceeded

from cortexguard.common.logging_config import setup_logging
from cortexguard.core.interfaces.base_online_learner import BaseOnlineLearner
from cortexguard.core.interfaces.base_policy_engine import BasePolicyEngine
from cortexguard.core.mocks.mock_cloud_agent import MockCloudAgentClient
from cortexguard.core.mocks.mock_controller import MockController
from cortexguard.core.mocks.mock_step_classifier import MockStepClassifier
from cortexguard.edge.api import health
from cortexguard.edge.api import metrics as metrics_router
from cortexguard.edge.api.health import get_health_router
from cortexguard.edge.api.ingestion import get_ingestion_router, limiter
from cortexguard.edge.arbiter import Arbiter
from cortexguard.edge.detectors.anomaly_detector import AnomalyDetector
from cortexguard.edge.detectors.numeric import StatisticalImpulseDetector
from cortexguard.edge.detectors.rule_based import HardLimitDetector, LogicalRuleDetector
from cortexguard.edge.detectors.vision.vision_safety_detector import (
    VisionSafetyDetector,
    VisionSafetyDetectorConfig,
)
from cortexguard.edge.edge_fusion import _VISION_AVAILABLE, EdgeFusion, VisionEmbedder
from cortexguard.edge.local_receiver import LocalReceiver
from cortexguard.edge.mayday_agent import MaydayAgent
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.capability_registry import CapabilityRegistry
from cortexguard.edge.observability.opentelemetry_tracing import setup_opentelemetry_tracing
from cortexguard.edge.online_learner_state_estimator import OnlineLearnerStateEstimator
from cortexguard.edge.orchestrator import Orchestrator
from cortexguard.edge.persistence.persistence_manager import PersistenceManager
from cortexguard.edge.policy.mistral_policy_engine import MistralLLMPolicyEngine
from cortexguard.edge.policy.policy_agent import PolicyAgent
from cortexguard.edge.river_online_learner import RiverOnlineLearner
from cortexguard.edge.safety_agent import SafetyAgent
from cortexguard.edge.step_executor import StepExecutor
from cortexguard.edge.utils.metrics import http_requests_total
from cortexguard.edge.utils.tracing import TraceSink

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
    executor_poll_interval: float = 0.05
    executor_idle_interval: float = 0.1

    # Anomaly detection settings
    anomaly_check_interval: float = 1.0

    # Sensor fusion settings
    sensor_fusion_rate: float = 0.1  # 10 Hz

    # Logging
    log_level: str = "INFO"

    # Graceful shutdown timeout
    shutdown_timeout: float = 10.0

    # Observability / OTEL
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTLP_ENDPOINT", "http://tempo:4318/v1/traces")
    )
    service_name: str = "CortexGuard-edge"

    # Device identity
    device_id: str = field(default_factory=lambda: os.getenv("DEVICE_ID", "mock_01"))

    # Policy engine
    policy_model_id: str = field(
        default_factory=lambda: os.getenv("POLICY_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    )
    policy_use_mock: bool = field(
        default_factory=lambda: os.getenv("POLICY_USE_MOCK", "true").lower() == "true"
    )

    # Ingest rate limit
    ingest_rate_limit: str = field(
        default_factory=lambda: os.getenv("INGEST_RATE_LIMIT", "100/second")
    )

    # Policy remediation cooldown
    policy_remediation_cooldown_s: float = field(
        default_factory=lambda: float(os.getenv("POLICY_REMEDIATION_COOLDOWN_S", "30.0"))
    )

    # Persistence
    persistence_enabled: bool = field(
        default_factory=lambda: os.getenv("PERSISTENCE_ENABLED", "false").lower() == "true"
    )
    persistence_file_path: str = field(
        default_factory=lambda: os.getenv(
            "PERSISTENCE_FILE_PATH", "/var/lib/cortexguard/blackboard.json"
        )
    )
    persistence_snapshot_interval: float = field(
        default_factory=lambda: float(os.getenv("PERSISTENCE_SNAPSHOT_INTERVAL", "5.0"))
    )
    fusion_force_min_n: float = field(
        default_factory=lambda: float(os.getenv("FUSION_FORCE_MIN_N", "0.0"))
    )
    fusion_force_drop_pct: float = field(
        default_factory=lambda: float(os.getenv("FUSION_FORCE_DROP_PCT", "100.1"))
    )
    fusion_drift_fail_mm: float = field(
        default_factory=lambda: float(os.getenv("FUSION_DRIFT_FAIL_MM", "10.0"))
    )

    # LLM circuit breaker
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "30.0")))
    llm_failure_threshold: int = field(
        default_factory=lambda: int(os.getenv("LLM_FAILURE_THRESHOLD", "3"))
    )
    llm_cooldown_s: float = field(
        default_factory=lambda: float(os.getenv("LLM_COOLDOWN_S", "60.0"))
    )

    # Safety and detection thresholds
    detector_temp_threshold_c: float = field(
        default_factory=lambda: float(os.getenv("DETECTOR_TEMP_THRESHOLD_C", "70.0"))
    )
    detector_z_score_threshold: float = field(
        default_factory=lambda: float(os.getenv("DETECTOR_Z_SCORE_THRESHOLD", "5.0"))
    )
    estimator_sigma_threshold: float = field(
        default_factory=lambda: float(os.getenv("ESTIMATOR_SIGMA_THRESHOLD", "3.0"))
    )
    safety_radius_m: float = field(
        default_factory=lambda: float(os.getenv("SAFETY_RADIUS_M", "0.5"))
    )
    fusion_smoke_ppm_threshold: float = field(
        default_factory=lambda: float(os.getenv("FUSION_SMOKE_PPM_THRESHOLD", "50.0"))
    )

    # Sensor timing
    fusion_expected_period_ms: int = field(
        default_factory=lambda: int(os.getenv("FUSION_EXPECTED_PERIOD_MS", "50"))
    )
    fusion_soft_degrade_ms: int = field(
        default_factory=lambda: int(os.getenv("FUSION_SOFT_DEGRADE_MS", "200"))
    )
    fusion_max_gap_ms: int = field(
        default_factory=lambda: int(os.getenv("FUSION_MAX_GAP_MS", "500"))
    )


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
        self.controller = MockController()  # or real hardware controller
        self.cloud_agent = MockCloudAgentClient()
        self.capability_registry = CapabilityRegistry()
        self.step_classifier = MockStepClassifier()

        self.arbiter = Arbiter(
            blackboard=self.blackboard,
            capability_registry=self.capability_registry,
            controller=self.controller,
        )
        self.safety_agent = SafetyAgent(
            self.blackboard, safety_radius_m=self.config.safety_radius_m
        )
        self.orchestrator = Orchestrator(
            blackboard=self.blackboard,
            arbiter=self.arbiter,
            safety_agent=self.safety_agent,
            tick_interval=self.config.orchestrator_tick_interval,
        )

        # Step executor
        self.executor = StepExecutor(
            blackboard=self.blackboard,
            step_classifier=self.step_classifier,
            capability_registry=self.capability_registry,
            default_max_retries=self.config.executor_max_retries,
            default_retry_delay=self.config.executor_retry_delay,
            default_poll_interval=self.config.executor_poll_interval,
            default_idle_interval=self.config.executor_idle_interval,
            controller=self.controller,
        )

        # --- PERCEPTION SUBSYSTEM (Fusion and Learning) ---
        # 1. Instantiate Vision Embedder (requires torch/torchvision — skipped in slim mode)
        self.vision_embedder = VisionEmbedder() if _VISION_AVAILABLE else None

        # 2. Instantiate the learning dependency
        self.online_learner: BaseOnlineLearner = RiverOnlineLearner()

        # 3. Instantiate the State Estimator (Translates residuals to Z-scores)
        self.state_estimator = OnlineLearnerStateEstimator(
            self.online_learner,
            self.blackboard,
            sigma_threshold=self.config.estimator_sigma_threshold,
        )

        # 4. Instantiate Edge Fusion
        self.edge_fusion = EdgeFusion(
            blackboard=self.blackboard,
            state_estimator=self.state_estimator,
            embedder=self.vision_embedder,
            force_min_n=self.config.fusion_force_min_n,
            force_drop_pct=self.config.fusion_force_drop_pct,
            drift_fail_mm=self.config.fusion_drift_fail_mm,
            smoke_ppm_threshold=self.config.fusion_smoke_ppm_threshold,
            expected_period_ms=self.config.fusion_expected_period_ms,
            soft_degrade_ms=self.config.fusion_soft_degrade_ms,
            max_gap_ms=self.config.fusion_max_gap_ms,
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
            blackboard=self.blackboard,
            z_score_threshold=self.config.detector_z_score_threshold,
        )
        self.anomaly_detector.register_detector(self.statistical_impulse_detector)
        # S0.2 Overheat Detector
        self.hard_limit_detector = HardLimitDetector(
            temp_threshold=self.config.detector_temp_threshold_c
        )
        self.anomaly_detector.register_detector(self.hard_limit_detector)
        # S1.1: Repeated system failures, S2.3: Sensor/Blackboard data freeze
        self.logical_rule_detector = LogicalRuleDetector()
        self.anomaly_detector.register_detector(self.logical_rule_detector)

        # S1.x Vision reflex detector (human proximity, occlusion)
        self.vision_safety_detector = VisionSafetyDetector(
            VisionSafetyDetectorConfig(
                safety_radius_m=self.config.safety_radius_m, min_confidence=0.6
            )
        )
        self.anomaly_detector.register_detector(self.vision_safety_detector)

        # --- REASONING SUBSYSTEM (Policy Agent) ---
        # 1. Instantiate the Policy Engine (LLM)
        self.policy_engine: BasePolicyEngine = MistralLLMPolicyEngine(
            use_mock=self.config.policy_use_mock,
            model_id=self.config.policy_model_id,
        )
        self.mayday_agent = MaydayAgent(
            cloud_agent_client=self.cloud_agent,
            device_id=self.config.device_id,
            trace_sink=TraceSink(blackboard=self.blackboard),
        )

        # 2. Instantiate the Policy Agent (The anomaly-to-policy rules engine)
        self.policy_agent = PolicyAgent(
            blackboard=self.blackboard,
            capability_registry=self.capability_registry,
            policy_engine=self.policy_engine,
            mayday_agent=self.mayday_agent,
            remediation_cooldown_s=self.config.policy_remediation_cooldown_s,
            llm_timeout_s=self.config.llm_timeout_s,
            llm_failure_threshold=self.config.llm_failure_threshold,
            llm_cooldown_s=self.config.llm_cooldown_s,
            plan_adder=self.orchestrator.add_plan,
        )

        # Persistence
        self.persistence_manager: PersistenceManager | None = None
        if self.config.persistence_enabled:
            self.persistence_manager = PersistenceManager(
                blackboard=self.blackboard,
                file_path=Path(self.config.persistence_file_path),
                snapshot_interval=self.config.persistence_snapshot_interval,
            )

        # Runtime state
        self._running = False
        self._stop_event = asyncio.Event()
        self._subsystems_started = False
        self._edge_fusion_context: EdgeFusion | None = None
        self._start_time: float | None = None

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
            # Enter EdgeFusion context manager
            self._edge_fusion_context = await self.edge_fusion.__aenter__()

            # Start persistence and restore Blackboard state before subsystems launch
            if self.persistence_manager is not None:
                await self.persistence_manager.start()
                await self.persistence_manager.restore()

            # Start all subsystems
            logger.info("Starting subsystems...")
            await self._start_subsystems()

            self._subsystems_started = True
            self._running = True
            self._start_time = time.monotonic()
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
        """Stop all subsystems gracefully."""
        logger.info("Stop signal received. Flushing state...")
        self._running = False
        self._stop_event.set()

        # Always attempt to stop subsystems even if _subsystems_started is False
        logger.info("Stopping EdgeRuntime subsystems...")
        try:
            stop_tasks = []

            if self.executor:
                stop_tasks.append(self.executor.stop())
            if self.orchestrator:
                stop_tasks.append(self.orchestrator.stop())
            if self.anomaly_detector:
                stop_tasks.append(self.anomaly_detector.stop())
            if self.policy_agent:
                stop_tasks.append(self.policy_agent.stop())

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
            # Flush final persistence snapshot AFTER subsystems stop (captures terminal state)
            if self.persistence_manager is not None:
                try:
                    await self.persistence_manager.stop()
                except Exception as e:
                    logger.exception("Error stopping PersistenceManager: %s", e)

            # Cleanup EdgeFusion context
            if self._edge_fusion_context is not None:
                try:
                    await self.edge_fusion.__aexit__(None, None, None)
                except Exception as e:
                    logger.exception("Error cleaning up EdgeFusion context: %s", e)
                finally:
                    self._edge_fusion_context = None

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
    async def managed(self) -> AsyncIterator[EdgeRuntime]:
        """
        Context manager for automatic lifecycle management.
        """
        try:
            await self.start()
            yield self
        finally:
            await self.stop()

    async def get_metrics(self) -> dict[str, int | float | str | None]:
        """
        Get runtime metrics for monitoring.
        """
        uptime = round(time.monotonic() - self._start_time, 1) if self._start_time else 0.0
        plans_executed = self.orchestrator._plans_executed if self.orchestrator else 0

        if not self.blackboard:
            return {
                "uptime_seconds": uptime,
                "plans_executed": plans_executed,
                "anomalies_detected": 0,
                "failed_plans": 0,
                "current_plan_id": None,
            }

        current_plan = await self.blackboard.get_current_plan()
        metrics = await self.blackboard.get_metrics()

        emergency_stop = await self.blackboard.get_safety_flag("emergency_stop")

        return {
            "uptime_seconds": uptime,
            "plans_executed": plans_executed,
            "anomalies_detected": metrics["active_anomalies_count"],
            "failed_plans": metrics["failed_plans_count"],
            "current_plan_id": current_plan.plan_id if current_plan else None,
            "emergency_stop": emergency_stop,
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
    `uvicorn cortexguard.edge.runtime:get_api_app`
    """
    import os

    # Ensure logging is set up before anything else
    setup_logging()
    config = RuntimeConfig()
    setup_opentelemetry_tracing(config.service_name, config.otlp_endpoint)

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
        title="CortexGuard Edge API",
        description="Edge ingestion endpoint for fused records from the simulator.",
        lifespan=lifespan,  # Set the custom lifespan handler
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> Response:
        http_requests_total.labels(method="POST", status_code="422").inc()
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded) -> Response:
        http_requests_total.labels(method="POST", status_code="429").inc()
        return Response(
            content=f'{{"error": "Rate limit exceeded: {exc.detail}"}}',
            status_code=429,
            media_type="application/json",
        )

    # --- Router Wiring and Dependency Injection ---

    # 1. Inject the LocalReceiver instance into the ingestion router factory
    receiver_instance = runtime.receiver
    ingestion_router = get_ingestion_router(
        receiver=receiver_instance, rate_limit=runtime.config.ingest_rate_limit
    )

    # 2. Include the routers
    app.include_router(health.router)
    app.include_router(get_health_router(runtime.health_check))
    app.include_router(ingestion_router, prefix="/api/v1")
    app.include_router(metrics_router.router)

    # 3. Rate limiting
    app.state.limiter = limiter

    # 4. OTEL FastAPI instrumentation — creates spans for every HTTP request
    FastAPIInstrumentor.instrument_app(app)

    @app.get("/runtime-metrics")
    async def get_metrics() -> dict[str, int | float | str | None]:
        """Exposes key runtime metrics (anomalies, plans, etc.)."""
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
