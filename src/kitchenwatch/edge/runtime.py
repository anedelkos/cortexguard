"""
Edge Runtime - Composition Root for KitchenWatch Edge Agent

This module serves as the composition root (Dependency Injection container)
for the edge agent. It instantiates all subsystems and wires them together
with shared dependencies like the Blackboard singleton.

Architectural Pattern: Composition Root
- Single place where all dependencies are created and wired
- Makes dependency graph explicit and auditable
- Enables easier testing (can create isolated runtime instances)
- Follows Dependency Inversion Principle (DIP)

Production Considerations (not fully implemented):
- Health checks and monitoring
- Graceful degradation on subsystem failures
- Configuration management (env vars, config files)
- Metrics collection and reporting
- Distributed tracing integration
"""

import asyncio
import logging
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from kitchenwatch.common.logging_config import setup_logging
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    """
    Configuration for edge runtime.

    In production, this would be loaded from:
    - Environment variables (12-factor app)
    - Config files (YAML, TOML)
    - Remote config service (AWS AppConfig, GCP Config)
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

    Responsibilities:
    - Create singleton Blackboard and inject into subsystems
    - Initialize all edge components (Orchestrator, Executor, Detectors, etc.)
    - Manage subsystem lifecycle (start, stop, health checks)
    - Handle graceful shutdown on signals
    - Coordinate error recovery across subsystems

    Usage:
        async with EdgeRuntime(config) as runtime:
            await runtime.run_until_stopped()

    or:
        runtime = EdgeRuntime(config)
        await runtime.start()
        try:
            await runtime.run_until_stopped()
        finally:
            await runtime.stop()
    """

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig()
        setup_logging()

        # Singleton Blackboard - shared across all subsystems
        self.blackboard = Blackboard()

        # Core subsystems
        self.orchestrator = Orchestrator(self.blackboard)

        # Runtime state
        self._running = False
        self._stop_event = asyncio.Event()
        self._subsystems_started = False

        logger.info("EdgeRuntime initialized with config: %s", self.config)

    async def start(self) -> None:
        """
        Initialize and start all subsystems.

        Startup Order:
        1. Core infrastructure (Blackboard already created)
        2. Perception (SensorFusion)
        3. Reasoning (LLMPlanner, AnomalyDetector)
        4. Execution (Orchestrator, Executor)

        This order ensures dependencies are ready before dependents start.
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

        # Start all subsystems concurrently
        await asyncio.gather(*start_tasks, return_exceptions=True)

    async def stop(self) -> None:
        """
        Stop all subsystems gracefully.

        Shutdown Order (reverse of startup):
        1. Execution (Orchestrator, Executor)
        2. Reasoning (AnomalyDetector, LLMPlanner)
        3. Perception (SensorFusion)
        4. Core (Blackboard state persisted if needed)
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

            # Stop execution layer first (prevent new work)
            if self.orchestrator:
                stop_tasks.append(self.orchestrator.stop())

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

        Blocks until:
        - SIGINT (Ctrl+C)
        - SIGTERM (kill)
        - stop() called programmatically
        """
        if not self._subsystems_started:
            await self.start()

        loop = asyncio.get_running_loop()
        stop_event = self._stop_event

        def handle_signal(sig: signal.Signals) -> None:
            logger.info(f"Received signal {sig.name}, initiating shutdown...")
            stop_event.set()

        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

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

        Returns:
            Dict mapping subsystem name to health status (True = healthy)

        Production: This would be exposed via HTTP endpoint for monitoring.
        """
        health = {
            "runtime": self._running,
            "blackboard": self.blackboard is not None,
            "orchestrator": self.orchestrator is not None
            and getattr(self.orchestrator, "_loop_running", False),
        }

        # Check subsystem health (would call health_check() on each)

        return health

    @asynccontextmanager
    async def managed(self) -> AsyncIterator["EdgeRuntime"]:
        """
        Context manager for automatic lifecycle management.

        Usage:
            async with EdgeRuntime(config).managed() as runtime:
                # runtime is started
                await runtime.run_until_stopped()
            # runtime is stopped automatically
        """
        try:
            await self.start()
            yield self
        finally:
            await self.stop()

    def get_metrics(self) -> dict[str, int | str | None]:
        """
        Get runtime metrics for monitoring.

        Production: Would export to Prometheus, CloudWatch, etc.
        """
        return {
            "uptime": "TODO",  # Track start time
            "plans_executed": "TODO",  # Track in orchestrator
            "anomalies_detected": len(self.blackboard.anomaly_flags),
            "failed_plans": len(self.blackboard.failed_plans),
            "current_plan_id": (
                self.blackboard.current_plan.plan_id if self.blackboard.current_plan else None
            ),
        }


# Factory function for creating runtime with different profiles
def create_runtime(profile: str = "default", **overrides: dict[str, Any]) -> EdgeRuntime:
    """
    Factory for creating runtime with preset configurations.

    Profiles:
    - default: Standard edge deployment
    - development: Verbose logging, lower thresholds
    - production: Optimized settings, structured logging
    - simulation: Mock sensors, faster tick rates

    Args:
        profile: Configuration profile name
        **overrides: Override specific config values

    Usage:
        runtime = create_runtime("development", log_level="DEBUG")
    """
    configs = {
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

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return EdgeRuntime(config)


# Main entry point
async def main() -> None:
    """
    Main entry point for edge agent.

    Usage:
        python -m kitchenwatch.edge.runtime

    or with profile:
        RUNTIME_PROFILE=development python -m kitchenwatch.edge.runtime
    """
    import os

    profile = os.getenv("RUNTIME_PROFILE", "default")
    logger.info(f"Starting EdgeRuntime with profile: {profile}")

    async with create_runtime(profile).managed() as runtime:
        await runtime.run_until_stopped()


if __name__ == "__main__":
    asyncio.run(main())
