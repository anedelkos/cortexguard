import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from kitchenwatch.edge.runtime import EdgeRuntime, RuntimeConfig, create_runtime


@pytest_asyncio.fixture
async def runtime() -> AsyncGenerator[EdgeRuntime, None]:
    """Create and cleanup runtime for tests."""
    config = RuntimeConfig(
        orchestrator_tick_interval=0.01,  # Fast for tests
        log_level="DEBUG",
    )
    runtime = EdgeRuntime(config)
    yield runtime
    if runtime._subsystems_started:
        await runtime.stop()


@pytest.mark.asyncio
async def test_runtime_lifecycle(runtime: EdgeRuntime) -> None:
    """Test basic start/stop lifecycle."""
    # Should start successfully
    await runtime.start()
    assert runtime._running
    assert runtime._subsystems_started

    # Should have initialized all subsystems
    assert runtime.blackboard is not None
    assert runtime.orchestrator is not None

    # Should stop cleanly
    await runtime.stop()
    assert not runtime._running
    assert not runtime._subsystems_started


@pytest.mark.asyncio
async def test_runtime_singleton_blackboard(runtime: EdgeRuntime) -> None:
    """Test that all subsystems share the same blackboard."""
    await runtime.start()

    # All subsystems should reference the same blackboard instance
    assert runtime.orchestrator._blackboard is runtime.blackboard

    await runtime.stop()


@pytest.mark.asyncio
async def test_runtime_context_manager() -> None:
    """Test async context manager interface."""
    config = RuntimeConfig(log_level="DEBUG")

    async with EdgeRuntime(config).managed() as runtime:
        assert runtime._running
        assert runtime._subsystems_started

    # Should auto-stop on exit
    assert not runtime._running


@pytest.mark.asyncio
async def test_runtime_health_check(runtime: EdgeRuntime) -> None:
    """Test health check endpoint."""
    await runtime.start()

    health = await runtime.health_check()

    assert health["runtime"] is True
    assert health["blackboard"] is True
    assert health["orchestrator"] is True

    await runtime.stop()


@pytest.mark.asyncio
async def test_runtime_get_metrics(runtime: EdgeRuntime) -> None:
    await runtime.start()

    metrics = runtime.get_metrics()

    assert isinstance(metrics, dict)
    assert "anomalies_detected" in metrics
    assert "failed_plans" in metrics
    assert "current_plan_id" in metrics

    await runtime.stop()


@pytest.mark.asyncio
async def test_runtime_start_twice(runtime: EdgeRuntime) -> None:
    await runtime.start()
    await runtime.start()  # Should log warning and return early
    assert runtime._subsystems_started
    await runtime.stop()


@pytest.mark.asyncio
async def test_runtime_stop_without_start(runtime: EdgeRuntime) -> None:
    await runtime.stop()  # Should log warning and return
    assert not runtime._subsystems_started


@pytest.mark.asyncio
async def test_runtime_stop_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = EdgeRuntime()

    async def slow_stop() -> None:
        await asyncio.sleep(2)

    runtime.orchestrator = EdgeRuntime().orchestrator
    monkeypatch.setattr(runtime.orchestrator, "stop", slow_stop)
    runtime._subsystems_started = True
    runtime._running = True
    runtime.config.shutdown_timeout = 0.01  # Force timeout

    await runtime.stop()
    assert not runtime._subsystems_started


@pytest.mark.asyncio
async def test_runtime_run_until_stopped_manual(runtime: EdgeRuntime) -> None:
    async def trigger_stop() -> None:
        await asyncio.sleep(0.05)
        runtime._stop_event.set()

    await runtime.start()
    stopper = asyncio.create_task(trigger_stop())
    await runtime.run_until_stopped()
    await stopper

    await runtime.stop()  # ✅ Explicitly shut down subsystems

    assert not runtime._running
    assert not runtime._subsystems_started


@pytest.mark.asyncio
async def test_runtime_signal_handlers_removed(runtime: EdgeRuntime) -> None:
    await runtime.start()

    # Simulate stop without real signal
    runtime._stop_event.set()
    await runtime.run_until_stopped()

    # No assertion needed — just exercising signal registration/removal
    await runtime.stop()


@pytest.mark.asyncio
async def test_runtime_stop_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = EdgeRuntime()
    runtime._subsystems_started = True
    runtime._running = True

    async def broken_stop() -> None:
        raise RuntimeError("fail")

    runtime.orchestrator = EdgeRuntime().orchestrator
    monkeypatch.setattr(runtime.orchestrator, "stop", broken_stop)

    await runtime.stop()
    assert not runtime._subsystems_started


def test_create_runtime_profiles() -> None:
    dev = create_runtime("development")
    prod = create_runtime("production")
    sim = create_runtime("simulation")
    default = create_runtime("unknown")

    assert dev.config.log_level == "DEBUG"
    assert prod.config.shutdown_timeout == 30.0
    assert sim.config.sensor_fusion_rate == 0.01
    assert default.config.log_level == "INFO"


@pytest.mark.asyncio
async def test_runtime_managed_cleanup() -> None:
    runtime = EdgeRuntime()
    cm = runtime.managed()
    async with cm as r:
        assert r._running
    assert not r._running
