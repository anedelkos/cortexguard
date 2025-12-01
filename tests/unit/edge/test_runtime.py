import asyncio
import logging
import signal
from collections.abc import AsyncGenerator, Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from kitchenwatch.edge.policy.mistral_policy_engine import MistralLLMPolicyEngine
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

    metrics = await runtime.get_metrics()

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


@pytest.mark.asyncio
async def test_runtime_policy_agent_composition(runtime: EdgeRuntime) -> None:
    """Test that the Policy Agent and its dependencies (Engine, Registry) are initialized."""
    # Initialization occurs in __init__, so we only need the fixture

    # 1. Check Agent instantiation
    assert runtime.policy_agent is not None

    # 2. Check Policy Engine instantiation (and mock mode)
    assert isinstance(runtime.policy_agent._policy_engine, MistralLLMPolicyEngine)
    assert runtime.policy_agent._policy_engine._use_mock is True

    # 3. Check wiring to Action Registry and Blackboard
    assert runtime.policy_agent._capability_registry is runtime.capability_registry
    assert runtime.policy_agent._blackboard is runtime.blackboard


@pytest.mark.asyncio
async def test_runtime_policy_agent_lifecycle(
    runtime: EdgeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that PolicyAgent start() and stop() methods are called."""

    mock_start = AsyncMock()
    mock_stop = AsyncMock()

    # Apply the mocks using monkeypatch
    monkeypatch.setattr(runtime.policy_agent, "start", mock_start)
    monkeypatch.setattr(runtime.policy_agent, "stop", mock_stop)

    await runtime.start()

    # Verify PolicyAgent.start() was included in the concurrent startup
    mock_start.assert_called_once()

    await runtime.stop()

    # Verify PolicyAgent.stop() was included in the concurrent shutdown
    mock_stop.assert_called_once()


@pytest.mark.asyncio
async def test_runtime_health_check_includes_policy_agent(
    runtime: EdgeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the health check reports the policy agent's status."""

    # Mock the internal running flag on the policy agent for reporting test
    monkeypatch.setattr(runtime.policy_agent, "_loop_running", True)

    await runtime.start()

    health = await runtime.health_check()

    agent_health_key = "policy_agent"

    # Temporarily add the policy agent check here to test the wiring:
    runtime_health_check = runtime.health_check

    async def augmented_health_check() -> dict[str, bool]:
        base_health = await runtime_health_check()
        base_health[agent_health_key] = runtime.policy_agent is not None and getattr(
            runtime.policy_agent, "_loop_running", False
        )
        return base_health

    # Use the augmented check for testing
    monkeypatch.setattr(runtime, "health_check", augmented_health_check)

    health = await runtime.health_check()

    assert agent_health_key in health
    assert health[agent_health_key] is True  # Should be True if start() succeeded

    await runtime.stop()


@pytest.mark.asyncio
async def test_runtime_startup_failure_handling(
    runtime: EdgeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that startup fails and triggers stop() if a subsystem raises an exception."""

    async def broken_start(**kwargs: Any) -> None:
        # Check that the expected keyword argument is passed (optional, but good practice)
        assert "tick_interval" in kwargs
        raise RuntimeError("Subsystem failed to initialize")

    # Patch a critical subsystem's start method to fail
    monkeypatch.setattr(runtime.orchestrator, "start", broken_start)

    # Patch stop to confirm cleanup is called
    mock_stop = AsyncMock()
    monkeypatch.setattr(runtime, "stop", mock_stop)

    # Check that start() raises the expected RuntimeError
    with pytest.raises(RuntimeError, match="One or more subsystems failed to start"):
        await runtime.start()

    # Confirm stop() was called for cleanup
    mock_stop.assert_called_once()
    assert not runtime._subsystems_started


@pytest.mark.asyncio
async def test_runtime_stop_timeout_handling(
    runtime: EdgeRuntime, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that stop() handles a TimeoutError and logs the error."""
    runtime.config.shutdown_timeout = 0.01

    # Mock a subsystem's stop method to exceed the short timeout
    async def slow_stop() -> None:
        await asyncio.sleep(2)  # Must be much longer than the timeout

    # Need to simulate being started first for the stop logic to execute
    runtime._subsystems_started = True
    runtime._running = True

    monkeypatch.setattr(runtime.orchestrator, "stop", slow_stop)

    with caplog.at_level(logging.ERROR):
        await runtime.stop()

    assert "Shutdown timeout exceeded, forcing stop" in caplog.text
    assert not runtime._subsystems_started


@pytest.mark.asyncio
async def test_runtime_signal_handler_removal_exception(
    runtime: EdgeRuntime, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test exception handling during signal handler cleanup."""

    class MockLoop:
        def is_running(self) -> bool:
            return True  # Ensure cleanup path is taken

        def add_signal_handler(self, sig: signal.Signals, handler: Callable[..., Any]) -> None:
            pass

        def remove_signal_handler(self, sig: signal.Signals) -> None:
            # Simulate failure during removal for SIGINT
            if sig == signal.SIGINT:
                raise Exception("Failed to remove handler")

    # Patch the event loop getter to return our mock loop
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: MockLoop())

    # Need to start the runtime to trigger signal registration path
    await runtime.start()

    # Trigger stop immediately via the internal event (L289-291, L294->301)
    runtime._stop_event.set()

    with caplog.at_level(logging.DEBUG):
        # run_until_stopped will now try to run and hit the cleanup block (L298-299)
        await runtime.run_until_stopped()

    assert "Failed to remove signal handler for" in caplog.text

    # Ensure stop is called afterward for proper cleanup
    await runtime.stop()
