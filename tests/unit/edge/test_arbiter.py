import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from cortexguard.core.interfaces.base_controller import BaseController
from cortexguard.edge.arbiter import Arbiter
from cortexguard.edge.models.agent_tool_call import AgentToolCall
from cortexguard.edge.models.blackboard import Blackboard
from cortexguard.edge.models.capability_registry import CapabilityRegistry, RiskLevel


@pytest.mark.asyncio
async def test_request_action_authorized_executes_and_sets_flag():
    # Arrange
    bb = Blackboard()
    # cast to Any so mypy won't complain about assigning to methods
    cast(Any, bb).set_safety_flag = AsyncMock()
    cast(Any, bb).add_trace_entry = AsyncMock()

    # CapabilityRegistry stub: schema exists and validation returns (True, LOW)
    cap = MagicMock(spec=CapabilityRegistry)
    cap.get_function_schema.return_value = {"parameters": {"type": "object"}}
    cap.validate_call.return_value = (True, RiskLevel.LOW)

    # Controller mock with execute method
    controller = AsyncMock(spec=BaseController)
    controller.execute = AsyncMock()
    controller.emergency_stop = AsyncMock()

    arbiter = Arbiter(blackboard=bb, capability_registry=cap, controller=controller)

    action = AgentToolCall(action_name="do_thing", arguments={"x": 1})

    # Act
    result = await arbiter.request_action("tester", action, reason="unit-test")

    # Allow scheduled tasks (add_trace_entry, set_safety_flag) to run
    await asyncio.sleep(0)

    # Assert
    assert result is True
    cap.get_function_schema.assert_called_once_with("do_thing")
    cap.validate_call.assert_called_once_with("do_thing", {"x": 1})
    controller.execute.assert_awaited_once_with("do_thing", {"x": 1})
    cast(Any, bb).set_safety_flag.assert_awaited()  # at least one boolean flag set
    cast(Any, bb).add_trace_entry.assert_awaited()  # trace entry scheduled


@pytest.mark.asyncio
async def test_request_action_unknown_capability_denied():
    # Arrange
    bb = Blackboard()
    cast(Any, bb).set_safety_flag = AsyncMock()
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)
    cap.get_function_schema.side_effect = KeyError("not found")

    controller = AsyncMock(spec=BaseController)
    controller.execute = AsyncMock()
    controller.emergency_stop = AsyncMock()

    arbiter = Arbiter(blackboard=bb, capability_registry=cap, controller=controller)

    action = AgentToolCall(action_name="unknown_fn", arguments={})

    # Act
    result = await arbiter.request_action("tester", action)

    # Allow scheduled tasks to run
    await asyncio.sleep(0)

    # Assert
    assert result is False
    cap.get_function_schema.assert_called_once_with("unknown_fn")
    controller.execute.assert_not_awaited()
    cast(Any, bb).add_trace_entry.assert_awaited()
    cast(Any, bb).set_safety_flag.assert_awaited()


@pytest.mark.asyncio
async def test_request_action_validation_failure_denied():
    # Arrange
    bb = Blackboard()
    cast(Any, bb).set_safety_flag = AsyncMock()
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)
    cap.get_function_schema.return_value = {"parameters": {"type": "object"}}
    # Simulate validation returning invalid + HIGH risk
    cap.validate_call.return_value = (False, RiskLevel.HIGH)

    controller = AsyncMock(spec=BaseController)
    controller.execute = AsyncMock()
    controller.emergency_stop = AsyncMock()

    arbiter = Arbiter(blackboard=bb, capability_registry=cap, controller=controller)

    action = AgentToolCall(action_name="dangerous", arguments={"bad": True})

    # Act
    result = await arbiter.request_action("tester", action)

    # Allow scheduled tasks to run
    await asyncio.sleep(0)

    # Assert
    assert result is False
    cap.validate_call.assert_called_once_with("dangerous", {"bad": True})
    controller.execute.assert_not_awaited()
    cast(Any, bb).add_trace_entry.assert_awaited()
    cast(Any, bb).set_safety_flag.assert_awaited()


@pytest.mark.asyncio
async def test_emergency_stop_calls_controller_and_sets_flag():
    # Arrange
    bb = Blackboard()
    cast(Any, bb).set_safety_flag = AsyncMock()
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)

    controller = AsyncMock(spec=BaseController)
    controller.emergency_stop = AsyncMock()

    arbiter = Arbiter(blackboard=bb, capability_registry=cap, controller=controller)

    # Act
    await arbiter.emergency_stop("overtemp")

    # Allow scheduled tasks to run
    await asyncio.sleep(0)

    # Assert
    controller.emergency_stop.assert_awaited_once()
    cast(Any, bb).set_safety_flag.assert_any_await("emergency_stop", True)
    cast(Any, bb).add_trace_entry.assert_awaited()


@pytest.mark.asyncio
async def test_request_action_validation_raises_exception():
    bb = Blackboard()
    cast(Any, bb).set_safety_flag = AsyncMock()
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)
    cap.get_function_schema.return_value = {"parameters": {"type": "object"}}
    cap.validate_call.side_effect = ValueError("bad args")

    controller = AsyncMock(spec=BaseController)
    controller.execute = AsyncMock()

    arbiter = Arbiter(bb, cap, controller)
    action = AgentToolCall(action_name="fn", arguments={})

    result = await arbiter.request_action("tester", action)
    await asyncio.sleep(0)

    assert result is False
    controller.execute.assert_not_awaited()
    cast(Any, bb).add_trace_entry.assert_awaited()


@pytest.mark.asyncio
async def test_request_action_validation_returns_none():
    bb = Blackboard()
    cast(Any, bb).set_safety_flag = AsyncMock()
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)
    cap.get_function_schema.return_value = {"parameters": {"type": "object"}}
    cap.validate_call.return_value = None

    controller = AsyncMock(spec=BaseController)

    arbiter = Arbiter(bb, cap, controller)
    action = AgentToolCall(action_name="fn", arguments={})

    result = await arbiter.request_action("tester", action)
    await asyncio.sleep(0)

    assert result is False
    controller.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_action_high_risk_denied():
    bb = Blackboard()
    cast(Any, bb).set_safety_flag = AsyncMock()
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)
    cap.get_function_schema.return_value = {"parameters": {"type": "object"}}
    cap.validate_call.return_value = (True, RiskLevel.HIGH)

    controller = AsyncMock(spec=BaseController)

    arbiter = Arbiter(bb, cap, controller)
    action = AgentToolCall(action_name="fn", arguments={})

    result = await arbiter.request_action("tester", action)
    await asyncio.sleep(0)

    assert result is False
    controller.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_action_controller_execution_fails():
    bb = Blackboard()
    cast(Any, bb).set_safety_flag = AsyncMock()
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)
    cap.get_function_schema.return_value = {"parameters": {"type": "object"}}
    cap.validate_call.return_value = (True, RiskLevel.LOW)

    controller = AsyncMock(spec=BaseController)
    controller.execute.side_effect = RuntimeError("boom")

    arbiter = Arbiter(bb, cap, controller)
    action = AgentToolCall(action_name="fn", arguments={})

    result = await arbiter.request_action("tester", action)
    await asyncio.sleep(0)

    assert result is False
    controller.execute.assert_awaited()


@pytest.mark.asyncio
async def test_emergency_stop_falls_back_to_execute():
    bb = Blackboard()
    cast(Any, bb).set_safety_flag = AsyncMock()
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)

    controller = AsyncMock(spec=BaseController)
    del controller.emergency_stop  # remove emergency_stop
    controller.execute = AsyncMock()

    arbiter = Arbiter(bb, cap, controller)

    await arbiter.emergency_stop("reason")
    await asyncio.sleep(0)

    controller.execute.assert_awaited_once_with("EMERGENCY_STOP", {})


# ---------------------------------------------------------------------------
# C1 regression tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_action_internal_error_does_not_set_emergency_stop_flag():
    """
    C1 regression: _append_and_publish must not set emergency_stop for internal
    request_action errors. Only an explicit emergency_stop() call should set that flag.

    A non-KeyError from get_function_schema escapes the inner except clause and
    hits the outer CRITICAL handler, which previously triggered fire-and-forget
    emergency_stop via _append_and_publish.
    """
    bb = Blackboard()
    emergency_stop_flagged = False

    async def track_flags(name: str, value: bool) -> None:
        nonlocal emergency_stop_flagged
        if name == "emergency_stop" and value:
            emergency_stop_flagged = True

    cast(Any, bb).set_safety_flag = track_flags
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)
    # RuntimeError (not KeyError) escapes the inner except and hits the CRITICAL path
    cap.get_function_schema.side_effect = RuntimeError("unexpected registry error")

    controller = AsyncMock(spec=BaseController)
    arbiter = Arbiter(bb, cap, controller)

    result = await arbiter.request_action("tester", AgentToolCall(action_name="fn", arguments={}))
    await asyncio.sleep(0)  # let any fire-and-forget tasks run

    assert result is False
    assert not emergency_stop_flagged, (
        "Internal request_action errors must not set the emergency_stop flag; "
        "only an explicit emergency_stop() call should do that."
    )


@pytest.mark.asyncio
async def test_emergency_stop_flag_set_exactly_once():
    """
    C1 regression: emergency_stop must set the safety flag exactly once via
    direct await. With create_task in _append_and_publish (fire-and-forget) AND
    in emergency_stop itself, the flag was set twice — once unguaranteed and once
    awaited. After the fix, exactly one direct await sets it.
    """
    bb = Blackboard()
    call_count = 0

    async def count_flag_calls(name: str, value: bool) -> None:
        nonlocal call_count
        if name == "emergency_stop" and value:
            call_count += 1

    cast(Any, bb).set_safety_flag = count_flag_calls
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)
    controller = AsyncMock(spec=BaseController)
    controller.emergency_stop = AsyncMock()

    arbiter = Arbiter(bb, cap, controller)
    await arbiter.emergency_stop("overtemp")
    await asyncio.sleep(0)  # flush any residual scheduled tasks

    assert call_count == 1, (
        f"emergency_stop flag set {call_count} times; expected exactly 1. "
        "Likely cause: fire-and-forget create_task in _append_and_publish duplicating the write."
    )


# ---------------------------------------------------------------------------
# M6 regression test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emergency_stop_not_blocked_by_slow_controller_in_request_action() -> None:
    """
    M6 regression: request_action holds _lock for the entire duration of
    controller.execute(), including slow/hanging hardware commands. emergency_stop
    also needs _lock, so it is blocked until the slow command finishes — preventing
    the E-STOP signal from being set.

    Bug:  emergency_stop cannot acquire _lock while request_action is executing
          a slow controller command; the E-STOP flag is delayed.
    Fix:  controller.execute() runs outside the lock; only validation and audit
          append are held under it.
    """
    controller_entered = asyncio.Event()
    allow_complete = asyncio.Event()

    class SlowController(BaseController):
        async def execute(self, name: str, args: dict[str, Any]) -> None:
            controller_entered.set()
            await allow_complete.wait()  # simulates a hanging hardware command

        async def emergency_stop(self) -> None:
            pass

    bb = Blackboard()
    cast(Any, bb).set_safety_flag = AsyncMock()
    cast(Any, bb).add_trace_entry = AsyncMock()

    cap = MagicMock(spec=CapabilityRegistry)
    cap.get_function_schema.return_value = {"parameters": {"type": "object"}}
    cap.validate_call.return_value = (True, RiskLevel.LOW)

    arbiter = Arbiter(blackboard=bb, capability_registry=cap, controller=SlowController())

    action = AgentToolCall(action_name="slow_move", arguments={})

    # Start request_action — it will block inside the slow controller while holding the lock
    request_task = asyncio.create_task(arbiter.request_action("tester", action))

    # Wait until the controller has actually started executing (lock held)
    await controller_entered.wait()

    # emergency_stop should complete without waiting for the slow controller to finish
    estop_completed = asyncio.Event()

    async def do_estop() -> None:
        await arbiter.emergency_stop("safety test")
        estop_completed.set()

    estop_task = asyncio.create_task(do_estop())

    # Yield to let estop attempt to run
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Bug: estop is still blocked waiting for the lock held by request_action
    assert estop_completed.is_set(), (
        "emergency_stop was blocked by the lock held during controller.execute(). "
        "E-STOP must not be gated behind slow hardware commands."
    )

    # Clean up
    allow_complete.set()
    await request_task
    await estop_task
