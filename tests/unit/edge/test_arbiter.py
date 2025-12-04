import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from kitchenwatch.core.interfaces.base_controller import BaseController
from kitchenwatch.edge.arbiter import Arbiter
from kitchenwatch.edge.models.agent_tool_call import AgentToolCall
from kitchenwatch.edge.models.blackboard import Blackboard
from kitchenwatch.edge.models.capability_registry import CapabilityRegistry, RiskLevel


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
    cast(Any, bb).set_safety_flag.assert_awaited_with("emergency_stop", True)
    cast(Any, bb).add_trace_entry.assert_awaited()
