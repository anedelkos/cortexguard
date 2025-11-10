import asyncio
from collections.abc import Generator

import pytest

from kitchenwatch.edge.utils.async_priority_queue import AsyncPriorityQueue


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a clean event loop for each test file."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def priority_queue() -> AsyncPriorityQueue[str]:
    """Provide a fresh AsyncPriorityQueue for each test."""
    return AsyncPriorityQueue[str]()


@pytest.mark.asyncio
async def test_put_and_get_single_item(priority_queue: AsyncPriorityQueue[str]) -> None:
    await priority_queue.put(5, "low")
    result: str = await priority_queue.pop()
    assert result == "low"
    assert await priority_queue.empty()


@pytest.mark.asyncio
async def test_priority_ordering(priority_queue: AsyncPriorityQueue[str]) -> None:
    await priority_queue.put(10, "low")
    await priority_queue.put(1, "high")
    await priority_queue.put(5, "mid")

    results: list[str] = [await priority_queue.pop() for _ in range(3)]
    assert results == ["high", "mid", "low"]


@pytest.mark.asyncio
async def test_concurrent_put_and_get(priority_queue: AsyncPriorityQueue[str]) -> None:
    async def producer() -> None:
        for i in range(3):
            await asyncio.sleep(0.01)
            await priority_queue.put(i, f"item-{i}")

    async def consumer() -> list[str]:
        results: list[str] = []
        for _ in range(3):
            results.append(await priority_queue.pop())
        return results

    producer_task: asyncio.Task[None] = asyncio.create_task(producer())
    consumer_task: asyncio.Task[list[str]] = asyncio.create_task(consumer())

    results: list[str] = await consumer_task
    await producer_task

    assert results == ["item-0", "item-1", "item-2"]


@pytest.mark.asyncio
async def test_empty_queue_waits(priority_queue: AsyncPriorityQueue[str]) -> None:
    async def delayed_put() -> None:
        await asyncio.sleep(0.05)
        await priority_queue.put(1, "late-item")

    put_task: asyncio.Task[None] = asyncio.create_task(delayed_put())
    get_task: asyncio.Task[str] = asyncio.create_task(priority_queue.pop())

    result: str = await get_task
    await put_task

    assert result == "late-item"


@pytest.mark.asyncio
async def test_multiple_priorities_with_same_value(priority_queue: AsyncPriorityQueue[str]) -> None:
    await priority_queue.put(1, "first")
    await asyncio.sleep(0.001)
    await priority_queue.put(1, "second")

    results: list[str] = [await priority_queue.pop() for _ in range(2)]
    assert results == ["first", "second"]


@pytest.mark.asyncio
async def test_len_and_empty(priority_queue: AsyncPriorityQueue[str]) -> None:
    assert await priority_queue.empty()
    await priority_queue.put(2, "a")
    await priority_queue.put(1, "b")

    assert not await priority_queue.empty()
    assert len(priority_queue) == 2

    await priority_queue.pop()
    await priority_queue.pop()

    assert await priority_queue.empty()
    assert len(priority_queue) == 0
