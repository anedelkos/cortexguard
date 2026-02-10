from __future__ import annotations

import asyncio
import heapq
from datetime import datetime
from typing import TypeVar

T = TypeVar("T")


class AsyncPriorityQueue[T]:
    """
    Async-safe priority queue with awaitable get().

    This provides both asyncio-safe concurrency and
    true priority-based ordering using a heap.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, datetime, T]] = []
        self._cv = asyncio.Condition()

    async def put(self, priority: int, item: T) -> None:
        async with self._cv:
            heapq.heappush(self._heap, (priority, datetime.now(), item))
            self._cv.notify()

    async def pop(self, block: bool = True, timeout: float | None = None) -> T | None:
        async with self._cv:
            if not block:
                if not self._heap:
                    return None
                _, _, item = heapq.heappop(self._heap)
                return item

            if timeout is not None:
                try:
                    await asyncio.wait_for(self._cv.wait(), timeout)
                except TimeoutError:
                    return None

            while not self._heap:
                await self._cv.wait()
            _, _, item = heapq.heappop(self._heap)
            return item

    async def pop_if_priority_lower_than(self, current_priority: int) -> T | None:
        async with self._cv:
            if not self._heap:
                return None

            priority, _, item = self._heap[0]
            if priority < current_priority:
                heapq.heappop(self._heap)
                return item

            return None

    async def empty(self) -> bool:
        async with self._cv:
            return len(self._heap) == 0

    async def peek(self) -> T | None:
        """Look at the highest-priority item without removing it."""
        async with self._cv:
            if not self._heap:
                return None
            _, _, item = self._heap[0]
            return item

    async def get_all_items(self) -> list[T]:
        """Return a snapshot of all items in priority order (without popping)."""
        async with self._cv:
            return [item for _, _, item in sorted(self._heap)]

    def __len__(self) -> int:
        return len(self._heap)
