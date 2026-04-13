from __future__ import annotations

import asyncio
import heapq
from typing import TypeVar

T = TypeVar("T")


class AsyncPriorityQueue[T]:
    """
    Async-safe priority queue with awaitable pop().

    Items with lower priority numbers are popped first (min-heap).
    Items with equal priority are popped in FIFO order.

    This queue is unbounded.

    Example:
        >> queue = AsyncPriorityQueue[str]()
        >> await queue.put(5, "low priority")
        >> await queue.put(1, "high priority")
        >> await queue.pop()  # Returns "high priority"
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, T]] = []
        self._counter = 0
        self._cv = asyncio.Condition()

    async def put(self, priority: int, item: T) -> None:
        """
        Add an item to the queue with given priority.

        Args:
            priority: Priority value (lower = higher priority)
            item: Item to enqueue

        Note:
            Notifies exactly one waiting consumer (if any).
        """
        async with self._cv:
            heapq.heappush(self._heap, (priority, self._counter, item))
            self._counter += 1
            self._cv.notify()

    async def pop(self, block: bool = True, timeout: float | None = None) -> T | None:
        async with self._cv:
            if not block:
                if not self._heap:
                    return None
                _, _, item = heapq.heappop(self._heap)
                return item

            if timeout is None:
                # Wait indefinitely until an item arrives
                while not self._heap:
                    await self._cv.wait()
            else:
                # Wait with timeout, accounting for spurious wakeups
                loop = asyncio.get_running_loop()
                end = loop.time() + timeout

                while not self._heap:
                    remaining = end - loop.time()
                    if remaining <= 0:
                        return None
                    try:
                        await asyncio.wait_for(self._cv.wait(), timeout=remaining)
                    except TimeoutError:
                        return None

            _, _, item = heapq.heappop(self._heap)
            return item

    async def pop_if_priority_lower_than(self, current_priority: int) -> T | None:
        """
        Pop the top item only if its priority is numerically lower than threshold.

        Used for preemption: if a higher-priority task arrives, pop it.

        Args:
            current_priority: Priority threshold

        Returns:
            Item if popped, None if queue empty or top item priority >= threshold

        Example:
            >> # Current task has priority 5
            >> await queue.put(3, "urgent task")  # Higher priority
            >> urgent = await queue.pop_if_priority_lower_than(5)  # Returns "urgent task"
        """

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
        """
        Return a snapshot of all items in priority order.

        WARNING: O(n log n) operation. Use sparingly for debugging/monitoring.
        Does not modify the queue.
        """
        async with self._cv:
            return [item for _, _, item in sorted(self._heap)]

    def __len__(self) -> int:
        """
        Return queue size (NOT async-safe).

        WARNING: May return stale value due to race conditions.
        For control flow decisions, use the async size() method.
        Provided for quick monitoring/debugging only.
        """
        return len(self._heap)

    async def size(self) -> int:
        """Return the number of items in the queue (async-safe)."""
        async with self._cv:
            return len(self._heap)
