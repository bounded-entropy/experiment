"""RunSignals: the await half of the blackboard.

Roles never call each other — they write the store, and they wait on
predicates OVER the store. This object is only the wake-up plumbing: the
store stays the single source of truth (a predicate re-reads it on every
check), and the periodic timeout means a writer outside this process (the
Phase-C multi-process future) is noticed too, just more slowly.
"""

from __future__ import annotations

import asyncio
from typing import Callable, TypeVar

T = TypeVar("T")


class RunSignals:
    def __init__(self, poll_seconds: float = 0.5) -> None:
        self._condition = asyncio.Condition()
        self._poll_seconds = poll_seconds

    async def notify(self) -> None:
        """Call after writing the store; wakes every waiting role."""
        async with self._condition:
            self._condition.notify_all()

    async def wait_for(self, predicate: Callable[[], T | None]) -> T:
        """Block until `predicate` (a read of the store) yields a truthy value,
        and return that value. Checked immediately, on every notify, and at
        least every poll interval."""
        while True:
            value = predicate()
            if value:
                return value
            async with self._condition:
                try:
                    await asyncio.wait_for(self._condition.wait(),
                                           self._poll_seconds)
                except TimeoutError:
                    pass
