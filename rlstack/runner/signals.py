"""RunSignals: the awaitable half of the blackboard.

Daemons never call each other — they write the store and wait on predicates
OVER the store. This object is only wake-up plumbing: the store stays the
single source of truth (a predicate re-reads it on every check) and notify is a
latency hint, so the periodic timeout is what makes a writer in another process
visible too, just more slowly.
"""

from __future__ import annotations

import asyncio
from typing import Callable, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


async def store_work(work: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run blocking store work without holding the host's event loop.

    Cancellation drains the worker before custody can end: a successor must
    never race an old writer still appending a ledger or sweeping its files.
    This also applies to predicates that realize replay rows into the store.
    """
    pending = asyncio.create_task(asyncio.to_thread(work, *args, **kwargs))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                continue
        pending.result()
        raise


class RunSignals:
    def __init__(self, poll_seconds: float = 0.5) -> None:
        self._condition = asyncio.Condition()
        self._poll_seconds = poll_seconds

    async def notify(self) -> None:
        """Call after writing the store; wakes every waiting daemon."""
        async with self._condition:
            self._condition.notify_all()

    async def wait_for(self, predicate: Callable[[], T | None]) -> T:
        """Block until `predicate` (a read of the store) yields a truthy value,
        and return that value. Checked immediately, on every notify, and at
        least every poll interval."""
        while True:
            value = await store_work(predicate)
            if value:
                return value
            async with self._condition:
                try:
                    await asyncio.wait_for(self._condition.wait(),
                                           self._poll_seconds)
                except TimeoutError:
                    pass
