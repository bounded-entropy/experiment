"""The allocation operations a provider supplies to the shared fleet runtime."""

from __future__ import annotations

from typing import Protocol


class AllocationProvider(Protocol):
    """Boot named capacity and confirm termination of an exact allocation.

    Boot requests capacity; registration establishes its identity and readiness.
    Terminate returns True only after that allocation has ended. False or an
    exception leaves ownership unresolved in the shared Desk.
    """

    def boot(self, name: str) -> None: ...

    async def terminate(self, allocation_id: str) -> bool: ...
