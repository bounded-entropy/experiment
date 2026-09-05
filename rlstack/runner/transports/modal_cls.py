"""The Modal transport: dict frames to one deployed Modal class's door.

ONE class serves every Modal address this fleet speaks, because every rlstack
Modal container wears the SAME two methods — `door` and `door_ask`, each ASYNC
since ADR 0008 (Q4) and each taking (host, verb, payload). A desk container
answers with host "" and so does a metal's own plane; a host inside a metal
container is named by the address's `#host` fragment. What differs between
venues is the app and the class, and both are IN the address (ADR 0007, Q3),
which is what lets one desk command metal in many apps.

WHY BOTH DOORS ARE ASYNC AND BOTH CALLS ARE BOUNDED. A cancelled input of a
SYNCHRONOUS method on a concurrent container has no clean interruption, so
Modal shuts the container down — the desk died that way three times on
2026-09-04, once to a harness kill and twice to a human's own timed probe. An
async method is one task the loop drops. And a wait with no bound is how one
unreachable metal wedges every submit behind it, so the deadline is the
CALLER's own and `Unreachable` is what an expired one raises (F3).
"""

from __future__ import annotations

import asyncio

import modal

from rlstack.runner.remote import DEADLINE_S, bounded, stamped


class ModalClsTransport:
    """Frames to one deployed Modal class, optionally addressed to one host
    inside it. `app`/`cls` name the deployed container; `host` is the name a
    metal container routes by, and None is that container's own plane;
    `epoch` is the INSTANCE the frames are addressed to (ADR 0008, F2) —
    stamped into every payload, refused by a container wearing another one,
    and empty for the frames that cannot name one yet (a registration)."""

    def __init__(self, app: str, cls: str, host: str | None = None,
                 epoch: str = "") -> None:
        self.app = app
        self.cls = cls
        self.host = host or ""
        self.epoch = epoch
        self._handle = None

    async def handle(self):
        """The deployed class's handle, resolved on first use and kept.

        Lazily, because a transport is constructed wherever an address is
        read — inside a desk rebuilding from its journal, for instance — and
        resolving a name is a network act that must not happen there. On a
        THREAD, because that resolution is blocking and this coroutine runs
        on a loop with other frames in flight."""
        if self._handle is None:
            self._handle = await asyncio.to_thread(
                lambda: modal.Cls.from_name(self.app, self.cls)())
        return self._handle

    def what(self, verb: str) -> str:
        """How a refusal names this transport's other end."""
        return f"modal://{self.app}/{self.cls}#{self.host}::{verb}"

    async def call(self, verb: str, payload: dict, *,
                   deadline_s: float = DEADLINE_S) -> dict:
        """An admitted verb, awaited on the caller's own loop under its own
        deadline."""
        return await bounded(self.frame("door", verb, payload), deadline_s,
                             self.what(verb))

    async def ask(self, verb: str, payload: dict, *,
                  deadline_s: float = DEADLINE_S) -> dict:
        """An admission-free verb — async since ADR 0008, because `door_ask`
        is an async method now and there is nothing left to put on a thread."""
        return await bounded(self.frame("door_ask", verb, payload), deadline_s,
                             self.what(verb))

    async def frame(self, door: str, verb: str, payload: dict) -> dict:
        """One frame to one of the container's two doors — the only place
        this module knows their names."""
        handle = await self.handle()
        return await getattr(handle, door).remote.aio(
            self.host, verb, stamped(payload, self.epoch))
