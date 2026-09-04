"""The Modal transport: dict frames to one deployed Modal class's door.

ONE class serves every Modal address this fleet speaks, because every rlstack
Modal container wears the SAME two methods — `door` (async, the admitted
verbs) and `door_ask` (sync, the admission-free ones), each taking
(host, verb, payload). A desk container answers with host "" and so does a
metal's own plane; a host inside a metal container is named by the address's
`#host` fragment. What differs between venues is the app and the class, and
both are IN the address (ADR 0007, Q3), which is what lets one desk command
metal in many apps.
"""

from __future__ import annotations

import concurrent.futures

import modal


def blocking_ask(fn):
    """One BLOCKING Modal call on its OWN thread.

    The admission-free half of the Transport contract is synchronous by
    design (registration and build facts, callable from sync call sites), and
    a blocking Modal portal call made from a thread that is running an event
    loop wedges that loop — an hour of silence on the venue, #77. So the call
    is handed to a thread that owns nothing.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as one:
        return one.submit(fn).result()


class ModalClsTransport:
    """Frames to one deployed Modal class, optionally addressed to one host
    inside it. `app`/`cls` name the deployed container; `host` is the name a
    metal container routes by, and None is that container's own plane."""

    def __init__(self, app: str, cls: str, host: str | None = None) -> None:
        self.app = app
        self.cls = cls
        self.host = host or ""
        self._handle = None

    def handle(self):
        """The deployed class's handle, resolved on first use and kept.

        Lazily, because a transport is constructed wherever an address is
        read — inside a desk rebuilding from its journal, for instance — and
        resolving a name is a network act that must not happen there."""
        if self._handle is None:
            self._handle = modal.Cls.from_name(self.app, self.cls)()
        return self._handle

    async def call(self, verb: str, payload: dict) -> dict:
        """An admitted verb: awaited on the caller's own loop."""
        return await self.handle().door.remote.aio(self.host, verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        """An admission-free verb: blocking, on its own thread (#77)."""
        return blocking_ask(
            lambda: self.handle().door_ask.remote(self.host, verb, payload))
