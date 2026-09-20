"""Modal class lifecycle and RPC doors around the shared DeskRuntime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from rlstack.data.stores.base import Store
from rlstack.runner.venues.modal.provider import ModalProvider
from rlstack.runner.venues.runtime import DeskRuntime


def desk_class(app, image, *, module: str, store_for: Callable[[], Store],
               volumes: dict, bootable_metals: frozenset[str],
               idle_s: float = 90.0,
               idle_tick_s: float = 30.0, cls: str = "Desk"):
    """Expose the same standing duties as the local backend on Modal's loop."""
    import modal

    class ModalDesk:
        @modal.enter()
        async def bring_up(self) -> None:
            from rlstack.runner.remote import transport_for

            provider = ModalProvider(lambda name: self.runtime.desk.metal_addresses.get(name, ""))
            self.runtime = DeskRuntime(
                store_for(), transport_for=transport_for, provider=provider,
                bootable_metals=bootable_metals,
                idle_s=idle_s, restored_idle_s=300.0, idle_tick_s=idle_tick_s)
            self.desk = self.runtime.desk
            await self.runtime.__aenter__()

        @modal.method()
        async def door(self, host: str, verb: str, payload: dict) -> dict:
            return await self.runtime.service_for(host).serve(verb, payload)

        @modal.method()
        async def door_ask(self, host: str, verb: str, payload: dict) -> dict:
            service = self.runtime.service_for(host)
            return await asyncio.to_thread(service.answer, verb, payload)

        @modal.exit()
        async def bring_down(self) -> None:
            await self.runtime.__aexit__(None, None, None)

    ModalDesk.__name__ = ModalDesk.__qualname__ = cls
    ModalDesk.__module__ = module
    return app.cls(
        image=image, volumes=volumes, timeout=3600, min_containers=1,
        max_containers=1, scaledown_window=1200,
    )(modal.concurrent(max_inputs=32)(ModalDesk))
