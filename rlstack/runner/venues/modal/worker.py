"""Modal decorators adapt one shared MetalRuntime to a persistent GPU class."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import os
import time

from rlstack.data.stores.base import Store
from rlstack.runner.residents import Builds
from rlstack.runner.venues.modal.provider import host_address, metal_address
from rlstack.runner.venues.runtime import MetalRuntime


def metal_class(app, app_name: str, metal: str, gpu, image, *, module: str,
                store_for: Callable[[], Store], desk_address: str, volumes: dict,
                idle_s: float = 1800.0, recipe: Builds | None = None,
                cls: str = "MetalS", secrets=(), max_containers: int = 1,
                heartbeat_s: float = 20.0, stop_fetching: Callable[[], None] | None = None):
    """Retain Modal identities and resources while sharing the worker lifecycle."""
    import modal

    class MetalS:
        @modal.enter()
        async def bring_up(self) -> None:
            from rlstack.runner.remote import RemoteDesk, transport_for

            self.born = time.time()
            self.runtime = MetalRuntime.measured(
                metal, store_for(), RemoteDesk(transport_for(desk_address)),
                address=metal_address(app_name, cls),
                host_address=lambda host, epoch: host_address(app_name, host, cls, epoch),
                container=os.environ.get("MODAL_TASK_ID", ""), idle_s=idle_s,
                builds=recipe, heartbeat_s=heartbeat_s)
            self.metal_service = self.runtime.service
            await self.runtime.__aenter__()

        @modal.method()
        async def door(self, host: str, verb: str, payload: dict) -> dict:
            return await self.runtime.service_for(host).serve(verb, payload)

        @modal.method()
        async def door_ask(self, host: str, verb: str, payload: dict) -> dict:
            service = self.runtime.service_for(host)
            return await asyncio.to_thread(service.answer, verb, payload)

        @modal.method()
        async def serve(self) -> dict:
            await self.runtime.wait()
            if stop_fetching is not None:
                stop_fetching()
            return {"released": True, "metal": metal,
                    "shift_s": round(time.time() - self.born, 1)}

        @modal.exit()
        async def bring_down(self) -> None:
            await self.runtime.__aexit__(None, None, None)

    MetalS.__name__ = MetalS.__qualname__ = cls
    MetalS.__module__ = module
    return app.cls(
        image=image, gpu=gpu, volumes=volumes, secrets=list(secrets),
        timeout=86400, scaledown_window=int(idle_s), max_containers=max_containers,
    )(modal.concurrent(max_inputs=64)(MetalS))
