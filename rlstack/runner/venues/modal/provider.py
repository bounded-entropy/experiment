"""Modal allocation and addressing; fleet decisions belong to the shared Desk."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from rlstack.runner.remote import parse_address


def metal_address(app_name: str, cls: str = "MetalS") -> str:
    return f"modal://{app_name}/{cls}"


def host_address(app_name: str, host: str, cls: str = "MetalS", epoch: str = "") -> str:
    return metal_address(app_name, cls) + f"#{host}" + (f"@{epoch}" if epoch else "")


def metal_handle(app_name: str, cls: str = "MetalS"):
    """Look up deployed capacity without creating another deployment."""
    import modal

    return modal.Cls.from_name(app_name, cls)()


def boot_by_spawn(app_name: str, cls: str = "MetalS"):
    """A deployment's explicit acquisition callback; registration proves ready."""
    def boot(name: str):
        call = metal_handle(app_name, cls).serve.spawn()
        print(f"[knock] {name}: spawned keepalive {call.object_id}", flush=True)
        return call.object_id
    return boot


async def terminate_container(container_id: str) -> bool:
    """Confirm that the exact Modal container ended before ownership transfers."""
    from modal.client import _Client
    from modal_proto import api_pb2

    client = await _Client.from_env()

    async def finished() -> bool:
        async with asyncio.timeout(10.0):
            info = await client.stub.TaskGetInfo(
                api_pb2.TaskGetInfoRequest(task_id=container_id))
        return bool(info.info.finished_at)

    if await finished():
        return True
    try:
        async with asyncio.timeout(30.0):
            await client.stub.ContainerStop(
                api_pb2.ContainerStopRequest(task_id=container_id, graceful=False))
    except Exception as error:
        print(f"[desk] stop response uncertain for {container_id}: {error}", flush=True)
    for attempt in range(15):
        try:
            if await finished():
                print(f"[desk] confirmed container {container_id} ended", flush=True)
                return True
        except Exception:
            pass
        if attempt < 14:
            await asyncio.sleep(1.0)
    return False


class ModalProvider:
    """Implement AllocationProvider against deployment addresses in the journal."""

    def __init__(self, address_for: Callable[[str], str]) -> None:
        self.address_for = address_for

    def boot(self, name: str) -> None:
        parsed = parse_address(self.address_for(name))
        if parsed.scheme != "modal":
            raise ValueError(f"metal {name!r} has no Modal deployment address")
        boot_by_spawn(parsed.app, parsed.cls)(name)

    async def terminate(self, allocation_id: str) -> bool:
        return await terminate_container(allocation_id)
