"""The desk pod's services for RunPod; placement stays in the shared Desk."""
from __future__ import annotations

import asyncio
import os
import signal

from rlstack.data.stores.local import LocalStore
from rlstack.runner.remote import RemoteDesk, transport_for
from rlstack.runner.venues.runpod.provider import DeskConfig, RunPodProvider, desk_token
from rlstack.runner.venues.runtime import DeskRuntime
from rlstack.runner.venues.serving import (
    EVERY_INTERFACE, observer, own_local_desk, runtime_token,
)


async def serve(config: DeskConfig, *, host: str = EVERY_INTERFACE,
                stop: asyncio.Event | None = None) -> None:
    """One journal owner on the volume, its door and observer, finite idle release.

    The door is the shared Desk's own plane, served as is: no gateway (every
    pod dials every other at its private endpoint) and no acquisition verb
    (pods are started by hand). No metal is bootable, so a knock is refused
    and journaled; automatic recovery is off, because it waits for a
    provider-confirmed termination this venue never gives (ADR 0021, Q5).
    `host` is every interface on the desk pod, where workers reach it over
    the private network and the operator through the proxy."""
    from rlstack.runner.transports.http import HttpServer

    with own_local_desk(config.state_dir / "desk.lock"):
        token = runtime_token(config.state_dir)
        os.environ["RLSTACK_HTTP_TOKEN"] = token
        runtime = DeskRuntime(
            LocalStore(config.store), transport_for=transport_for, provider=RunPodProvider(),
            bootable_metals=frozenset(), idle_s=config.idle_s,
            host_idle_s=config.host_idle_s, auto_recover=False)
        if stop is None:
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for number in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(number, stop.set)
        async with runtime:
            async with HttpServer(runtime.service_for, host=host, port=config.desk_port,
                                  token=token):
                with observer(LocalStore(config.store), config.observer_port,
                              host=host, token=token):
                    print(f"desk: http://{host}:{config.desk_port}; "
                          f"observer: http://{host}:{config.observer_port}", flush=True)
                    await stop.wait()
        # A CPU process stopping says nothing about the GPU pods: they keep
        # their workers, and they bill until a person stops them.
        print("desk stopped; RunPod pods are not stopped by rlstack: "
              "stop idle ones by hand", flush=True)


def desk(config: DeskConfig) -> RemoteDesk:
    """The operator's handle: the desk at its operator door, with the token
    every http transport reads from the environment."""
    os.environ["RLSTACK_HTTP_TOKEN"] = desk_token(config)
    return RemoteDesk(transport_for(config.operator_door))
