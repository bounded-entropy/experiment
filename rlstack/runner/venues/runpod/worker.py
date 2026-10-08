"""The GPU process on a hand-started RunPod pod: measured shared metal over the volume.

The pod's start command runs this with the desk config and the slot's name
(ADR 0021). It reaches the desk at its private endpoint, opens the store at
the volume path every pod shares, and registers its own endpoint with the
pod id RunPod gives it. When the desk releases it, it exits; the pod stays
up, and bills, until a person stops it.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path

from rlstack.data.stores.local import LocalStore
from rlstack.runner.remote import RemoteDesk, transport_for
from rlstack.runner.venues.runpod.provider import DeskConfig, MetalConfig, desk_token
from rlstack.runner.venues.runtime import MetalRuntime
from rlstack.runner.venues.serving import EVERY_INTERFACE, health, serve_metal

DESK_WAIT_S = 60.0


@dataclass(frozen=True)
class WorkerConfig:
    """One slot of the desk config, on the pod that was started for it."""
    metal: MetalConfig
    store: str
    desk: str
    port: int
    pod_id: str
    host: str = EVERY_INTERFACE

    @classmethod
    def from_desk(cls, config: DeskConfig, name: str) -> WorkerConfig:
        """The slot named at pod start, and the pod's own id as its container:
        the id a person stops by hand, recorded beside the metal it served.
        It listens on the slot's own port when the slot names one."""
        pod_id = os.environ.get("RUNPOD_POD_ID", "")
        if not pod_id:
            raise RuntimeError("RUNPOD_POD_ID is unset: this is not a RunPod pod, "
                               "or set it by hand in the pod's environment")
        metal = config.metal(name)
        return cls(metal, config.store, config.desk_endpoint,
                   config.worker_port if metal.port is None else metal.port, pod_id)


async def run(config: WorkerConfig, token: str) -> None:
    """Refuse startup until this pod can reach the desk; then serve the shared metal.

    The slot's environment and the desk's token are installed in this
    process first, so every resident it spawns inherits them."""
    os.environ.update(config.metal.environment or {})
    os.environ["RLSTACK_HTTP_TOKEN"] = token
    deadline = asyncio.get_running_loop().time() + DESK_WAIT_S
    while True:
        try:
            if await asyncio.to_thread(health, config.desk, token=token, timeout=5):
                break
        except (OSError, ValueError):
            pass
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError(f"the desk at {config.desk} is unreachable: start the desk "
                               "pod first, with global networking on both pods")
        await asyncio.sleep(0.25)
    metal = config.metal
    runtime = MetalRuntime.measured(
        metal.name, LocalStore(config.store), RemoteDesk(transport_for(config.desk)),
        address=metal.address, host_address=metal.host_address,
        container=config.pod_id, idle_s=metal.idle_s, builds=metal.builds)
    await serve_metal(runtime, port=config.port, token=token, host=config.host,
                      ready={"pod_id": config.pod_id, "endpoint": metal.address,
                             "hf_home": os.environ.get("HF_HOME")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metal", required=True, help="the slot this pod was started for")
    args = parser.parse_args()
    config = DeskConfig.read(args.config)
    asyncio.run(run(WorkerConfig.from_desk(config, args.metal), desk_token(config)))


if __name__ == "__main__":
    main()
