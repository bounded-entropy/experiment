"""The GPU process: measured shared metal, authoritative scratch and HTTP doors."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal

from rlstack.data.stores.strangeloop import ScratchClient, StrangeLoopLocalStore
from rlstack.runner.remote import RemoteDesk, transport_for
from rlstack.runner.residents import Builds
from rlstack.runner.venues.runtime import MetalRuntime
from rlstack.runner.venues.strangeloop.provider import health


@dataclass(frozen=True)
class WorkerConfig:
    name: str
    store: str
    desk: str
    address: str
    port: int
    idle_s: float
    lease_id: str
    artifact_dir: str
    source_sha256: str
    builds: Builds | None

    @classmethod
    def read(cls, path: Path) -> WorkerConfig:
        """Install runtime credentials before spawned children reopen the Store."""
        try:
            row = json.loads(path.read_text())
            environment = row.pop("environment")
            for key in ("SL_API_BASE", "RLSTACK_HTTP_TOKEN"):
                if key not in environment:
                    raise ValueError(f"worker environment is missing {key}")
            # The scratch token itself, or the file the desk keeps it in —
            # the file is what a reauth rewrites when the token rotates.
            if not {"SL_API_TOKEN", "SL_API_TOKEN_FILE"} & set(environment):
                raise ValueError("worker environment is missing SL_API_TOKEN or SL_API_TOKEN_FILE")
            os.environ.update(environment)
            if row["builds"] is not None:
                row["builds"] = Builds.from_row(row["builds"])
            return cls(**row)
        finally:
            # The process environment, not a persistent artifact, holds auth.
            path.unlink(missing_ok=True)


async def run(config: WorkerConfig) -> None:
    """Refuse startup until this pod can read the desk and publish scratch."""
    from rlstack.runner.transports.http import HttpServer

    token = os.environ.get("RLSTACK_HTTP_TOKEN", "")
    deadline = asyncio.get_running_loop().time() + 60
    while True:
        try:
            if await asyncio.to_thread(health, config.desk, token=token, timeout=5):
                break
        except (OSError, ValueError):
            pass
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError("reverse SSH route to the standing desk is unavailable")
        await asyncio.sleep(0.25)
    client = ScratchClient.from_locator(config.store)
    store = StrangeLoopLocalStore(Path("/scratch") / client.prefix, client)
    await asyncio.to_thread(store.verify_publication)
    runtime = MetalRuntime.measured(
        config.name, store, RemoteDesk(transport_for(config.desk)),
        address=config.address,
        host_address=lambda host, epoch: config.address + "/" + host + "@" + epoch,
        container=config.lease_id, idle_s=config.idle_s, builds=config.builds)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(number, stop.set)
    # Start listening before the registration task can deliver a parked run
    # back to this metal. The contexts share this one owning event loop.
    async with HttpServer(runtime.service_for, port=config.port, token=token):
        async with runtime:
            print(json.dumps({"ready": True, "metal": config.name,
                              "epoch": runtime.service.epoch, "source_sha256": config.source_sha256,
                              "idle_s": config.idle_s, "pid": os.getpid(),
                              "hf_home": os.environ.get("HF_HOME"),
                              "hf_hub_cache": os.environ.get("HF_HUB_CACHE")}), flush=True)
            stopped = asyncio.create_task(stop.wait())
            released = asyncio.create_task(runtime.wait())
            try:
                await asyncio.wait((stopped, released), return_when=asyncio.FIRST_COMPLETED)
            finally:
                for pending in (stopped, released):
                    pending.cancel()
                await asyncio.gather(stopped, released, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(WorkerConfig.read(args.config)))


if __name__ == "__main__":
    main()
