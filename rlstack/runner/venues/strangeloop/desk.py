"""Local service wiring for Strange Loop; placement stays in the shared Desk."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import subprocess

from rlstack.data.stores.strangeloop import ScratchClient, StrangeLoopStore
from rlstack.runner.remote import RemoteDesk, transport_for
from rlstack.runner.venues.runtime import DeskRuntime
from rlstack.runner.venues.serving import (
    EVERY_INTERFACE, LOOPBACK, observer, own_local_desk, runtime_token,
)
from rlstack.runner.venues.strangeloop.provider import DeskConfig, Gateway, StrangeLoopVenue


def operator_token(config: DeskConfig) -> str:
    """The operator's bearer: `RLSTACK_HTTP_TOKEN` when the environment has
    it, else the local state file's — which only a laptop-run desk shares.
    A hosted desk's token lives on its volume and is copied into the
    environment once; without it the verbs refuse rather than mint a fresh
    local token the desk would never accept."""
    told = os.environ.get("RLSTACK_HTTP_TOKEN", "")
    if told:
        return told
    if config.operator_endpoint:
        raise RuntimeError("the hosted desk's token is not on this machine: export "
                           "RLSTACK_HTTP_TOKEN from `modal volume get "
                           "rlstack-strangeloop-desk-state desk/http-token`")
    return runtime_token(config.state_dir)


async def serve(config: DeskConfig, source_root: Path, *, exposed: bool = False,
                stop: asyncio.Event | None = None) -> None:
    """One journal owner and gateway, with mandatory finite idle release.

    `exposed` is the hosted desk: the gateway and the observer bind every
    interface behind the container's TLS ingress, both guarded by the
    runtime token. `stop` is handed in by a host that owns the process (a
    container's exit hook); a desk running as its own process stops on
    SIGINT/SIGTERM instead."""
    from rlstack.runner.transports.http import HttpServer

    host = EVERY_INTERFACE if exposed else LOOPBACK
    with own_local_desk(config.state_dir / "desk.lock"):
        token = runtime_token(config.state_dir)
        os.environ["RLSTACK_HTTP_TOKEN"] = token
        scratch = ScratchClient.from_locator(config.store)
        store = StrangeLoopStore(scratch)
        venue = StrangeLoopVenue(config, source_root, token=token)
        gateway = Gateway(config, token=token)
        runtime = DeskRuntime(
            store, transport_for=transport_for, provider=venue,
            bootable_metals=frozenset(metal.name for metal in config.metals),
            idle_s=config.idle_s, auto_recover=config.automatic_recovery)
        venue.registered = lambda name, lease_id, epoch: (
            runtime.desk.metal_containers.get(name) == lease_id
            and bool(epoch) and runtime.desk.epoch_of(name) == epoch and runtime.desk.leased(name)
            and name in runtime.desk.metal_remotes)
        gateway.desk_service = LocalDeskService(runtime, venue)
        if stop is None:
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for number in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(number, stop.set)
        try:
            # Reconnect routes for existing live leases BEFORE the runtime
            # ticks: its reaper probes every listing, and a listing whose
            # tunnel is not back yet is silent — on 2026-09-17 a restart
            # under automatic recovery concluded two live 7B hosts dead,
            # decarved them and stranded their runs. No new lease is
            # acquired merely because the desk restarted, and a lease that
            # cannot be reached (a pod without SSH yet, a tunnel refused) is
            # reported and left for `up` or `release` — never fatal.
            for metal in config.metals:
                state = venue.state(metal.name)
                if state and state.lease_id:
                    try:
                        row = await asyncio.to_thread(venue.cli.status, state.lease_id)
                        if row["status"] == "ready":
                            await asyncio.to_thread(venue.open_forward, metal, state.lease_id)
                            # A desk restart usually follows a login: hand
                            # every booted pod the profile's current token
                            # (2026-09-17), so runs paused on a 401 resume.
                            if state.boot_started:
                                await asyncio.to_thread(venue.reauth, metal.name)
                    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                        print(f"{metal.name}: lease {state.lease_id} not reconnected: {exc}",
                              flush=True)
            async with runtime:
                async with HttpServer(gateway.service_for, host=host, port=config.gateway_port,
                                      token=token):
                    with observer(StrangeLoopStore(scratch, read_only=True), config.observer_port,
                                  host=host, token=token if exposed else ""):
                        print(f"desk: http://{host}:{config.gateway_port}; "
                              f"observer: http://{host}:{config.observer_port}", flush=True)
                        await stop.wait()
        finally:
            venue.close()
            # A CPU process stopping does not prove a GPU stopped. Retain the
            # provider ids for explicit release/reconnection; leases remain bounded.
            active = [m.name for m in config.metals if venue.state(m.name)]
            if active:
                print("Desk stopped; inspect and explicitly release outstanding lease slots: " +
                      ", ".join(active), flush=True)


def desk(config: DeskConfig) -> RemoteDesk:
    """The operator's handle: the desk at its operator door, with the token
    every http transport reads from the environment."""
    os.environ["RLSTACK_HTTP_TOKEN"] = operator_token(config)
    return RemoteDesk(transport_for(config.operator_door))


class LocalDeskService:
    """One explicit acquisition door beside the unchanged campaign/Desk verbs."""
    def __init__(self, runtime: DeskRuntime, venue: StrangeLoopVenue) -> None:
        self.runtime, self.venue = runtime, venue

    async def serve(self, verb: str, payload: dict) -> dict:
        if verb == "boot":
            name = payload["metal"]
            if name not in self.venue.metals:
                raise ValueError(f"metal {name!r} is not an authorized allocation slot")
            await asyncio.to_thread(self.venue.boot, name)
            return {"metal": name, "registered": name in self.runtime.desk.metal,
                    "lease_id": self.venue.state(name).lease_id}
        if verb == "reauth":
            name = payload["metal"]
            if name not in self.venue.metals:
                raise ValueError(f"metal {name!r} is not an authorized allocation slot")
            return await asyncio.to_thread(self.venue.reauth, name)
        current = self.venue.state(payload["metal"]) if verb == "release" and payload["metal"] in self.venue.metals else None
        unmatched = current is not None and current.lease_id != self.runtime.desk.metal_containers.get(payload["metal"])
        if verb == "release" and (payload["metal"] not in self.runtime.desk.metal or unmatched):
            name = payload["metal"]
            if name not in self.venue.metals:
                raise ValueError(f"unknown allocation slot {name!r}")
            state = self.venue.state(name)
            if state is None or not state.lease_id:
                raise RuntimeError("allocation has no known lease id; reconcile its saved label through Strange Loop")
            ended = await self.venue.terminate(state.lease_id)
            return {"metal": name, "released": ended, "registered": False}
        if verb == "export":
            name = payload["metal"]
            return await asyncio.to_thread(self.venue.export, name, payload["run_ref"], payload["objective"])
        return await self.runtime.campaigns.serve(verb, payload)

    def answer(self, verb: str, payload: dict) -> dict:
        return self.runtime.campaigns.answer(verb, payload)
