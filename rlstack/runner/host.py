"""Host: the owner of one GpuSet's quartet — engines, learner, arbiter, store.

The experiment ↔ metal relationship gets ONE owner (CONTEXT #35). A Host
holds the metal: engine objects (each serving a base), ONE multi-tenant
learner, and the GpuArbiter — and `await host.submit(spec, schema)` is how an
experiment reaches it:

    bind      each declared pool onto an owned engine serving that pool's
              base (exact base first, wildcard fake metal as fallback)
    fit       refuse a submission whose new residents' declared fractions
              would push the GpuSet past capacity — honest now, because the
              host sees every tenant on this metal
    attest    roster the tenancy in memory (the admission-relevant truth,
              dies with the host's process) and journal it to the store
              (hosts/<name>/log.jsonl — observability the CLI reads,
              correctness never consults)
    run       run_experiment_async under the host's shared arbiter

Deploy scripts shrink to "build one Host, submit N specs"; the Phase-C
resident daemon is a Host kept alive behind a submission queue. The CLI over
the journals: `python -m rlstack hosts <store-root>`.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from rlstack.data.stores.base import Store
from rlstack.policy.siteschema import SiteSchema
from rlstack.runner.arbiter import GpuArbiter
from rlstack.runner.interfaces import Engine, Learner
from rlstack.runner.loop import (
    RunReport, experiment_identity, run_experiment_async,
)
from rlstack.spec.specs import ExperimentSpec, PoolMember


class HostError(RuntimeError):
    """A submission this host cannot serve (no matching engine, no capacity)."""


@dataclass
class Tenancy:
    """One experiment on this host, as the roster tracks it."""

    run_id: str
    pools: dict[str, str]           # pool name -> base the bound engine serves
    status: str = "running"         # running | done | failed
    updates_completed: int | None = None
    attached_at: float = field(default_factory=time.time)


class Host:
    def __init__(self, name: str, *, engines: Sequence[Engine],
                 learner: Learner, store: Store,
                 arbiter: GpuArbiter | None = None,
                 capacity: float = 1.0, sampler=None) -> None:
        self.name = name
        self.engines = tuple(engines)
        self.learner = learner
        self.store = store
        self.arbiter = arbiter or GpuArbiter()
        self.capacity = capacity
        self.sampler = sampler or sample_gpu
        self.roster: dict[str, Tenancy] = {}
        store.append_host_event(name, {
            "event": "host-up", "t": time.time(),
            "engines": [engine.base or "*" for engine in self.engines],
            "store": store.describe()})

    # ---- the submission steps, one named method per rule --------------------

    def bind_pools(self, spec: ExperimentSpec) -> dict[str, Engine]:
        """Each declared pool gets an owned engine serving its base (exact
        match preferred; base None is fake metal's wildcard). No match is a
        HostError — this host does not carry that model."""
        binding: dict[str, Engine] = {}
        for group in spec.gpu_config.groups:
            for member in group.members:
                if not isinstance(member, PoolMember):
                    continue
                wanted = member.base or spec.policy.base
                exact = [e for e in self.engines if e.base == wanted]
                wildcard = [e for e in self.engines if e.base is None]
                if not exact and not wildcard:
                    raise HostError(
                        f"host {self.name!r} has no engine serving {wanted!r} "
                        f"for pool {member.name!r}; it serves "
                        f"{sorted({e.base or '*' for e in self.engines})}")
                binding[member.name] = (exact or wildcard)[0]
        return binding

    def check_fit(self, spec: ExperimentSpec,
                  binding: dict[str, Engine]) -> None:
        """Refuse a submission whose NEW residents' declared fractions push
        the GpuSet past capacity. Already-attached residents add nothing
        (object-keyed: a shared engine or learner is one footprint), so
        multi-tenancy over shared metal is free and honestly bounded."""
        addition = 0.0
        seen: set[int] = set()
        for group in spec.gpu_config.groups:
            for member in group.members:
                if isinstance(member, PoolMember):
                    obj = binding[member.name]
                else:
                    obj = self.learner
                if self.arbiter.is_attached(obj) or id(obj) in seen:
                    continue
                seen.add(id(obj))
                addition += member.fraction or 0.0
        load = self.arbiter.declared_load() + addition
        if load > self.capacity:
            raise HostError(
                f"host {self.name!r} cannot fit this submission: declared "
                f"load would be {load:.2f} of capacity {self.capacity:.2f}")

    # ---- submit -------------------------------------------------------------

    async def submit(self, spec: ExperimentSpec, schema: SiteSchema,
                     max_inflight: int = 64) -> RunReport:
        binding = self.bind_pools(spec)
        self.check_fit(spec, binding)
        rid = experiment_identity(spec, schema)
        self.roster[rid] = Tenancy(rid, pools={
            name: (engine.base or "*") for name, engine in sorted(binding.items())})
        self.store.append_host_event(self.name, {
            "event": "attach", "t": time.time(), "run_id": rid,
            "pools": sorted(binding),
            "n_updates": spec.algo.schedule.n_updates if spec.algo else None,
            "store": self.store.describe()})
        try:
            report = await run_experiment_async(
                spec, schema, self.store, binding, self.learner,
                max_inflight, arbiter=self.arbiter)
        except BaseException:
            self.roster[rid].status = "failed"
            self.store.append_host_event(self.name, {
                "event": "detach", "t": time.time(), "run_id": rid,
                "status": "failed"})
            raise
        self.roster[rid].status = "done"
        self.roster[rid].updates_completed = report.updates_completed
        self.store.append_host_event(self.name, {
            "event": "detach", "t": time.time(), "run_id": rid,
            "status": "done", "updates_completed": report.updates_completed})
        return report

    # ---- observability ------------------------------------------------------

    async def run_stats(self, every: float = 30.0) -> None:
        """Journal one GPU sample every `every` seconds until cancelled —
        the CLI's `gpu` view (utilization, memory, downtime) is computed from
        these events; a gap in them IS the downtime. Run it alongside
        submissions: create_task(host.run_stats()), cancel when done."""
        while True:
            sample = await asyncio.to_thread(self.sampler)
            if sample is not None:
                self.store.append_host_event(self.name, {
                    "event": "stats", "t": time.time(), **sample})
            await asyncio.sleep(every)

    def status(self) -> dict:
        """The GpuSet as this host sees it: state (arbiter residency),
        declared load, and the tenant roster."""
        return {
            "host": self.name,
            "engines": [engine.base or "*" for engine in self.engines],
            "declared_load": round(self.arbiter.declared_load(), 3),
            "residency": self.arbiter.residency(),
            "tenants": {rid: {"status": t.status, "pools": t.pools,
                              "updates_completed": t.updates_completed}
                        for rid, t in sorted(self.roster.items())},
        }


def sample_gpu() -> dict | None:
    """One nvidia-smi sample: per-GPU utilization %% and memory MiB. None
    when no NVIDIA runtime is visible (fakes, laptops, CPU CI) — sampling
    quietly does nothing off the metal."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        try:
            util, used, total = [int(part.strip()) for part in line.split(",")]
        except ValueError:
            continue
        gpus.append({"util": util, "mem_used": used, "mem_total": total})
    return {"gpus": gpus} if gpus else None
