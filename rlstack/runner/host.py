"""Host: an ATOMIC, PURPOSED PARTITION of GPU capacity — the fleet's unit.

Not a GPU, not a node, not a container (#43): a host is some slice of some
GPUs (half of one L4 up through sixteen devices across two nodes) dedicated
to a declared capability, irreducible from a tenant's point of view. One
node can carry several hosts; one host can span several nodes; partition
boundaries are logical and the wire contract (runner/remote.py) is per-host
either way.

A host's CAPABILITY IS A BIRTH FACT: it is born with its partition and its
regimes (Regime below — inference or training over one base at one shard
shape), attests that the metal it was handed matches them, and never grows
or reshapes afterward. Experiments attach to hosts; only carving creates
them. A host born with one regime is dedicated; born with several, it
ALTERNATES them on the same partition — one host wearing masks, switched by
its own arbiter — never two hosts coordinating.

The experiment ↔ metal relationship gets ONE owner (CONTEXT #35). A Host
holds the metal: engine objects (each serving a base at a build shape), ONE
multi-tenant learner (never remote — the runner comes to it), and the
GpuArbiter — and `await host.submit(spec, schema)` is how an experiment
reaches it:

    bind      each declared pool onto an owned engine serving that pool's
              base (exact base first, wildcard fake metal as fallback)
    fit       refuse a submission whose new residents' declared fractions
              would push the GpuSet past capacity — honest now, because the
              host sees every tenant on this metal
    attest    roster the tenancy in memory (the admission-relevant truth,
              dies with the host's process) and journal it to the host's
              JOURNAL store (hosts/<name>/log.jsonl — observability the CLI
              reads, correctness never consults)
    run       run_experiment_async under the host's shared arbiter, against
              the experiment's OWN run store

    STORE OWNERSHIP INVARIANT (Samarth, #37): the run store is a
    PER-EXPERIMENT binding — one experiment, one store, for life — because
    run_id is global (I3) but existence is store-scoped: the same spec run
    against two stores forks history silently. submit() takes the run store
    explicitly (defaulting to the host's own for convenience); the journal
    records it per attach; the observer's runs view flags the same run_id
    seen in two stores as a fork.

Deploy scripts shrink to "build one Host, submit N specs"; the Phase-C
resident daemon is a Host kept alive behind a submission queue. The CLI over
the journals: `python -m rlstack hosts <store-root>`.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Mapping, Sequence
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


@dataclass(frozen=True)
class Partition:
    """The metal a host is born onto: an atomic slice of one GpuSet — the KIND
    of GPU it is made of ("L4", "H100"), the device indices, and the memory
    fraction it owns on each (vLLM's gpu_memory_utilization is a reservation,
    torch's set_per_process_memory_fraction a cap; both honor this number).
    Memory partitions honestly; SMs still time-share across partition
    boundaries — a stated cost, visible in latency, not hidden by this record.
    `gpu` is descriptive, never decisive (#49): the fleet carves by fraction
    and carries the kind down from the Metal it carved, so the journal, the
    status and the observer all say what the metal IS — a fraction alone
    cannot tell 0.5 of an L4 from 0.5 of an H100."""

    gpuset: str
    devices: tuple[int, ...]
    memory: float = 1.0
    gpu: str = ""

    def row(self) -> dict:
        """The partition as a JSON row — ONE home for the shape the host-up
        event journals, status() reports over the wire, and the observer's
        hosts view reads back (#49)."""
        return {"gpuset": self.gpuset, "gpu": self.gpu,
                "devices": list(self.devices), "memory": self.memory}


@dataclass(frozen=True)
class Regime:
    """One capability a host can wear: inference (an engine built tp=shape)
    or training (a learner built fsdp=shape) over one base. The fleet
    matches joins against these; the host attests its metal against them at
    birth and never changes them after."""

    name: str
    kind: str                       # "inference" | "training"
    base: str | None
    shape: int = 1

    def __post_init__(self) -> None:
        if self.kind not in ("inference", "training"):
            raise ValueError(
                f"Regime.kind must be 'inference' or 'training', "
                f"got {self.kind!r}")


@dataclass
class Tenancy:
    """One experiment on this host, as the roster tracks it."""

    run_id: str
    pools: dict[str, str]           # pool name -> base the bound engine serves
    store: str = ""                 # locator of the run's OWN store
    status: str = "running"         # running | done | failed
    updates_completed: int | None = None
    attached_at: float = field(default_factory=time.time)


class Host:
    def __init__(self, name: str, *, engines: Sequence[Engine],
                 learner: Learner | None, store: Store,
                 arbiter: GpuArbiter | None = None,
                 partition: Partition | None = None,
                 regimes: tuple[Regime, ...] = (),
                 capacity: float = 1.0, sampler=None) -> None:
        self.name = name
        self.engines = tuple(engines)
        self.learner = learner
        self.store = store
        self.arbiter = arbiter or GpuArbiter()
        self.partition = partition
        self.regimes = regimes
        self.capacity = capacity
        self.sampler = sampler or sample_gpu
        self.roster: dict[str, Tenancy] = {}
        self.attest_regimes()
        self._attach_regimes()
        store.append_host_event(name, {
            "event": "host-up", "t": time.time(),
            "engines": [engine.base or "*" for engine in self.engines],
            "partition": partition.row() if partition is not None else None,
            "regimes": [{"name": r.name, "kind": r.kind, "base": r.base,
                         "shape": r.shape} for r in regimes],
            "store": store.describe()})

    # ---- birth facts, one named method per rule -----------------------------

    def attest_regimes(self) -> None:
        """A host IS its regimes: each inference regime must be backed by an
        owned engine BUILT at exactly (base, tp=shape); each training regime
        by the learner built at fsdp=shape. Capability is a birth fact —
        attested here, never mutated after (#43); the deploy handing wrong
        metal dies at construction, not mid-run."""
        for regime in self.regimes:
            if regime.kind == "inference":
                if self.engine_for(regime.base, regime.shape) is None:
                    raise HostError(
                        f"host {self.name!r} declares regime {regime.name!r} "
                        f"({regime.base!r}, tp={regime.shape}) but no owned "
                        f"engine is built for it; engines: "
                        f"{[(e.base, e.tp) for e in self.engines]}")
            elif self.learner is None or self.learner.fsdp != regime.shape:
                built = None if self.learner is None else self.learner.fsdp
                raise HostError(
                    f"host {self.name!r} declares training regime "
                    f"{regime.name!r} at fsdp={regime.shape} but the learner "
                    f"handed is built fsdp={built}")

    def _attach_regimes(self) -> None:
        """A regime-host's residents attach AT BIRTH: the partition is the
        footprint, so joining tenants' declared fractions never count here
        (fraction is a carve hint, meaningless on a join — the weights
        already live, #43). Several regimes share ONE exclusive group: the
        alternation IS the host, switched by its own arbiter."""
        group = f"host:{self.name}" if len(self.regimes) > 1 else None
        for regime in self.regimes:
            obj = (self.learner if regime.kind == "training"
                   else self.engine_for(regime.base, regime.shape))
            # fraction 0.0, not None: a later tenant's declared fraction is
            # a carve hint and must never be adopted into this host's load
            self.arbiter.attach(obj, label=f"{self.name}:{regime.name}",
                                group=group, fraction=0.0)

    def engine_for(self, base: str | None, tp: int) -> Engine | None:
        """THE shape-matched lookup — exact base first, fake-metal wildcard
        as fallback, tp always exact (a build fact has no wildcard). One home
        for the rule: binding, attestation, and the wire's HostService all
        resolve through this."""
        exact = [e for e in self.engines if e.base == base and e.tp == tp]
        wildcard = [e for e in self.engines if e.base is None and e.tp == tp]
        return (exact or wildcard or [None])[0]

    # ---- the submission steps, one named method per rule --------------------

    def bind_pools(self, spec: ExperimentSpec,
                   remotes: frozenset[str] = frozenset()) -> dict[str, Engine]:
        """Each declared pool gets an owned engine serving its base at its
        declared tp (exact base preferred; base None is fake metal's
        wildcard). No match is a HostError — this host does not carry that
        model at that shape. Pools named in `remotes` are served by ANOTHER
        host and skip local binding."""
        binding: dict[str, Engine] = {}
        for group in spec.gpu_config.groups:
            for member in group.members:
                if not isinstance(member, PoolMember):
                    continue
                if member.name in remotes:
                    continue
                wanted = member.base or spec.policy.base
                engine = self.engine_for(wanted, member.tp)
                if engine is None:
                    raise HostError(
                        f"host {self.name!r} has no engine serving {wanted!r} "
                        f"at tp={member.tp} for pool {member.name!r}; it "
                        f"serves {[(e.base or '*', e.tp) for e in self.engines]}")
                binding[member.name] = engine
        return binding

    def check_fit(self, spec: ExperimentSpec, binding: dict[str, Engine],
                  remotes: frozenset[str] = frozenset()) -> None:
        """Refuse a submission whose NEW residents' declared fractions push
        the GpuSet past capacity. Already-attached residents add nothing
        (object-keyed: a shared engine or learner is one footprint), so
        multi-tenancy over shared metal is free and honestly bounded — and a
        regime-host's residents attached at birth, so every join is
        fraction-free by the same rule. Remote pools are another partition's
        footprint and never count here."""
        addition = 0.0
        seen: set[int] = set()
        for group in spec.gpu_config.groups:
            for member in group.members:
                if isinstance(member, PoolMember):
                    if member.name in remotes:
                        continue
                    obj = binding[member.name]
                else:
                    if self.learner is None:
                        raise HostError(
                            f"host {self.name!r} has no training regime; it "
                            f"cannot serve this spec's learner member")
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
                     store: Store | None = None,
                     max_inflight: int = 64,
                     remotes: Mapping[str, Engine] | None = None) -> RunReport:
        """Run one experiment on this host's metal. `remotes` maps pool names
        served by OTHER hosts to their RemotePools (the runner goes to the
        learner's host and reaches every other partition through the wire —
        #43); this host binds, fits, and journals only what it serves."""
        run_store = store if store is not None else self.store
        remote_pools = dict(remotes or {})
        binding = self.bind_pools(spec, remotes=frozenset(remote_pools))
        self.check_fit(spec, binding, remotes=frozenset(remote_pools))
        binding |= remote_pools
        rid = experiment_identity(spec, schema)
        self.roster[rid] = Tenancy(rid, pools={
            name: (engine.base or "*") for name, engine in sorted(binding.items())},
            store=run_store.describe())
        self.store.append_host_event(self.name, {
            "event": "attach", "t": time.time(), "run_id": rid,
            "pools": sorted(binding), "remotes": sorted(remote_pools),
            "n_updates": spec.algo.schedule.n_updates if spec.algo else None,
            "store": run_store.describe()})
        try:
            report = await run_experiment_async(
                spec, schema, run_store, binding, self.learner,
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
        """The GpuSet as this host sees it: the partition it was born onto
        (kind of GPU, devices, fraction — as a row, so a status crosses the
        wire unchanged), state (arbiter residency), declared load, and the
        tenant roster."""
        return {
            "host": self.name,
            "engines": [engine.base or "*" for engine in self.engines],
            "partition": self.partition.row() if self.partition else None,
            "regimes": [regime.name for regime in self.regimes],
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
