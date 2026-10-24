"""The Trainer: awaits data, trains, commits — the ledger's only writer.

One update, and the two-step protocol that makes kill -9 safe at any point
(ADR 0014):

    await the plan's wave → await the SCORER's part → INLINE POST →
    write the merged postdata → flatten/broadcast/pack →
    forward_backward × microbatches × epochs → optim_step → bump →
    DELIVER (push the bundle to every serving pool, under `wire`) →
    COMMIT (APPEND LEDGER — one line, every update) →
    CHECKPOINT (write blobs, APPEND CHECKPOINT — at the declared cadence,
    at the extent, and on a drained stop) → notify

The ledger line is what the Generator pins from; the checkpoint is what
resume restores. Everything above the CHECKPOINT tail is unsealed: attach
rewinds to it and the updates regenerate — at most `every` of them, which is
the trade the submission declared. Under `wire` delivery the bundle is on the
pools BEFORE the ledger line, so a reader of the commit never pins a version
its pool lacks; under `store` delivery this runner touches no engine and every
consumer faults in from the blobs, which is why `store` is bound to every=1.

THE POST PHASE HAS TWO HALVES since the Scorer exists (#65). The processors
that address a pool are the Scorer's, run beside that pool and arriving here as
a postdata PART; the pool-less ones are arithmetic over columns and run here,
over the part as `given`. The Trainer still writes the one merged
postdata/<u>.json, so everything downstream — flatten, the ledger's column
means, the observer, the wave browser — reads exactly what it always did. A
pipeline with no pooled half awaits nothing and plans no Scorer, and this
runner is byte-for-byte the runner it was.

A WAVE WHOSE ROWS TRAIN UNDER A LIBRARY SET WAITS FOR IT (ADR 0019): a row
whose `route` names `lib:<name>` (its leaf's role, or the route it was sampled
under) holds the update until that named adapter is present in the store;
the bytes are then handed to the learner as a frozen set before the first
forward_backward, and dropped after the step unless the next wave's plan
names them too. The learner never reads a store. A wave that names no library
takes none of these steps, and this runner is byte-for-byte the runner it was.

The gradient admits the LEARNER; the inline half admits nothing, because
occupying metal is what put a processor on the other side of the split. The
learner it admits may live on ANOTHER host (ADR 0006 Part A) — then the local
admission is bookkeeping over a free resident and the real one happens per
frame at the host that wears it, which this runner cannot tell and was never
meant to.

Those boundaries are also the update's four measured phases — collect / post /
train / seal — lapped into an UpdateClock and journaled to the HOST after the
commit. Durations are wall clock, so they live in the host journal and nowhere
near a run directory. `post` now measures the WAIT for the part plus the inline
half, which is the number that says whether the Scorer is keeping up: a scorer
running far enough ahead makes it collapse to the arithmetic alone.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from collections import defaultdict, deque
import math
from dataclasses import replace
import json
from functools import lru_cache, partial
import time

from rlstack.data.flatten import broadcast, flatten, pack
from rlstack.data.plan import EVAL, RunPlan
from rlstack.runner.assemble import realize
from rlstack.runner.refs import RefReader
from rlstack.data.stores.base import RunHandle, bump
from rlstack.data.stores.retention import DEFAULT_RETENTION, RetentionPolicy
from rlstack.data.trajectory import trajectory_from_row, wave_from_rows
from rlstack.policy.compile import compile_bundle
from rlstack.code import load_code
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.interfaces import Emitted, Learner, TrainStats
from rlstack.runner.pool import Pool
from rlstack.runner.meters import HostJournal, UpdateClock
from rlstack.runner.names import (
    check_plan_names, names_ready, names_subdir, row_lib_names,
    wave_lib_names,
)
from rlstack.runner.training_client import TrainingClient
from rlstack.runner.post import NamesReader, run_pipeline
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.weight_transfer import WeightTransfer
from rlstack.runner.publication import publish_tensors
from rlstack.runner.roles.base import Runner, StopRequest
from rlstack.runner.roles.scorer import SCORER
from rlstack.runner.signals import RunSignals, store_work
from rlstack.spec.flow import split_pipeline
from rlstack.spec.validate import serving_pools
from rlstack.spec.specs import ExperimentSpec, SamplingSpec
from rlstack.spec.canonical import canonical_json


class Trainer(Runner):
    def __init__(self, signals: RunSignals, arbiter: Arbiter, run: RunHandle, *,
                 spec: ExperimentSpec, plan: RunPlan, refs: RefReader,
                 learner: Learner, tenant: str,
                 initial_version: dict[str, int],
                 checkpointing: Checkpointing,
                 weight_transfer: WeightTransfer = WeightTransfer(),
                 pools: Mapping[str, Pool] | None = None,
                 stop: StopRequest | None = None,
                 journal: HostJournal | None = None,
                 retention: RetentionPolicy = DEFAULT_RETENTION,
                 training_client: TrainingClient | None = None) -> None:
        super().__init__(signals, arbiter, run)
        # Only a processor declaring training receives the composition client.
        self.training_client = training_client
        self.tenant = tenant
        self.checkpointing = checkpointing
        # THE WIRE (ADR 0014, Part B): the serving pools this runner pushes
        # each committed bundle to — empty under `store` delivery, and empty
        # for a run that declares no serving pool at all
        self.pools = dict(pools or {})
        self.has_serving = bool(serving_pools(spec))
        self.stop = stop
        # the last committed update's payload, held until it is checkpointed
        # — what a drained stop writes
        self.adapters: Mapping[str, bytes] | None = None
        self.dirty = False
        self.weight_transfer = weight_transfer
        self.publication_stats = None
        self.checkpoint_stats = None
        self.pin_s = None
        self.last_update = 0
        self.journal = journal
        self.retention = retention
        self.spec = spec
        split = split_pipeline(spec.algo.post)
        self.pooled, self.inline = split.pooled, split.inline
        self.schedule = spec.algo.schedule
        self.sampling = spec.gen.sampling if spec.gen else SamplingSpec()
        self.plan = plan
        self.refs = refs
        # Repeated injected prompts need token IDs once per tenant. This
        # bounded cache is rebuilt on resume and never contacts inference.
        self.tokenize = lru_cache(maxsize=4096)(partial(learner.tokenize, tenant))
        self.learner = learner
        self.version = dict(initial_version)
        bank = spec.policy.bank
        self.trainable = sorted(n for n, a in bank.items() if a.trainable)
        self.servable = sorted(n for n, a in bank.items()
                               if ADAPTER_TYPES.get(a.adapter_type).instance.serving is not None)
        self.adapter_types = {n: bank[n].adapter_type for n in self.servable}
        self.code = load_code(spec.code)
        # Named frozen states are interpreted by each adapter, never by a
        # recipe name or route grammar in the Trainer.
        self.named_adapters = tuple(ADAPTER_TYPES.get(a.adapter_type).instance for a in bank.values())
        check_plan_names(plan, self.named_adapters)
        self.library_entries = sorted(n for n, a in bank.items()
                                      if ADAPTER_TYPES.get(a.adapter_type).instance.named_states)
        self.loaded_libraries: tuple[tuple[str, str], ...] = ()
        self.library_requirements: dict[str, tuple[str, ...]] = {}

    # ---- the acquisition condition (override to change the alternation) -----

    def next_rows(self, update: int) -> list[dict] | None:
        """Update u's rows as its plan names them, or None while a leaf is
        still unsealed — the ONE await this runner has (#59).

        Realized rows are written into this run's own waves/ before they are
        used, so a wave is self-contained however far its leaves reached.
        """
        try:
            return self.run.read_wave(update)          # already realized
        except FileNotFoundError:
            pass
        rows = realize(self.plan.wave(update), self.refs)
        if rows is None:
            return None
        self.run.write_wave(update, rows)
        return rows

    def scored(self, update: int) -> bool:
        """The Scorer's part for this update exists — the SECOND await, and it
        exists only when the pipeline has a pooled half.

        A predicate over the store, like every other runner condition: the part
        file IS the handshake, and this runner never learns that a Scorer is
        running, only that a file appeared.
        """
        return self.run.read_postdata_part(update, SCORER) is not None

    async def await_scored(self, update: int) -> dict[str, list] | None:
        """The pooled half's columns, in wave order — {} when there is no
        pooled half, which is when the Trainer is exactly what it was before
        the Scorer existed; None when a stop arrived while waiting."""
        if not self.pooled:
            return {}
        if await self.wait_unless_stopped(lambda: self.scored(update)) is None:
            return None
        return await store_work(self.run.read_postdata_part, update, SCORER) or {}

    async def wait_unless_stopped(self, predicate):
        """`signals.wait_for`, with a second way out: a stop requested while
        waiting answers None, so the caller drains instead of waiting for
        work another runner will never finish."""
        while True:
            if self.stop is not None and self.stop.requested:
                return None
            value = await store_work(predicate)
            if value:
                return value
            await self.signals.wait_once()

    # ---- the runner ---------------------------------------------------------

    async def run_forever(self) -> None:
        await store_work(self.sweep_stale)  # converge interrupted work
        committed = await store_work(self.committed)
        self.last_update = committed
        if self.schedule.lr_scales is not None and len(self.schedule.lr_scales) != len(self.plan):
            raise ValueError("Schedule.lr_scales needs exactly one mapping per train wave")
        if committed == 0:
            await self.measure_probes(0)
        for update in range(committed + 1, len(self.plan) + 1):
            clock = UpdateClock()
            # THE WAITS ARE WHERE A STOP IS READ (ADR 0014, Q6): at the top of
            # an update, and inside every await for another runner's work —
            # a Trainer held waiting for a wave that will never come must
            # still drain when asked
            rows = await self.wait_unless_stopped(lambda: self.next_rows(update))
            if rows is None:
                await self.drain()
                return
            # the library sets these rows train under, awaited like the rows
            # themselves (and counted in `collect`): a missing name blocks,
            # an orphaned one refuses, a stop drains
            self.library_requirements = {entry: row_lib_names(rows, (
                ADAPTER_TYPES.get(self.spec.policy.bank[entry].adapter_type).instance,))
                for entry in self.library_entries}
            libraries = tuple(dict.fromkeys(name for names in self.library_requirements.values() for name in names))
            if libraries and not await self.libraries_ready(libraries):
                await self.drain()
                return
            wave = wave_from_rows(rows)
            clock.collected()

            # the pooled half, awaited on the store; then the inline half over
            # it; then the ONE merged file every reader downstream still reads
            given = await self.await_scored(update)
            if given is None:
                await self.drain()
                return
            if self.training_client is not None:
                self.training_client.update = update
            produced = await run_pipeline(
                self.inline, wave, {},
                self.sampling, self.spec.seeds.master, update, given=given,
                training=self.training_client, names=NamesReader(self.run))
            postdata = {**given, **produced}
            await store_work(self.run.write_postdata, update, postdata)
            clock.posted()

            # Injected text uses the learner's CPU tokenizer. Admit its
            # resident before issuing synchronous frames, just like training.
            tokenize = self.tokenize
            trained, columns = trained_rows(wave, postdata)
            async with self.arbiter.admit(self.learner):
                flats = await asyncio.to_thread(
                    lambda: [flatten(t, tokenize) for t in trained])
            docs = list(zip(flats, broadcast(columns, flats)))

            # ONE WAVE IS ONE GRADIENT UPDATE: every microbatch accumulates
            # into a single step, and the ledger line below is written only
            # after all of them — so a checkpoint is never half a wave (#59).
            stats: list[TrainStats] = []
            async with self.arbiter.admit(self.learner):
                if libraries:
                    await self.load_libraries(libraries)
                # every Learner verb is synchronous by protocol (ADR 0002 Q6)
                # and WAITS — on a resident child, or on another host's door
                # — so each is asked from a thread and this loop keeps
                # dispatching: the pool's other tenants sample through a
                # train step, and a learner across the wire cannot park the
                # host that is waiting on it (check_off_loop)
                # ONE VERB PER UPDATE (ADR 0022): a dp learner splits the
                # microbatches across its groups and sums their gradients in
                # microbatch order, so the step is the same bytes at any dp
                batches = training_batches(self.spec, docs, self.schedule.microbatch_tokens)
                stats.extend(await asyncio.to_thread(
                    self.learner.accumulate, self.tenant, batches))
                scales = (self.schedule.lr_scales[update - 1]
                          if self.schedule.lr_scales is not None else None)
                if scales is None:
                    await asyncio.to_thread(self.learner.optim_step, self.tenant)
                else:
                    await asyncio.to_thread(self.learner.optim_step, self.tenant, scales)
                self.version = bump(self.version, active_entries(self.spec, trained))
                due = self.checkpointing.due(update, len(self.plan))
                tensor_pin = None
                export_started = time.monotonic()
                self.publication_stats, self.checkpoint_stats = None, None
                self.pin_s = None
                if self.pools and self.weight_transfer.backend == "nccl":
                    tensor_pin = await publish_tensors(self.learner, self.tenant,
                        self.version, self.servable, self.adapter_types, self.pools,
                        self.weight_transfer, self.journal, update)
                    self.adapters = None
                else:
                    self.adapters = (await asyncio.to_thread(
                        self.learner.emit_adapters, self.tenant, self.version)
                        if self.has_serving or due else None)
                    self.publication_stats = {
                        "backend": "base64" if self.has_serving else None,
                        "adapter_export_s": (time.monotonic() - export_started
                                             if self.adapters is not None else None),
                        "adapter_bytes": (sum(map(len, self.adapters.values()))
                                          if self.adapters is not None else None)}
                    if not self.has_serving:
                        pin_started = time.monotonic()
                        tensor_pin = await asyncio.to_thread(self.learner.policy_pin,
                            self.tenant, self.version, self.servable, self.adapter_types)
                        self.pin_s = time.monotonic() - pin_started
                if self.loaded_libraries:
                    await self.drop_libraries(keep=self.next_libraries(update))
            await self.measure_probes(update)
            if any(stat.components for stat in stats):
                await store_work(self.run.write_postdata_part, update, "training",
                                 training_measurements(trained, flats, batches, stats))
            clock.trained()

            bundle = tensor_pin or compile_bundle(self.adapters or {}, self.version,
                                    self.servable if self.has_serving else (),
                                    self.adapter_types, self.code)
            delivery_started = time.monotonic()
            await self.deliver(bundle)
            if self.publication_stats is not None:
                self.publication_stats["transfer_install_s"] = (
                    time.monotonic() - delivery_started if self.pools else None)
            await self.commit(update, bundle, wave, postdata, stats)
            self.dirty, self.last_update = True, update
            if due:
                await self.checkpoint(update)
            clock.sealed()
            await store_work(self.journal_update, clock, update)
            await self.signals.notify()

    async def measure_probes(self, update: int) -> None:
        """Persist complete read-only probe vectors at declared boundaries."""
        probes = [probe for probe in self.schedule.probes or () if probe.update == update]
        if not probes:
            return
        rows = []
        for probe in probes:
            loaded = await store_work(self.refs.rows, probe.rows)
            if loaded is None:
                raise FileNotFoundError(probe.rows)
            rows.extend(loaded)
        trajectories = [trajectory_from_row(row) for row in rows]
        async with self.arbiter.admit(self.learner):
            flats = await asyncio.to_thread(
                lambda: [flatten(trajectory, self.tokenize) for trajectory in trajectories])
            values = [None] * len(flats)
            batches = training_batches(self.spec, [(flat, {}) for flat in flats],
                                       self.schedule.microbatch_tokens)
            positions = document_indices(trajectories, flats, batches)
            for batch, indices in zip(batches, positions, strict=True):
                measured = await asyncio.to_thread(self.learner.forward, self.tenant, batch)
                for index, value in zip(indices, measured, strict=True):
                    values[index] = float(value)
        counts = [sum(flat.loss_mask) for flat in flats]
        await store_work(self.run.write_postdata_part, update, "probe", {
            "row_ids": [row["task"]["id"] for row in rows],
            "routes": [trajectory_route(trajectory) for trajectory in trajectories],
            "nll": values, "scored_tokens": counts,
            "nll_sum": [value * count for value, count in zip(values, counts)],
            "versions": dict(self.version),
            "reduction": "selected-token mean NLL per sealed document"})

    # ---- the library: named adapters the rows train under (ADR 0019) --------

    async def libraries_ready(self, names: tuple[str, ...]) -> bool:
        """Every name present — True — or a stop arrived while waiting —
        False, and the caller drains. An orphaned name raises out of the
        run: its writer ended, so waiting is not an answer and neither is
        training from init."""
        if not self.library_entries:
            raise RuntimeError(
                f"this wave's rows train under library sets {list(names)} but "
                f"the bank holds no adapter with named state slots")
        await names_ready(self.run.store, names_subdir(self.run), names,
                          self.signals, journal=self.journal,
                          stopped=lambda: self.stop is not None and self.stop.requested)
        return not (self.stop is not None and self.stop.requested)

    async def load_libraries(self, names: tuple[str, ...]) -> None:
        """Hand the learner each name's bytes as the frozen set
        in each declared adapter-owned state slot it does not already hold. This runner reads the store; the learner is handed a payload,
        as `Learner.load` already works (ADR 0019, Q2). Called inside the
        gradient's admission, each verb from a thread like the rest."""
        subdir = names_subdir(self.run)
        payloads = {}
        for entry, needed in self.library_requirements.items():
            adapter = ADAPTER_TYPES.get(self.spec.policy.bank[entry].adapter_type).instance
            for name in needed:
                if (entry, name) in self.loaded_libraries:
                    continue
                if name not in payloads:
                    payloads[name] = await store_work(self.run.store.read_named, subdir, name)
                await asyncio.to_thread(self.learner.load_set, self.tenant, entry,
                                        adapter.named_state_route(name), payloads[name])
                self.loaded_libraries += ((entry, name),)

    def next_libraries(self, update: int) -> tuple[str, ...]:
        """The names the NEXT wave's plan mentions — what is worth keeping
        loaded across the step. A WaveRef or the plan's end mentions none:
        its names are dropped now and loaded again if its rows carry them."""
        if update >= len(self.plan):
            return ()
        return wave_lib_names(self.plan.wave(update + 1), self.named_adapters)

    async def drop_libraries(self, keep: tuple[str, ...]) -> None:
        """After the step, free every loaded library set the next wave does
        not name: a stream of hundreds of memories must not accumulate on the
        learner."""
        for entry, name in self.loaded_libraries:
            if name in keep:
                continue
            adapter = ADAPTER_TYPES.get(self.spec.policy.bank[entry].adapter_type).instance
            await asyncio.to_thread(self.learner.drop_set, self.tenant, entry,
                                    adapter.named_state_route(name))
        self.loaded_libraries = tuple(pair for pair in self.loaded_libraries if pair[1] in keep)

    # ---- the three steps, named (ADR 0014) ----------------------------------


    async def deliver(self, bundle) -> None:
        """THE WIRE: the new bundle on every pool that serves the policy,
        BEFORE the ledger line — so any runner reading the commit can pin it
        at once — over whatever transport each pool's proxy speaks (local,
        pipe, Modal, HTTP). Asked from a thread: the pool may be another
        container and the attach is its wait. Under `store` delivery, or in a
        run with no serving pool, there is nothing here to do.

        EVERY LIVE REPLICA of each pool receives it (ADR 0022); a replica
        found dead here is revived at once and brought up to this bundle
        before it is readmitted, so the commit below never names a version
        a serving replica lacks."""
        for name in sorted(self.pools):
            await self.pools[name].deliver(bundle)

    async def commit(self, update: int, bundle, wave, postdata: dict,
                     stats: list[TrainStats]) -> None:
        """THE COMMIT: one ledger line — the update's facts, the version map
        and the bundle id — cheap, every update, the Trainer's alone."""
        await store_work(self.run.append_ledger, {
            "update": update,
            "versions": dict(self.version),
            "bundle_id": bundle.bundle_id,
            "wave": {"trajectories": len(wave), "groups": len(wave.groups)},
            "post": _column_means(postdata),
            "train": _train_summary(stats),
        })

    async def checkpoint(self, update: int) -> None:
        """THE DURABLE POINT: every trainable entry's adapter and optimizer
        blobs at the committed version, then the checkpoint line that seals
        them, then the sweep of what that line moved past. At the declared
        cadence, at the extent's last update, and on a drained stop — the
        only times bytes of policy state reach the store."""
        if not self.dirty or self.checkpointing.state == "recompute":
            return                      # nothing committed since the last checkpoint
        started = time.monotonic()
        adapters = self.adapters
        if adapters is None:
            async with self.arbiter.admit(self.learner):
                adapters = await asyncio.to_thread(
                    self.learner.emit_adapters, self.tenant, self.version)
        # This read cannot require waking a learner: on an alternating host
        # its own Generator may be waiting under engine admission during a
        # drained stop. The resident serializes the read with all mutations.
        optim = await asyncio.to_thread(self.learner.emit_optim, self.tenant)
        exported = time.monotonic()
        for name in self.trainable:
            await store_work(self.run.write_blob, "adapters", name,
                             self.version[name], adapters[name])
            await store_work(self.run.write_blob, "optim", name,
                             self.version[name], optim[name])
        if update == len(self.plan):
            await self.publish_outputs(adapters, update)
        await store_work(self.run.append_checkpoint, update, dict(self.version))
        self.adapters, self.dirty = None, False
        self.checkpoint_stats = {"export_s": exported - started,
            "store_s": time.monotonic() - exported,
            "adapter_bytes": sum(len(adapters[name]) for name in self.trainable),
            "optimizer_bytes": sum(map(len, optim.values())),
            "every": self.checkpointing.every, "update": update}
        if self.journal is not None:
            await store_work(self.journal.append, {"event": "checkpoint",
                "run_id": self.tenant, "versions": dict(self.version),
                **self.checkpoint_stats})
        await store_work(self.sweep_stale)

    async def publish_outputs(self, adapters: Mapping[str, bytes], update: int) -> None:
        """Seal named exports before the final checkpoint declares completion.

        The snapshot's blobs already exist. A crash before the checkpoint
        append replays the ordinary update and checks this same name/provenance.
        Consumers therefore cannot see a completed producer with an owed name.
        """
        from rlstack.spec.canonical import canonical_json

        outputs = [(name, adapter) for name, adapter in self.spec.policy.bank.items()
                   if adapter.out is not None]
        if not outputs:
            return
        subdir = names_subdir(self.run)
        run_ref = f"{subdir}/{self.tenant}"
        for name, adapter in outputs:
            vectors, probe_uri = [], None
            for probe in self.schedule.probes or ():
                result = await store_work(self.run.read_postdata_part, probe.update, "probe")
                if result is None:
                    continue
                vector = [nll for route, nll in zip(result["routes"], result["nll"])
                          if route == adapter.route]
                if vector and probe.update not in [number for number, _ in vectors]:
                    vectors.append((probe.update, vector))
                if probe_uri is None and vector:
                    rows = await store_work(self.refs.rows, probe.rows)
                    selected = [row for row in rows
                                if trajectory_route(trajectory_from_row(row)) == adapter.route]
                    probe_uri = await store_work(self.run.store.cas_put,
                        b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                                 + b"\n" for row in selected))
            meta = {"base": self.spec.policy.base, "adapter_type": adapter.adapter_type,
                    "site": adapter.site, "init": dict(adapter.init),
                    "run_ref": run_ref, "entry": name, "update": update,
                    "version": self.version[name], "loss": self.spec.algo.loss,
                    "parent": self.spec.init.policy if self.spec.init else None,
                    "steps": self.version[name], "probe": probe_uri,
                    "probes": [vector for _, vector in sorted(vectors)]}
            if "r" in adapter.init:
                meta["r"] = int(adapter.init["r"])
            existing = await store_work(self.run.store.named_meta, subdir, adapter.out)
            if existing is not None and canonical_json(
                    {key: value for key, value in existing.items() if key != "sha256"}) != canonical_json(meta):
                raise ValueError(f"named output {adapter.out!r} already has different provenance")
            await store_work(self.run.store.write_named, subdir, adapter.out,
                             adapters[name], meta)

    async def drain(self) -> None:
        """A DELIBERATE STOP LOSES NOTHING (ADR 0014, Q6): at the update
        boundary the stop was read, seal the last committed update with a
        checkpoint — unless it already is one — and let the run end."""
        await self.checkpoint(self.last_update)
        self.stop.drained = True

    # ---- retention (at the checkpoint, because that is when a version goes stale)

    def sweep_stale(self) -> None:
        """Free what the checkpoint record has moved past — every optimizer
        blob below the checkpoint tail, and nothing else the default policy
        can name.

        AT THE CHECKPOINT, because that is the moment the previous
        checkpoint's moments stopped having a reader, and the moment the
        whole design already serializes on — no runner, no clock, no second
        authority. AND WHEN THIS RUNNER STARTS, because a process killed
        between a checkpoint and its sweep leaves one stale blob nobody would
        ever come back for: the run's LAST checkpoint has no later one to
        sweep after it. Sweeping on attach makes the store CONVERGE — after
        any checkpoint and after any attach it holds every checkpointed
        adapter and exactly one optim per delta — which is what keeps a run
        directory a pure function of (spec, code, data) even though bytes now
        leave it.

        NEVER FATAL. The checkpoint is sealed; its line says so, and a failed
        unlink leaves bytes on a volume, which is the cheapest failure in the
        system. It also heals: the policy is a pure function of the ledger, so
        every later commit names the same blobs again, and `python -m rlstack
        sweep` is the backstop for the last one. What was freed is deliberately
        recorded NOWHERE — it is a fact about a directory, not about the
        experiment, and a run directory holding it would no longer be a pure
        function of (spec, code, data).
        """
        try:
            self.run.sweep(self.retention)
        except Exception:      # any backend failure; see the rule above
            pass

    # ---- the emission (observability; never a run-directory byte) -----------

    def journal_update(self, clock: UpdateClock, update: int) -> None:
        """One `update` event on the HOST's journal: how long this update took
        and where the time went. Wall clock may never enter a run directory —
        resume-equivalence is byte-identical run dirs — so the ledger line
        above carries the update's FACTS and this line carries its DURATION,
        in the one place durations are allowed. A run on no host emits
        nothing."""
        if self.journal is not None:
            self.journal.append(clock.row(self.tenant, update))
            if self.publication_stats is not None and self.publication_stats["backend"] is not None:
                self.journal.append({"event": "weight-transfer", "run_id": self.tenant,
                    "update": update, "versions": dict(self.version), **self.publication_stats})
            if self.pin_s is not None:
                self.journal.append({"event": "policy-pin", "run_id": self.tenant,
                    "update": update, "versions": dict(self.version), "hash_s": self.pin_s})


def trajectory_route(trajectory) -> str | None:
    """A document has one explicit route, or uses the default policy slot."""
    routes = {turn.turn_extras["route"] for turn in trajectory.turns
              if "route" in turn.turn_extras}
    if len(routes) > 1:
        raise ValueError(f"one supervised document has multiple routes: {sorted(routes)}")
    return next(iter(routes), None)


def training_measurements(trajectories, flats, batches, stats) -> dict:
    """Join exact microbatch sufficient statistics to sealed document IDs."""
    measured = []
    for indices, stat in zip(document_indices(trajectories, flats, batches), stats, strict=True):
        measured.append({"row_ids": [trajectories[index].task.id for index in indices],
                         "routes": [trajectory_route(trajectories[index]) for index in indices],
                         "tokens": stat.tokens, "loss": stat.loss, "components": dict(stat.components)})
    return {"microbatches": measured,
            "cadence": "every committed optimizer update",
            "reduction": "loss-defined; per-document numerators, counts and weights retained unaveraged"}


def document_indices(trajectories, flats, batches) -> list[list[int]]:
    """Packing may reorder cohorts, while measurements keep declared row IDs."""
    identities = defaultdict(deque)
    for index, (trajectory, flat) in enumerate(zip(trajectories, flats, strict=True)):
        identities[(flat.token_ids, flat.loss_mask, trajectory_route(trajectory))].append(
            index)
    positions = []
    for batch in batches:
        indices = []
        ends = (*batch.doc_starts[1:], len(batch))
        for index, (start, stop) in enumerate(zip(batch.doc_starts, ends, strict=True)):
            facts = batch.doc_turn_extras[index] if batch.doc_turn_extras else ()
            route = next((fact["route"] for fact in facts if "route" in fact), None)
            indices.append(identities[(batch.token_ids[start:stop], batch.loss_mask[start:stop], route)].popleft())
        positions.append(indices)
    if any(identities.values()):
        raise ValueError("packed measurement omitted sealed documents")
    return positions


def active_entries(spec, trajectories) -> list[str]:
    routes = {trajectory_route(trajectory) for trajectory in trajectories}
    return sorted(name for name, adapter in spec.policy.bank.items()
                  if adapter.trainable and (adapter.route is None or adapter.route in routes))


def training_batches(spec, docs, budget):
    """Compatible routes share forwards; other lowerings keep uniform rows.

    Packing never changes a route's statistical weight. The independent SFT
    loss gets its explicit per-document weights from the normal post phase.
    """
    routed = {adapter.route for adapter in spec.policy.bank.values() if adapter.route is not None}
    if not routed:
        return list(pack(docs, budget))
    cohorts, route_keys = {}, {}
    for route in routed | {None}:
        entries = [adapter for _, adapter in sorted(spec.policy.bank.items()) if adapter.route == route]
        signature = tuple(canonical_json({"type": adapter.adapter_type, "site": adapter.site,
                                         "init": {key: value for key, value in adapter.init.items() if key != "seed"}})
                          for adapter in entries)
        mixed = entries and all(ADAPTER_TYPES.get(adapter.adapter_type).instance.supports_mixed_routes
                                for adapter in entries)
        route_keys[route] = signature if mixed else ("route", route)
    for doc in docs:
        flat, _ = doc
        routes = {turn["route"] for turn in flat.turn_extras if "route" in turn}
        route = next(iter(routes), None)
        key = route_keys.get(route, ("route", route))
        cohorts.setdefault(key, []).append(doc)
    batches = [batch for cohort in cohorts.values() for batch in pack(cohort, budget)]
    return [replace(batch, microbatches_in_update=len(batches), documents_in_update=len(docs))
            for batch in batches]


def trained_rows(wave, postdata: Mapping[str, Sequence]) -> tuple[list, dict[str, list]]:
    """THE EVAL RULE: a row whose role is `eval` (data/plan.py) is scored —
    its postdata rides the wave and the ledger — but never forwarded. The
    loss would only mask it, and the learner's padded forward pays rows ×
    longest document for the whole microbatch, so a wave of short answers
    beside long dreams cost tens of gigabytes for nothing (measured: a
    58-row wave with 54 answers asked the learner for 8.5 GiB of logits).
    A wave with no trained row is a plan error, not an empty update."""
    kept = [i for i, trajectory in enumerate(wave.trajectories)
            if not any(turn.turn_extras.get("role") == EVAL for turn in trajectory.turns)]
    if not kept:
        raise RuntimeError(
            f"update's wave holds {len(wave.trajectories)} rows and every one is eval; "
            f"a train wave needs at least one row to train")
    return ([wave.trajectories[i] for i in kept],
            {name: [values[i] for i in kept] for name, values in postdata.items()})


def _column_means(columns) -> dict[str, float]:
    """Scalar columns mean over trajectories; token_level columns over all
    their tokens — one ledger scalar either way."""
    out = {}
    for name, values in sorted(columns.items()):
        flat = [v for value in values
                for v in (value if isinstance(value, (list, tuple)) else [value])]
        if flat:
            out[name] = math.fsum(flat) / len(flat)
    return out


def _train_summary(stats: list[TrainStats]) -> dict:
    """One update's training facts. The bank's PROVIDED tensors are folded in
    first and the standing rails written over them, so the rails always own
    their own names; every other provided name lands here beside them, mean of
    the microbatch means — the same convention `loss` already uses, and exact
    for a quantity that is a function of the parameters alone (a latent KL is
    identical in every microbatch of one update)."""
    n = len(stats)
    return {
        **_provided_means(stats),
        "microbatches": n,
        "tokens": sum(s.tokens for s in stats),
        "loss": math.fsum(s.loss for s in stats) / n,
        "mean_ratio": math.fsum(s.mean_ratio for s in stats) / n,
        "logprob_gap": max(s.logprob_gap for s in stats),
        "grad_norm": max(s.grad_norm for s in stats),
        **({"loss_components": [dict(s.components) for s in stats],
            "objective_sum": math.fsum(s.loss for s in stats)}
           if any(s.components for s in stats) else {}),
    }


def _provided_means(stats: list[TrainStats]) -> dict[str, float]:
    """Each declared provided name, meaned across the update's microbatches."""
    names = sorted({name for s in stats for name in s.provided})
    return {name: math.fsum(s.provided[name] for s in stats if name in s.provided)
                  / sum(1 for s in stats if name in s.provided)
            for name in names}
