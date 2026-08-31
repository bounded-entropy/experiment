"""Measurement: observation OUTSIDE the run (#70).

A run's identity is its training loop; measuring the policy is not part of
it. A Measurement is an OBSERVATION of a run, configured by its own manifest
(held-out task ids, samples, cadence, scoring pipeline, its own seed — none
of it hashed into anything), executed by any process that can reach a pool,
and written to measurements/<run_id>/<name>/ — never the run dir, which is
the run's own record. Swapping the observation mid-run is therefore a
non-event: stop one measurer, start another under a NEW name.

What makes this sound is two standing rules: eval was always FIREWALLED
(nothing training-side reads measurement), and retention keeps every adapter
version forever (KeepRestorable) precisely so "a version the run has long
moved past" can be rebuilt — so a measurer can BACKFILL the past, which no
in-run daemon ever could, and follow the ledger's future on any cadence.

`measure_run` is one idempotent pass: every EVERY-th committed version not
yet measured, restored from blobs (content-addressed proof), sampled on the
pinned problems through the given pool, scored by the post pipeline, reduced
IN WAVE ORDER (float addition is not associative; completion order must not
reach the bytes, #53), appended as one point. It PEEKS, never attaches — an
attach would race a live trainer's staged blobs — and it needs no arbiter:
admission happens at the pool's own host.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from rlstack.data.plan import GroupPlan, Sample, WavePlan
from rlstack.data.stores.base import Store
from rlstack.data.trajectory import Task, Wave
from rlstack.policy.compile import Bundle, restore_bundle
from rlstack.runner.assemble import sample_wave
from rlstack.runner.post import run_pipeline
from rlstack.spec.specs import SamplingSpec


@dataclass(frozen=True)
class Measurement:
    """One observation's whole configuration — the manifest IS this value."""

    name: str                      # measurements/<run_id>/<name>/
    env: str                       # the environment the problems run under
    task_ids: tuple[str, ...]      # the pinned held-out problems
    samples: int                   # completions per problem per point
    every: int                     # measure each EVERY-th committed update
    post: tuple[str, ...]          # scoring pipeline over the measured wave
    seed: int                      # the measurement's own master, never the run's
    temperature: float = 1.0
    max_tokens: int = 512

    def manifest(self) -> dict:
        return {**asdict(self), "task_ids": list(self.task_ids),
                "post": list(self.post)}


def measurement_bank(store: Store, run_id: str) -> dict[str, str]:
    """{bank name: adapter_type} off the run's manifest — what restore needs,
    read from the canonical row without decoding a single spec class."""
    manifest = store.peek_manifest(run_id)
    if manifest is None:
        raise FileNotFoundError(f"no manifest for {run_id!r} in this store")
    row = manifest["spec"]
    spec = json.loads(row) if isinstance(row, str) else row
    return {name: entry["adapter_type"]
            for name, entry in spec["policy"]["bank"].items()}


def restore_at(store: Store, run_id: str, entry: Mapping) -> Bundle:
    """The committed bundle a ledger entry names, rebuilt from blobs with the
    content-addressed id as the proof — by PEEKS, never an attach."""
    prefix = store.run_prefix(run_id)

    def read_blob(section: str, name: str, version: int) -> bytes:
        return store._read(f"{prefix}/{section}/{name}@{version}.bin")

    bank = measurement_bank(store, run_id)
    versions = {name: int(v) for name, v in entry["versions"].items()}
    return restore_bundle(versions, entry["bundle_id"], read_blob,
                          sorted(bank), bank)


async def measure_run(store: Store, run_id: str, measurement: Measurement,
                      pool, tasks: Mapping[str, Task],
                      max_inflight: int = 64) -> list[int]:
    """One idempotent pass over the run's ledger; returns the updates newly
    measured. Call it on any cadence — a schedule, a loop, by hand — and it
    backfills whatever is missing, then returns."""
    store.open_measurement(run_id, measurement.name, measurement.manifest())
    done = store.measured_updates(run_id, measurement.name)
    sampling = SamplingSpec(temperature=measurement.temperature,
                            max_tokens=measurement.max_tokens)
    plan = WavePlan(tuple(
        GroupPlan(task, tuple(Sample(task, measurement.env)
                              for _ in range(measurement.samples)))
        for task in measurement.task_ids))
    fresh: list[int] = []
    for entry in store.peek_ledger(run_id):
        update = int(entry["update"])
        if update % measurement.every or update in done:
            continue
        bundle = restore_at(store, run_id, entry)
        if not pool.knows_bundle(bundle.bundle_id):
            pool.add_bundle(bundle)
        routes = {"main": (pool, bundle)}
        wave = await sample_wave(plan, index=update, tasks=tasks,
                                 sampling=sampling, routes=routes,
                                 master=measurement.seed, phase="eval",
                                 max_inflight=max_inflight)
        columns = await run_pipeline(measurement.post, wave, routes, sampling,
                                     measurement.seed, update,
                                     phase="eval-post")
        store.append_measurement_point(
            run_id, measurement.name, reduce_point(update, wave, columns))
        fresh.append(update)
    return fresh


def reduce_point(update: int, wave: Wave, columns: dict[str, list]) -> dict:
    """One point, folded in WAVE order — which is task order: means per
    column, and per-task means so a curve can be unbundled problem by
    problem."""
    means = {name: math.fsum(values) / len(values)
             for name, values in sorted(columns.items()) if values}
    task_means: dict[str, dict[str, float]] = {}
    index = 0
    for group in wave.groups:
        width = len(group)
        for name, values in sorted(columns.items()):
            slice_ = values[index:index + width]
            if slice_:
                task_means.setdefault(name, {})[group.key] = (
                    math.fsum(slice_) / len(slice_))
        index += width
    return {"update": update, "episodes": len(wave),
            "means": means, "task_means": task_means}
