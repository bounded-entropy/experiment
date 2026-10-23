"""Compile independent supervised curricula into ordinary sealed Replay plans.

All students share each forward where packing permits it, but retain separate
adapter parameters, optimizer ages, loss denominators and learning-rate schedules.
There is no optimizer loop in this module.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import random
from typing import Any

from rlstack.client import Training, TrainingResult
from rlstack.data.plan import GroupPlan, Replay, RunPlan, WavePlan, encode
from rlstack.data.trajectory import trajectory_from_row
from rlstack.spec.specs import (
    AdapterSpec, AlgoSpec, ExperimentSpec, HostSpec, OptimSpec, Plans,
    PolicySpec, ProbeSpec, Schedule, Seeds, Topology, WarmStart, learner,
)

BASE_LR = 1e-4


@dataclass(frozen=True)
class SftStage:
    """One stage's complete optimizer recipe; its data is already sealed."""

    epochs: int
    batch: int
    lr: float
    decay: str = "linear"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch < 1:
            raise ValueError("SFT epochs and batch must be positive")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("SFT learning rate must be finite and positive")
        if self.decay not in ("linear", "constant"):
            raise ValueError("SFT decay must be linear or constant")


@dataclass(frozen=True)
class SftCurriculum:
    """One independent student: a sealed row set for each successive stage."""

    datasets: tuple[tuple[Mapping[str, Any], ...], ...]
    stages: tuple[SftStage, ...]

    def __post_init__(self) -> None:
        if not self.stages or len(self.datasets) != len(self.stages):
            raise ValueError("each SFT stage needs exactly one sealed dataset")
        if any(not rows for rows in self.datasets):
            raise ValueError("an SFT stage cannot have an empty dataset")
        for rows in self.datasets:
            for row in rows:
                trajectory_from_row(row)


def student_name(index: int) -> str:
    return f"student{index}"


def student_route(index: int) -> str:
    return f"student/{index}"


def curriculum_spec(base: str, adapter: AdapterSpec,
                    curricula: Sequence[SftCurriculum], seed: int,
                    put: Callable[[bytes], str], *,
                    start_payloads: Sequence[bytes | None] | None = None,
                    microbatch_tokens: int = 16384,
                    optim: OptimSpec | None = None,
                    probe_rows: Sequence[Sequence[Mapping[str, Any]]] | None = None) -> ExperimentSpec:
    """One Replay wave per shared optimizer step; finished students disappear.

    Fresh students use the same explicit initialization seed. Epoch shuffles
    use the historical common fork key, so identical curricula are controls.
    Each wave group contains precisely one student's next independent batch.
    """
    if not curricula or not adapter.trainable or microbatch_tokens < 1:
        raise ValueError("supervision needs trainable students and a positive token budget")
    starts = tuple(start_payloads) if start_payloads is not None else (None,) * len(curricula)
    if len(starts) != len(curricula):
        raise ValueError("one initial payload is required per student")
    if probe_rows is not None and len(probe_rows) != len(curricula):
        raise ValueError("one probe row set is required per student")
    bank = {}
    steps = []
    probe_at: dict[int, list[dict]] = {}
    for index, curriculum in enumerate(curricula):
        name, route = student_name(index), student_route(index)
        bank[name] = replace(adapter, route=route, init={**adapter.init, "seed": seed})
        student_steps = []
        probes = []
        if probe_rows is not None:
            for row in probe_rows[index]:
                trajectory_from_row(row)
                probes.append({**row, "turns": [
                    {**turn, "turn_extras": {**turn.get("turn_extras", {}), "route": route}}
                    for turn in row["turns"]]})
            probe_at.setdefault(0, []).extend(probes)
        for stage_index, (rows, stage) in enumerate(zip(curriculum.datasets, curriculum.stages, strict=True)):
            uri = put(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                               for row in rows))
            count = stage.epochs * math.ceil(len(rows) / stage.batch)
            step = 0
            for epoch in range(stage.epochs):
                order = list(range(len(rows)))
                random.Random(f"fit:fork:{stage_index}:{epoch}").shuffle(order)
                for offset in range(0, len(order), stage.batch):
                    leaves = tuple(Replay(f"{uri}#{row}", role=route)
                                   for row in order[offset:offset + stage.batch])
                    scale = stage.lr / BASE_LR
                    if stage.decay == "linear":
                        scale *= (count - step) / count
                    student_steps.append((GroupPlan(name, leaves), scale))
                    step += 1
            if probes:
                probe_at.setdefault(len(student_steps), []).extend(probes)
        steps.append(student_steps)
    waves, scales = [], []
    for update in range(max(map(len, steps))):
        active = [(index, student[update]) for index, student in enumerate(steps)
                  if update < len(student)]
        waves.append(WavePlan(tuple(item[0] for _, item in active)))
        scales.append({student_name(index): item[1] for index, item in active})
    initial = {student_name(index): base64.b64encode(payload).decode("ascii")
               for index, payload in enumerate(starts) if payload is not None}
    # This is the existing Emitted wire shape, not a second checkpoint codec.
    init = (WarmStart(put(json.dumps({"adapters": initial, "optim": {}},
                                  sort_keys=True).encode())) if initial else None)
    probes = tuple(ProbeSpec(update, put(b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows))) for update, rows in sorted(probe_at.items())) or None
    optimizer = replace(optim, lr=BASE_LR) if optim is not None else OptimSpec("adamw", BASE_LR)
    return ExperimentSpec(
        policy=PolicySpec(base, bank), gen=None,
        plans=Plans(train=put(encode(RunPlan(tuple(waves))))),
        algo=AlgoSpec("independent_sft", ("sft_document_weights",), optimizer,
                      Schedule(microbatch_tokens=microbatch_tokens, max_policy_lag=0,
                               lr_scales=tuple(scales), probes=probes)),
        topology=Topology(hosts=(HostSpec((learner(),)),)),
        seeds=Seeds(master=seed), init=init)


def supervised_spec(base: str, adapter: AdapterSpec,
                    datasets: Sequence[Sequence[Mapping[str, Any]]],
                    stages: SftStage | Sequence[SftStage], seed: int,
                    put: Callable[[bytes], str], **kwargs) -> ExperimentSpec:
    """Common single-corpus recipe, with one stage shared or chosen per student."""
    recipes = (stages,) * len(datasets) if isinstance(stages, SftStage) else tuple(stages)
    if len(recipes) != len(datasets):
        raise ValueError("one SFT stage is required per student")
    curricula = tuple(SftCurriculum((tuple(rows),), (stage,))
                      for rows, stage in zip(datasets, recipes, strict=True))
    return curriculum_spec(base, adapter, curricula, seed, put, **kwargs)


async def supervised_run(client: Training, adapter: AdapterSpec,
                         datasets: Sequence[Sequence[Mapping[str, Any]]],
                         stages: SftStage | Sequence[SftStage], *, seed: int,
                         checkpoint_every: int,
                         start_payloads: Sequence[bytes | None] | None = None) -> TrainingResult:
    """Materialize a compiled Replay spec and let the ordinary Trainer execute it."""
    pending: dict[str, bytes] = {}
    def put(payload: bytes) -> str:
        uri = "cas://" + hashlib.sha256(payload).hexdigest()
        pending[uri] = payload
        return uri
    spec = supervised_spec(client.base, adapter, datasets, stages, seed, put,
                           start_payloads=start_payloads,
                           microbatch_tokens=client.microbatch_tokens)
    for expected, payload in pending.items():
        if await client.put(payload) != expected:
            raise RuntimeError("supervised CAS content identity differs from stored bytes")
    return await client.run(spec, checkpoint_every=checkpoint_every)
