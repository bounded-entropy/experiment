"""Shared fixtures for the data-path and runner tests."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any


from rlstack import (
    AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup, GroupPlan,
    LocalStore, Message, OptimSpec, Plans, PolicySpec, Role, Rollout, RunPlan,
    Sample, Schedule, Seeds, Task, Trajectory, Turn, WavePlan, WaveRef, encode,
    gpus, learner, lora, pool,
)


TRAIN, HELDOUT = ("arith", 16, 0), ("held", 8, 1)


def arith_task_bytes(n: int, seed: int, prefix: str = "arith") -> bytes:
    """n two-digit-sum tasks as a jsonl blob for the cas.

    Ids carry the set's prefix because a plan's leaf names a task id ALONE:
    two sets sharing an id would make a leaf ambiguous, and load_task_sets
    refuses that rather than resolving it by set order."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randrange(10, 99), rng.randrange(10, 99)
        rows.append({"id": f"{prefix}-{i:04d}", "prompt": f"What is {a}+{b}?",
                     "meta": {"answer": a + b}})
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


def task_ids(spec: tuple[str, int, int]) -> list[str]:
    """The ids of one declared set, without reading it: the fixture generates
    them, so a spec can name them before any store exists."""
    prefix, n, _ = spec
    return [f"{prefix}-{i:04d}" for i in range(n)]


def cas_uri(data: bytes) -> str:
    """The uri `Store.cas_put` will return for these bytes — content addressing
    means the writer and the namer never have to meet."""
    return f"cas://{hashlib.sha256(data).hexdigest()}"


def grpo_plan(task_ids: list[str], *, groups: int, size: int, updates: int,
              env: str = "math_single_turn", seed: int = 0) -> RunPlan:
    """`updates` waves of `groups` groups of `size` samples — GRPO's shape,
    written out. What a helper would build; the plan itself is just data."""
    rng = random.Random(seed)
    return RunPlan(tuple(
        WavePlan(tuple(
            GroupPlan(task, tuple(Sample(task, env) for _ in range(size)))
            for task in rng.sample(task_ids, groups)))
        for _ in range(updates)))


def trains_on_rollouts(updates: int) -> RunPlan:
    """Update u trains on rollout u, whole — the plain on-policy pairing."""
    return RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                         for u in range(1, updates + 1)))


def arith_plan_blobs(with_eval: bool, updates: int = 4,
                     every: int = 2) -> dict[str, bytes]:
    """The plan bytes an arith run needs, keyed by kind. Pure: the same bytes
    every time, so `cas_uri` names them and `arith_store` writes them."""
    blobs = {
        "train": encode(trains_on_rollouts(updates)),
        "rollout": encode(grpo_plan(task_ids(TRAIN), groups=2, size=2,
                                    updates=updates)),
    }
    if with_eval:
        held = task_ids(HELDOUT)
        blobs["eval"] = encode(grpo_plan(held, groups=len(held), size=2,
                                         updates=updates // every))
    return blobs


def arith_plans(with_eval: bool, updates: int = 4, every: int = 2) -> Plans:
    """Those blobs' uris — the spec's view of the same three artifacts."""
    blobs = arith_plan_blobs(with_eval, updates, every)
    return Plans(**{kind: cas_uri(data) for kind, data in blobs.items()})


def arith_spec(train_uri: str, heldout_uri: str | None = None,
               plans: Plans | None = None,
               **overrides: Any) -> ExperimentSpec:
    """A small live GRPO spec over the arithmetic tasks."""
    fields: dict[str, Any] = dict(
        policy=PolicySpec(base="Qwen/Qwen3-0.6B",
                          bank={"pi": lora("layers.0-3.self_attn.*", r=16)}),
        gen=GenSpec(envs=("math_single_turn",),
                    tasks=(train_uri,) + ((heldout_uri,) if heldout_uri else ())),
        plans=plans or arith_plans(heldout_uri is not None),
        algo=AlgoSpec(loss="grpo", post=("verifier", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-5),
                      schedule=Schedule(microbatch_tokens=64)),
        eval=(EvalSpec(every=2, post=("verifier",))
              if heldout_uri else None),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), learner())),)),
        seeds=Seeds(master=17),
    )
    fields.update(overrides)
    return ExperimentSpec(**fields)


def arith_store(root: str | Any) -> tuple[LocalStore, str, str]:
    """A store seeded with train + held-out tasks AND the standard plans."""
    store = LocalStore(root)
    for blobs in (arith_plan_blobs(True), arith_plan_blobs(False)):
        for data in blobs.values():
            store.cas_put(data)
    return (store,
            store.cas_put(arith_task_bytes(TRAIN[1], TRAIN[2], TRAIN[0])),
            store.cas_put(arith_task_bytes(HELDOUT[1], HELDOUT[2], HELDOUT[0])))


def sealed(task_id: str, content: str = "4", answer: int = 4) -> Trajectory:
    """A minimal sealed trajectory: one prompt, one generated turn."""
    turn = make_turn(content, tuple(ord(c) for c in content))
    rollout = Rollout(
        task=Task(task_id, f"What is 2+2?", {"answer": answer}),
        messages=[Message(Role.USER, "What is 2+2?"), turn.message],
        turns=[turn],
    )
    return rollout.seal()


def make_turn(
    content: str,
    token_ids: tuple[int, ...],
    *,
    logprobs: tuple[float, ...] | None = None,
    finish: str = "eos",
    role: Role = Role.ASSISTANT,
    token_extras: dict[str, tuple] | None = None,
    turn_extras: dict[str, Any] | None = None,
) -> Turn:
    """A Turn with defaults for everything I6 records but a test does not care about."""
    return Turn(
        message=Message(role, content),
        token_ids=token_ids,
        behavior_logprobs=logprobs if logprobs is not None else tuple(-0.5 for _ in token_ids),
        finish=finish,
        stop_hit=None,
        bundle_id="bundle:abc123",
        policy_version={"pi": 3},
        seed=17,
        token_extras=token_extras or {},
        turn_extras=turn_extras or {},
    )


def char_tokenize(text: str) -> tuple[int, ...]:
    """Deterministic stand-in tokenizer: one token per character."""
    return tuple(ord(c) for c in text)



