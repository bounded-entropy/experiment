"""Shared fixtures for the data-path and runner tests."""

from __future__ import annotations

import json
import random
from typing import Any

from rlstack import (
    AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
    LocalStore, Message, OptimSpec, PolicySpec, Role, Rollout, TrajectorySource,
    Schedule, Seeds, Task, Trajectory, Turn, engines, gpus, learner, lora,
)


def arith_task_bytes(n: int, seed: int) -> bytes:
    """n two-digit-sum tasks as a jsonl blob for the cas."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randrange(10, 99), rng.randrange(10, 99)
        rows.append({"id": f"arith-{i:04d}", "prompt": f"What is {a}+{b}?",
                     "meta": {"answer": a + b}})
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


def arith_spec(train_uri: str, heldout_uri: str | None = None,
               **overrides: Any) -> ExperimentSpec:
    """A small live GRPO spec over the arithmetic tasks."""
    fields: dict[str, Any] = dict(
        policy=PolicySpec(base="Qwen/Qwen3-0.6B",
                          bank={"pi": lora("layers.0-3.self_attn.*", r=16)}),
        gen=GenSpec(env="math_single_turn", tasks=train_uri),
        trajectories=TrajectorySource("live"),
        algo=AlgoSpec(loss="grpo", post=("verifier", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-5),
                      schedule=Schedule(group_size=2, trajectories_per_wave=4,
                                        n_updates=4, microbatch_tokens=64)),
        eval=(EvalSpec(tasks=heldout_uri, every=2, n_samples=2,
                       post=("verifier",))
              if heldout_uri else None),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (engines("main"), learner())),)),
        seeds=Seeds(master=17),
    )
    fields.update(overrides)
    return ExperimentSpec(**fields)


def arith_store(root: str | Any) -> tuple[LocalStore, str, str]:
    """A store seeded with train + held-out arithmetic tasks."""
    store = LocalStore(root)
    return (store,
            store.cas_put(arith_task_bytes(16, seed=0)),
            store.cas_put(arith_task_bytes(8, seed=1)))


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



