"""Qualify the shared loop on fake metal: python3.13 examples/arith_fake.py.

The demo owns its plain interaction and reward. All seeds, sealing, updates
and recovery use the normal framework. This script allocates no real metal.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlstack import (
    AlgoSpec, ExperimentSpec, FakeEngine, FakeLearner, GenSpec, GroupPlan,
    HostSpec, LocalStore, OptimSpec, Plans, PolicySpec, RunPlan, Sample,
    Schedule, Seeds, Topology, WavePlan, WaveRef, encode, fake_qwen_schema,
    learner, lora, pool, run_experiment,
)
from rlstack.runner.checkpointing import Checkpointing

FUNCTION = "arith_interaction:answer"


def arith_tasks(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randrange(10, 99), rng.randrange(10, 99)
        rows.append({"id": f"arith-{i:04d}", "prompt": f"What is {a}+{b}?",
                     "meta": {"answer": a + b}})
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        store = LocalStore(root)
        tasks = store.cas_put(arith_tasks(16, seed=0))
        rollout = RunPlan(tuple(WavePlan(tuple(
            GroupPlan(f"arith-{i:04d}", tuple(Sample(f"arith-{i:04d}", FUNCTION)
                                            for _ in range(4)))
            for i in range(2))) for _ in range(4)))
        train = RunPlan(tuple(WaveRef(f"self://rollouts/{update}") for update in range(1, 5)))
        spec = ExperimentSpec(
            code=("arith_interaction",),
            policy=PolicySpec(base="Qwen/Qwen3-0.6B",
                              bank={"pi": lora("layers.0-3.self_attn.*", r=16)}),
            gen=GenSpec((FUNCTION,), (tasks,)),
            plans=Plans(train=store.cas_put(encode(train)),
                        rollout=store.cas_put(encode(rollout))),
            algo=AlgoSpec(loss="grpo", post=("demo_arithmetic_reward", "grpo_advantage"),
                          optim=OptimSpec("adamw", lr=1e-5),
                          schedule=Schedule(microbatch_tokens=256)),
            topology=Topology((HostSpec((pool("main"),)), HostSpec((learner(),)))),
            seeds=Seeds(master=17),
        )
        schema = fake_qwen_schema(4, base=spec.policy.base)
        cadence = Checkpointing(every=2)  # fake recovery loses at most two updates
        report = run_experiment(spec, schema, store, FakeEngine(), FakeLearner(),
                                checkpointing=cadence)
        again = run_experiment(spec, schema, store, FakeEngine(), FakeLearner(),
                               checkpointing=cadence, resume=True)
        print(f"run_id={report.run_id} train={report.completed}")
        print(f"resume completed={again.completed} resumed_from={again.resumed_from}")


if __name__ == "__main__":
    main()
