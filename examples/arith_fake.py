"""Example 1, executable today: GRPO over synthetic arithmetic on fake metal.

    $ python3 examples/arith_fake.py

Runs the full choreography — identity, waves, seal, advantage, pack, commit
protocol, bundle sync, eval — against FakeEngine/FakeLearner, then attaches a
second time to show the run is already complete. Rerunning the script is a
no-op: same spec, same code, same data → same run_id.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlstack import (
    AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, HostSpec, LocalStore,
    OptimSpec, PolicySpec, Schedule, Seeds, TrajectorySource,
    fake_qwen_schema, learner, lora, pool,
)
from rlstack.runner.fakes import FakeEngine, FakeLearner
from rlstack.runner.loop import run_experiment


def arith_tasks(n: int, seed: int) -> bytes:
    """n single-digit-sum tasks as a jsonl blob for the cas."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randrange(10, 99), rng.randrange(10, 99)
        rows.append({"id": f"arith-{i:04d}", "prompt": f"What is {a}+{b}?",
                     "meta": {"answer": a + b}})
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


def main() -> None:
    store = LocalStore(Path(__file__).resolve().parent / "store")
    train_uri = store.cas_put(arith_tasks(64, seed=0))
    heldout_uri = store.cas_put(arith_tasks(16, seed=1))

    spec = ExperimentSpec(
        policy=PolicySpec(base="Qwen/Qwen3-0.6B",
                          bank={"pi": lora("layers.0-3.self_attn.*", r=16)}),
        gen=GenSpec(env="math_single_turn", tasks=train_uri),
        trajectories=TrajectorySource("live"),
        algo=AlgoSpec(
            loss="grpo", post=("verifier", "grpo_advantage"),
            optim=OptimSpec("adamw", lr=1e-5),
            schedule=Schedule(group_size=4, trajectories_per_wave=16, n_updates=4,
                              microbatch_tokens=256),
        ),
        eval=EvalSpec(tasks=heldout_uri, every=2, post=("verifier",)),
        gpu_config=GpuConfig(hosts=(HostSpec((pool("main"),)),
                                    HostSpec((learner(),)))),
        seeds=Seeds(master=17),
    )

    schema = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
    report = run_experiment(spec, schema, store, FakeEngine(), FakeLearner())
    print(f"run_id={report.run_id}  {report.extent}={report.completed}  "
          f"resumed_from={report.resumed_from}")

    again = run_experiment(spec, schema, store, FakeEngine(), FakeLearner())
    print(f"resubmit attached at update {again.resumed_from} (already complete)")

    run = store.open_run(report.run_id)
    for entry in run.read_ledger():
        print(f"  update {entry['update']}: "
              f"mean reward {entry['post']['reward']:.3f}  "
              f"bundle {entry['bundle_id']}")


if __name__ == "__main__":
    main()
