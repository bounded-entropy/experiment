"""The first real run: Qwen3-0.6B + LoRA + GRPO on one Modal L4.

    modal run deploy/modal_app.py::run_tests    # the whole suite, in-image
    modal run deploy/modal_app.py::run_arith    # the experiment, end to end

Everything semantics-bearing lives in rlstack; this file is DEPLOYMENT ONLY
(I5): it builds the image, mounts the volume, constructs the spec, and hands
real metal to run_experiment. The store is a ModalVolumeStore — the volume's
stage-then-commit semantics map onto the ledger commit point, so this run
doubles as the Store ABC's second-backend test.
"""

from __future__ import annotations

import modal

app = modal.App("rlstack")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)

# I7: pinned to the versions the first green run resolved and printed
# (2026-08-27, run 442bcf59b515). torch==2.13.0 is the default PyPI Linux
# wheel, which reports itself as 2.13.0+cu130.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    # debian_slim has no nvcc: FlashInfer's JIT sampling kernels cannot
    # build in-container, so force vLLM's native torch sampler instead
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("rlstack", "rlstack_engine")
    .add_local_dir("tests", remote_path="/root/tests")
)

BASE = "Qwen/Qwen3-0.6B"


def arith_tasks(n: int, seed: int) -> bytes:
    """Two-digit sums phrased for raw completion: the continuation after
    "The answer is" is where the verifier finds its last number."""
    import json
    import random

    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randrange(10, 99), rng.randrange(10, 99)
        rows.append({"id": f"arith-{i:04d}",
                     "prompt": f"What is {a}+{b}? The answer is",
                     "meta": {"answer": a + b}})
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


@app.function(image=image, gpu="L4", volumes={"/store": store_volume},
              timeout=3600)
def run_arith(n_updates: int = 4, trajectories_per_wave: int = 16,
              group_size: int = 4, lr: float = 1e-4, master_seed: int = 17):
    import torch
    import transformers
    import vllm

    from rlstack import (
        AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
        ModalVolumeStore, OptimSpec, PolicySpec, SamplingSpec, Schedule, Seeds,
        Host, TrajectorySource, gpus, learner, lora, pool,
    )
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.learners.torch_learner import TorchLearner

    print(f"[pins] vllm={vllm.__version__} torch={torch.__version__} "
          f"transformers={transformers.__version__}")

    store = ModalVolumeStore("/store", volume=store_volume)
    train = store.cas_put(arith_tasks(64, seed=0))
    heldout = store.cas_put(arith_tasks(16, seed=1))

    spec = ExperimentSpec(
        policy=PolicySpec(base=BASE,
                          bank={"pi": lora("layers.*.self_attn.*", r=16)}),
        gen=GenSpec(env="math_single_turn", tasks=train,
                    sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                          max_tokens=12)),
        trajectories=TrajectorySource("live"),
        algo=AlgoSpec(loss="grpo", post=("verifier", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=lr),
                      schedule=Schedule(group_size=group_size,
                                        trajectories_per_wave=trajectories_per_wave,
                                        n_updates=n_updates,
                                        microbatch_tokens=2048)),
        eval=EvalSpec(tasks=heldout, every=2, n_samples=2, post=("verifier",)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main", fraction=0.45),
                                 learner(fraction=0.40))),)),
        seeds=Seeds(master=master_seed),
    )

    import asyncio

    host = Host("l4-arith", engines=(
        VllmEngine(BASE, gpu_memory_utilization=0.45, max_model_len=512,
                   max_lora_rank=16),),
        learner=TorchLearner(), store=store)
    async def submit_with_stats():
        stats = asyncio.get_running_loop().create_task(host.run_stats(30.0))
        try:
            return await host.submit(spec, hf_schema(BASE))
        finally:
            stats.cancel()

    report = asyncio.run(submit_with_stats())
    store_volume.commit()   # persist anything staged after the last ledger line

    run = store.open_run(report.run_id)
    print(f"\nrun_id={report.run_id}  updates={report.updates_completed}  "
          f"resumed_from={report.resumed_from}")
    for entry in run.read_ledger():
        print(f"  update {entry['update']}: reward {entry['post']['reward']:.3f}  "
              f"loss {entry['train']['loss']:+.4f}  "
              f"gap {entry['train']['logprob_gap']:.4f}  "
              f"grad {entry['train']['grad_norm']:.3f}")
    return report.run_id


@app.function(image=image, volumes={"/store": store_volume}, timeout=300)
def hosts():
    """The hosts CLI against the volume:  modal run deploy/modal_app.py::hosts"""
    from rlstack import ModalVolumeStore
    from rlstack.__main__ import render_gpu, render_hosts, render_runs

    store = ModalVolumeStore("/store", volume=store_volume)
    for view in (render_hosts, render_runs, render_gpu):
        print(view([store]), end="")
        print("-" * 72)


@app.function(image=image, timeout=900)
def run_tests():
    """The full fakes suite inside the deploy image (CPU): proves the code
    that ships is the code that passes, before any GPU minute is spent."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "/root/tests",
         "-p", "test_*.py", "-v"],
        check=True)
