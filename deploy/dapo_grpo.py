"""GRPO on DAPO-Math-17k: Qwen3-14B, LoRA, tp=2 inference beside fsdp=4 training.

    modal run deploy/dapo_grpo.py::shakeout      # 2 updates, the defaults below
    modal run deploy/dapo_grpo.py::full --go     # the 50-update campaign

Two containers and one wire: PolicyHost serves the sampling partition (its own
Host, its own arbiter, admission at the partition), and `run` drives the whole
experiment beside the learner, whose GPUs are local because the learner is
never remote. Everything semantics-bearing is in the spec below — the plans,
the bank, the loss — and everything else here is venue (I5).

The plans are written out rather than helper-built: waves x groups x leaves is
the run's shape, and this file is where a campaign states it.
"""

import random

import modal

app = modal.App("rlstack-dapo-grpo")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlstack-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
          "OMP_NUM_THREADS": "1",
          "HF_HOME": "/hf",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-14B"
# the task sets built by deploy/tasks_dapo.py (#60): 17,547 train / 370 held out,
# prompts chat-formatted with thinking OFF and DAPO's own "Answer: N" instruction
TRAIN_TASKS = "cas://09499d32b51e5e1b2a644b1c65e01b44aa42ff1a5bfac78ead41f98f89f09c93"
EVAL_TASKS = "cas://82ae4626dbb59a2c50e2b13cbe7250c5f1ddd02dfb81edc7495efb77759d420b"
STORE = "modal://rlstack-store"
GROUPS_PER_WAVE = 8          # tasks per update
GROUP_SIZE = 8               # completions per task: the advantage's baseline
MAX_TOKENS = 2048


# ---------------------------------------------------------------------------
# the science: three plans and one spec
# ---------------------------------------------------------------------------

def rollout_plan(task_ids, updates, seed):
    """What the generator samples: one group per drawn task, GROUP_SIZE deep.

    A group IS the advantage's baseline scope, so the group's members are the
    completions being compared — which is why GRPO's shape is legible here as
    literal nesting rather than as a pair of schedule knobs.
    """
    from rlstack import GroupPlan, RunPlan, Sample, WavePlan
    rng = random.Random(seed)
    return RunPlan(tuple(
        WavePlan(tuple(
            GroupPlan(task, tuple(Sample(task, "dapo_math")
                                  for _ in range(GROUP_SIZE)))
            for task in rng.sample(task_ids, GROUPS_PER_WAVE)))
        for _ in range(updates)))


def train_plan(updates):
    """Update u trains on rollout u, whole: plain on-policy GRPO. Cold-start
    SFT would be the same plan with Replay waves in front of these."""
    from rlstack import RunPlan, WaveRef
    return RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                         for u in range(1, updates + 1)))


def eval_plan(task_ids, updates, every):
    """One held-out wave at every eval point, empty elsewhere: eval is a plan,
    so "measure here, not there" is data rather than a flag."""
    from rlstack import GroupPlan, RunPlan, Sample, WavePlan
    measured = WavePlan(tuple(GroupPlan(task, (Sample(task, "dapo_math"),))
                              for task in task_ids))
    return RunPlan(tuple(measured if u % every == 0 else WavePlan(())
                         for u in range(1, updates + 1)))


def spec_for(store, train_tasks, eval_tasks, updates, master):
    """The experiment as one value."""
    from rlstack import (AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig,
                         GpuGroup, GpuSet, LearnerMember, OptimSpec, Plans,
                         PolicySpec, PoolMember, SamplingSpec, Schedule, Seeds,
                         encode, lora)
    from rlstack.data.tasks import load_tasks

    train_ids = [t.id for t in load_tasks(store, train_tasks)]
    eval_ids = [t.id for t in load_tasks(store, eval_tasks)][:32]
    plans = Plans(
        train=store.cas_put(encode(train_plan(updates))),
        rollout=store.cas_put(encode(rollout_plan(train_ids, updates, master))),
        eval=store.cas_put(encode(eval_plan(eval_ids, updates, every=10))))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE,
                          bank={"pi": lora("layers.*.self_attn.*", r=16)}),
        gen=GenSpec(envs=("dapo_math",), tasks=(train_tasks, eval_tasks),
                    sampling=SamplingSpec(temperature=1.0, max_tokens=MAX_TOKENS)),
        plans=plans,
        algo=AlgoSpec(loss="grpo", post=("final_answer", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-4),
                      # 512, not 2048: `pack` fills a forward TO this budget,
                      # so short documents alone never shrink one. Thinking-off
                      # completions are a few hundred tokens, which is what
                      # makes the knob bite again (#58 measured it stuck when a
                      # single document was longer than the budget).
                      schedule=Schedule(microbatch_tokens=512, max_policy_lag=1)),
        eval=EvalSpec(every=10, post=("final_answer",)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus=GpuSet(n=2), members=(PoolMember("main", tp=2),)),
            GpuGroup(gpus=GpuSet(n=2), members=(LearnerMember(fsdp=2),)))),
        seeds=Seeds(master=master))


# ---------------------------------------------------------------------------
# the venue: the serving partition, and the runner beside the learner
# ---------------------------------------------------------------------------

@app.cls(image=image, gpu="L4:2", volumes={"/store": store_volume, "/hf": hf_cache},
         timeout=14400, scaledown_window=300, max_containers=1)
@modal.concurrent(max_inputs=64)
class PolicyHost:
    """The sampling partition in its own container: born with its Partition and
    Regime, admitting its own traffic under its own arbiter."""

    tp: int = modal.parameter(default=2)

    @modal.enter()
    def bring_up(self) -> None:
        from rlstack import ModalVolumeStore
        from rlstack.runner.engines.vllm_engine import VllmEngine
        from rlstack.runner.host import Host, Partition, Regime
        from rlstack.runner.remote import HostService

        store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
        engine = VllmEngine(BASE, tp=self.tp, gpu_memory_utilization=0.90,
                            max_model_len=4096, max_bundles=8, max_rank=16)
        self.host = Host(
            "dapo-policy", engines=(engine,), learner=None, store=store,
            partition=Partition("modal-l4", tuple(range(self.tp)), 0.90, "L4"),
            regimes=(Regime(f"serve-tp{self.tp}", "inference", BASE, self.tp),))
        self.service = HostService(self.host)
        self.stats_task = None
        print(f"[host dapo-policy] up: {BASE} tp={self.tp}")

    @modal.method()
    async def call(self, verb: str, payload: dict) -> dict:
        import asyncio
        if self.stats_task is None:
            self.stats_task = asyncio.create_task(self.host.run_stats())
        return await self.service.serve(verb, payload)

    @modal.method()
    def ask(self, verb: str, payload: dict) -> dict:
        return self.service.answer(verb, payload)


class ModalTransport:
    """The client end: `call` awaits an admitted verb, `ask` is admission-free.
    Frames are JSON-safe dicts, which HostService already guarantees."""

    def __init__(self, partition: PolicyHost) -> None:
        self.partition = partition

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.partition.call.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return self.partition.ask.remote(verb, payload)


def _run(train_tasks: str, eval_tasks: str, updates: int, master: int,
         label: str) -> dict:
    """Drive the experiment from the learner's container: local learner,
    remote policy pool, one shared store on the volume."""
    import asyncio

    from rlstack import ModalVolumeStore
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.host import Host, Partition, Regime
    from rlstack.runner.learners.fsdp_torch import lead_fsdp_learner
    from rlstack.runner.remote import RemotePool

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    spec = spec_for(store, train_tasks, eval_tasks, updates, master)
    pool = RemotePool(ModalTransport(PolicyHost(tp=2)), base=BASE, tp=2)
    # rank 0 starts the chorus and then IS the learner; the followers exist
    # only to stand in its collectives, and stop() is what ends them
    learner = lead_fsdp_learner(2)
    host = Host("dapo-learner", engines=(), learner=learner, store=store,
                partition=Partition("modal-a100", (0, 1), 0.90, "A100-80GB"),
                regimes=(Regime("train-fsdp2", "training", BASE, 2),))
    print(f"[chorus] rank 0 of 2, learner.fsdp={learner.fsdp}")
    try:
        report = asyncio.run(host.submit(spec, hf_schema(BASE), store,
                                         remotes={"main": pool}))
    finally:
        learner.stop()
    store_volume.commit()
    # the measurement the L4 attempts could not survive to report: if the peak
    # is near one shard the blocks reshard as designed and the L4 was simply
    # too small; if it is near the whole base they accumulate, and that is ours
    import torch
    print(f"[{label}] rank0 peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print(f"[{label}] run_id={report.run_id} "
          f"updates={report.updates_completed}")
    return {"run_id": report.run_id, "updates": report.updates_completed}


@app.function(image=image, gpu="A100-80GB:2", volumes={"/store": store_volume, "/hf": hf_cache},
              timeout=7200)
def shakeout(train_tasks: str = TRAIN_TASKS, eval_tasks: str = EVAL_TASKS,
             master: int = 7) -> dict:
    """Two updates end to end: the wire, the chorus, one real gradient."""
    return _run(train_tasks, eval_tasks, updates=2, master=master,
                label="shakeout")


@app.function(image=image, gpu="A100-80GB:2", volumes={"/store": store_volume, "/hf": hf_cache},
              timeout=86400)
def full(train_tasks: str = TRAIN_TASKS, eval_tasks: str = EVAL_TASKS,
         master: int = 7, updates: int = 50, go: bool = False) -> dict:
    """The campaign. Gated: real money, and the shakeout's numbers decide."""
    if not go:
        raise SystemExit("refusing to spend: pass --go once the shakeout is read")
    return _run(train_tasks, eval_tasks, updates=updates, master=master,
                label="full")
