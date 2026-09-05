"""The fleet under the conditions the specs never tried, on the smallest metal
that can show each: TP and FSDP with every adapter type, a tenant joining a
learner mid-run, and bases that are not Qwen.

    modal run deploy/stress_fleet.py::smoke           # the image, exercised, before any deploy (ADR 0008)
    modal run deploy/stress_fleet.py::bases           # five bases on single L4s, in parallel, no desk
    modal deploy deploy/desk.py                       # THE desk, once, for every venue
    modal deploy deploy/stress_fleet.py               # one L4:2 metal (tp=2, fsdp=2)
    PYTHONUNBUFFERED=1 modal run deploy/stress_fleet.py::topology   # three tenants (lora / steer /
                                                      #   soft_prompt) through the desk, released
    PYTHONUNBUFFERED=1 modal run deploy/stress_fleet.py::latejoin   # A runs; B joins A's learner mid-run
    PYTHONUNBUFFERED=1 modal run deploy/stress_fleet.py::remote_learner  # UNRUN: anchored on main,
                                                      #   the learner on the other listing (ADR 0006 A)
    PYTHONUNBUFFERED=1 modal run deploy/stress_fleet.py::learner_sleep   # UNRUN: the fsdp=2 learner
                                                      #   hands both cards back and comes back (#82)
    modal run deploy/stress_fleet.py::down --call-id <id>
    modal run deploy/desk.py::status / ::sweep        # the fleet, at the desk

    RLSTACK_STRESS_GPU=L4 RLSTACK_STRESS_TP=1 RLSTACK_STRESS_FSDP=1 modal deploy ...   # the 1x1 shape
    RLSTACK_HF_SECRET=huggingface modal run deploy/stress_fleet.py::bases                # gated bases
        (after: modal secret create huggingface HF_TOKEN=hf_...)

Every door that acquires metal ends by releasing ITS OWN metal through the
desk and asserting that metal is freed (ADR 0004 promise 5, scoped by ADR
0007 Q2: one desk serves every venue, so the plane may hold someone else's
card). The releases are GUARDED — this venue never tears down a run that is
not its check's (Q6). Everything semantics-bearing is in the specs;
everything else here is the chassis' (I5).
"""

from __future__ import annotations

import json
import os
import time

import modal

from modal_venue import (
    a_store, cpu_image_for, desk, fleet, gpu_image_for, hf_cache, metal_class,
    metal_handle, progress_function, smoke_function, store_volume,
    submit_spec, take_down,
    wait_for_metal,
)

APP = "rlstack-stress-fleet"
app = modal.App(APP)

# The shape, as three flags read at deploy time on the client and inside the
# container: the card(s) the metal is deployed on, the engine's width, the
# learner's width.
GPU = os.environ.get("RLSTACK_STRESS_GPU", "L4:2")
TP = int(os.environ.get("RLSTACK_STRESS_TP", "2"))
FSDP = int(os.environ.get("RLSTACK_STRESS_FSDP", "2"))
HF_SECRET = os.environ.get("RLSTACK_HF_SECRET", "")
SECRETS = [modal.Secret.from_name(HF_SECRET)] if HF_SECRET else []

cpu_image = cpu_image_for()
gpu_image = gpu_image_for({"RLSTACK_STRESS_TP": str(TP),
                           "RLSTACK_STRESS_FSDP": str(FSDP)})

BASE = "Qwen/Qwen3-0.6B"        # the fleet's base
HIDDEN = 1024
TRAIN_TASKS = "cas://09499d32b51e5e1b2a644b1c65e01b44aa42ff1a5bfac78ead41f98f89f09c93"
EVAL_TASKS = "cas://82ae4626dbb59a2c50e2b13cbe7250c5f1ddd02dfb81edc7495efb77759d420b"
SCREENED_TASK = "dapo-math-17k/a6d38312-86c7-4022-b8d2-adcf19fa0c3a"

LORA_SITE = "layers.*.self_attn.*"
STEER_SITE = "resid_pre.8-20"
PROMPT_SITE, PROMPT_N = "prompt[:8]", 8
RANK = 16
STEER_LR = 1e-3
PROMPT_LR = 5e-3

UPDATES = 2
UPDATES_A = 4                   # the early tenant in the late-join check
GROUPS_PER_WAVE = 2
GROUP_SIZE = 8
MAX_TOKENS = 1024

MAIN_GB = 7.2                   # per device; the spec declares the TOTAL across shards
LEARNER_GB = 9.6

METAL = "stress-l4"
SUBDIR = "stress"
IDLE_S = 90.0
MINE = (METAL,)                 # the metals THIS venue registers, and the only ones it releases

# The base matrix (single L4 each, no desk). Gated bases need RLSTACK_HF_SECRET.
BASES = (
    ("Qwen/Qwen3-0.6B", "the baseline every other check ran on"),
    ("allenai/OLMo-2-0425-1B", "OLMo 2, 1B — llama-shaped names, q/k norms, post-norms"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "a Llama architecture that is not Qwen"),
    ("EleutherAI/pythia-1b", "GPT-NeoX: gpt_neox.layers.N.attention — NOT llama-shaped"),
    ("google/gemma-3-1b-it", "Gemma 3, 1B — gated on the Hub"),
)


def proposed_recipe():
    """WHAT THIS METAL IS FOR, PROPOSED (ADR 0007, Q4): an engine that SERVES
    all three adapter types this file stresses, at eager mode, and a plain
    learner. The desk journals it as its own declaration; `deploy/desk.py::
    recipe` can overwrite it, and whichever came last rides the carve."""
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    return Builds(
        engine=EngineBuild(max_model_len=2048, max_bundles=8, max_rank=RANK,
                           serves=("lora", "steer", "soft_prompt"),
                           enforce_eager=True),
        learner=LearnerBuild())


MetalS = metal_class(app, APP, METAL, GPU, gpu_image, module=__name__,
                     idle_s=IDLE_S, recipe=proposed_recipe(), secrets=SECRETS)


smoke = smoke_function(
    app, gpu_image, module=__name__,
    imports=("rlstack.policy.adapters.plora", "rlstack.policy.adapters.lora",
             "rlstack.runner.learners.fsdp_torch",
             "rlstack.runner.engines.vllm_engine"))
"""THE IMAGE, EXERCISED BEFORE ANY DEPLOY (ADR 0008, F5): the science this
venue serves, imported inside the container it will be served from, on no
metal and in seconds. A build is a declaration until something runs in it."""


# ---------------------------------------------------------------------------
# the science: three banks, one topology at (TP, FSDP), the same plans
# ---------------------------------------------------------------------------

def rollout_plan(task_ids, updates: int):
    from rlstack import GroupPlan, RunPlan, Sample, WavePlan

    def wave(u: int):
        chosen = [task_ids[(u * GROUPS_PER_WAVE + g) % len(task_ids)]
                  for g in range(GROUPS_PER_WAVE)]
        return WavePlan(tuple(
            GroupPlan(f"{task_id}#{g}", tuple(Sample(task_id, "dapo_math")
                                              for _ in range(GROUP_SIZE)))
            for g, task_id in enumerate(chosen)))
    return RunPlan(tuple(wave(u) for u in range(updates)))


def train_plan(updates: int):
    from rlstack import RunPlan, WaveRef
    return RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                         for u in range(1, updates + 1)))


def spec_for(store, bank: dict, overrides: dict, updates: int, master: int):
    """One experiment: the bank on the shared (TP, FSDP) topology. The
    spec declares GB TOTAL across shards, so the per-device need is what the
    metal books (ADR 0001)."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, GenSpec, HostSpec, LearnerMember, OptimSpec,
        Plans, PolicySpec, PoolMember, SamplingSpec, Schedule, Seeds, Topology,
        encode,
    )
    from rlstack.data.tasks import load_tasks

    if not any(t.id == SCREENED_TASK for t in load_tasks(store, TRAIN_TASKS)):
        raise ValueError(f"{SCREENED_TASK!r} is not in {TRAIN_TASKS}")
    plans = Plans(train=store.cas_put(encode(train_plan(updates))),
                  rollout=store.cas_put(encode(rollout_plan([SCREENED_TASK],
                                                            updates))))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank=bank),
        gen=GenSpec(envs=("dapo_math",), tasks=(TRAIN_TASKS, EVAL_TASKS),
                    sampling=SamplingSpec(temperature=1.0, max_tokens=MAX_TOKENS)),
        plans=plans,
        algo=AlgoSpec(loss="grpo", post=("final_answer", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-5, overrides=overrides),
                      schedule=Schedule(microbatch_tokens=512, max_policy_lag=1)),
        topology=Topology(hosts=(
            HostSpec((PoolMember("main", tp=TP, vram_gb=MAIN_GB * TP),)),
            HostSpec((LearnerMember(fsdp=FSDP, vram_gb=LEARNER_GB * FSDP),)))),
        seeds=Seeds(master=master))


def the_banks() -> dict[str, tuple[dict, dict]]:
    """{name: (bank, optimizer overrides)} — one adapter type per tenant."""
    from rlstack import lora, soft_prompt, steer
    return {
        "lora": ({"pi": lora(LORA_SITE, r=RANK)}, {}),
        "steer": ({"nudge": steer(STEER_SITE, d=HIDDEN, init_std=0.02)},
                  {"nudge": {"lr": STEER_LR}}),
        "soft_prompt": ({"prefix": soft_prompt(PROMPT_SITE, n=PROMPT_N, d=HIDDEN)},
                        {"prefix": {"lr": PROMPT_LR}}),
    }


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def build_specs(names: list[str], updates: list[int], masters: list[int]) -> dict:
    from rlstack import canonical_json

    store = a_store()
    banks = the_banks()
    rows = {}
    for name, n, master in zip(names, updates, masters):
        bank, overrides = banks[name]
        rows[name] = json.loads(canonical_json(
            spec_for(store, bank, overrides, n, master)))
    store_volume.commit()
    return rows


ledgers = progress_function(app, cpu_image, module=__name__, name="ledgers",
                            tail=8)
"""Each run's committed updates and its train blocks — the chassis' one
extent reader."""


@app.function(image=cpu_image, timeout=300)
def host_status(address: str) -> dict:
    """A listed host's own status — its roster, its residents — over the wire."""
    from rlstack.runner.remote import RemoteHost, transport_for
    return RemoteHost(transport_for(address)).status()


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=300)
def learner_custody(host: str) -> list[dict]:
    """WHO IS ON A LEARNER, off that host's journal: the learner-attach /
    learner-detach rows a foreign frame writes (ADR 0006 Part A). A run
    anchored elsewhere leaves no tenancy in this host's roster, so this is
    the only place its custody of the training metal is recorded."""
    store_volume.reload()
    return [e for e in a_store().read_host_log(host)
            if str(e.get("event", "")).startswith("learner-")]


# ---------------------------------------------------------------------------
# the sharded learner's sleep (#82): the memory, the numbers, the door
# ---------------------------------------------------------------------------

def gpu_memory_used() -> list[int]:
    """Per-device memory in MiB as the DRIVER sees it, over every process on
    the card. That is the only vantage from which a FOLLOWER rank's shard is
    visible at all: ranks 1..n-1 are separate processes and nothing here can
    read their allocators, so the claim "the base left the devices" has to be
    made against the driver's books, not torch's."""
    import subprocess

    told = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    return [int(line) for line in told.stdout.split()]


def a_fixed_batch():
    """One microbatch that depends on nothing: fixed ids, every token
    trainable, behavior logprobs zero. No tokenizer and no engine, because
    what is under test is a move — the same tokens through the same frozen
    weights must be the same numbers before and after it."""
    from rlstack.data.flatten import TokenBatch

    ids = tuple(range(11, 11 + 64))
    return TokenBatch(token_ids=ids, loss_mask=(1,) * len(ids),
                      behavior_logprobs=(0.0,) * len(ids),
                      segment_ids=(0,) * len(ids), doc_starts=(0,))


def one_lora_tenant():
    """The Parameterization a learner needs and nothing more: one trainable
    lora entry on the resolved weighted sites, under `sft` — the one loss
    that reads logprobs alone, so a forward's number is a pure function of
    the weights the sleep moved."""
    from rlstack.policy.siteschema import hf_schema, resolve
    from rlstack.runner.interfaces import (
        EntryInstall, OptimSettings, Parameterization,
    )

    return Parameterization(
        base=BASE, loss="sft",
        entries=(EntryInstall(name="pi", adapter_type="lora",
                              init={"r": RANK, "seed": 82}, trainable=True,
                              sites=resolve(hf_schema(BASE).sites, LORA_SITE)),),
        optim=OptimSettings("adamw", lr=1e-5, betas=(0.9, 0.999),
                            weight_decay=0.0, overrides={}))


def rails_of(stats) -> dict:
    """The numbers a forward is compared on. `loss` and `logprob_gap` are
    pure functions of THIS forward (sft's gap is the mean |logprob| against
    behavior zeros), so they are what bit-identity is asserted on;
    `grad_norm` is not — grads ACCUMULATE across the two calls, and its
    roughly doubling is its own evidence that the tenant's gradients came
    back from host RAM rather than being quietly dropped."""
    return {"loss": stats.loss, "logprob_gap": stats.logprob_gap,
            "mean_ratio": stats.mean_ratio, "grad_norm": stats.grad_norm}


@app.function(image=gpu_image, gpu=GPU,
              volumes={"/store": store_volume, "/hf": hf_cache}, timeout=3600)
def sleep_probe() -> dict:
    """WRITTEN, UNRUN: can an fsdp=FSDP learner hand its devices back?

    ADR 0002 called a sharded offload "its own proof" and a chorus reported
    `sleeps: false`; #82 retires that non-promise on the argument that
    `FSDPModule._apply` already reshards and re-aliases FSDP's flat view of
    each shard, so the offload is a plain `Module.to`. Everything in that
    sentence is checkable off metal except the two things that matter: that
    the memory actually comes back off every rank's card, and that a forward
    is the same number after the round trip.

    TWO STAGES, because a learner is reached two ways. In-process is the
    CHORUS — rank 0 here, ranks 1..n-1 as its children — where sleep and wake
    are announced verbs and the driver's books say whether every rank put its
    shard down. Through a RESIDENT is the DOOR: the same learner as a child
    process, its hello reporting the PROBED `sleeps`, and `sleep`/`wake`
    entering as door frames the way a host's arbiter sends them.

    It takes no metal through the desk — it is a plain GPU function, like
    `probe_base` — so there is nothing on the plane to release; the entrypoint
    asserts the plane is empty anyway, because a door that leaves one standing
    is the failure ADR 0003 exists for.

    Samarth runs metal; this has never been run.
    """
    import torch

    from rlstack.runner.learners.fsdp_torch import probe_sharded_sleep

    report: dict = {"gpu": GPU, "fsdp": FSDP, "base": BASE,
                    "torch": torch.__version__,
                    "devices": torch.cuda.device_count()}
    probe = probe_sharded_sleep()
    report["probe"] = {"supported": probe.supported, "reason": probe.reason}
    print(f"[probe] {json.dumps(report['probe'])}", flush=True)
    if not probe.supported:
        raise SystemExit("this torch cannot offload a shard: nothing below "
                         "would be testing what it claims to test")

    report["chorus"] = sleep_the_chorus()
    report["door"] = sleep_through_the_door()
    print(json.dumps(report, indent=1), flush=True)
    return report


def sleep_the_chorus() -> dict:
    """STAGE ONE: the learner in this process, its followers beside it.

    What it asserts: the base is on the cards (a shard report), one forward
    lands, the sleep drops per-device memory to near the idle floor ON EVERY
    DEVICE (the followers included — that is the whole point of announcing
    the verb), the wake brings it back, and the SAME batch through the woken
    learner gives bit-identical loss and logprob_gap.
    """
    import asyncio

    import torch

    from rlstack.runner.learners.fsdp_torch import lead_fsdp_learner

    batch = a_fixed_batch()
    idle = gpu_memory_used()
    print(f"[memory] idle, before the learner: {idle} MiB", flush=True)

    learner = lead_fsdp_learner(FSDP, dtype=torch.bfloat16)
    told: dict = {"idle_mib": idle, "sleeps": learner.sleeps}
    try:
        learner.install("sleeper", one_lora_tenant())
        told["shard_report"] = learner.shard_report()
        told["loaded_mib"] = gpu_memory_used()
        print(f"[memory] base sharded and a tenant installed: "
              f"{told['loaded_mib']} MiB", flush=True)

        before = rails_of(learner.forward_backward("sleeper", batch))
        told["before"] = before

        asyncio.run(learner.sleep())
        told["asleep_mib"] = gpu_memory_used()
        told["returned_mib"] = [held - slept for held, slept
                                in zip(told["loaded_mib"], told["asleep_mib"])]
        told["still_held_over_idle_mib"] = [
            slept - free for slept, free in zip(told["asleep_mib"], idle)]
        print(f"[memory] asleep: {told['asleep_mib']} MiB "
              f"(returned {told['returned_mib']}, still held over idle "
              f"{told['still_held_over_idle_mib']})", flush=True)

        asyncio.run(learner.wake())
        told["awake_mib"] = gpu_memory_used()
        after = rails_of(learner.forward_backward("sleeper", batch))
        told["after"] = after
        told["loss_delta"] = abs(after["loss"] - before["loss"])
        told["gap_delta"] = abs(after["logprob_gap"] - before["logprob_gap"])
        told["grad_norm_ratio"] = (after["grad_norm"] / before["grad_norm"]
                                   if before["grad_norm"] else None)
        print(f"[numbers] before {json.dumps(before)} after {json.dumps(after)}",
              flush=True)

        if not (told["loss_delta"] == 0.0 and told["gap_delta"] == 0.0):
            raise SystemExit(
                f"the forward moved across the cycle: loss delta "
                f"{told['loss_delta']}, gap delta {told['gap_delta']} — an "
                f"offload may cost PCIe time and nothing else")
        if any(returned <= 0 for returned in told["returned_mib"]):
            raise SystemExit(
                f"a device gave nothing back: {told['returned_mib']} MiB — a "
                f"rank held its shard through the sleep (#82)")
    finally:
        learner.uninstall("sleeper")
        learner.stop()
    return told


def sleep_through_the_door() -> dict:
    """STAGE TWO: the same learner as a RESIDENT, driven through its door.

    What it asserts that stage one cannot: the hello carries the PROBED
    `sleeps` (and an empty `sleep_refusal`), so a host would wire the
    alternation hooks; `sleep` and `wake` cross as door frames, which is
    exactly how an arbiter's evict/wake reach a child; and the learner verbs
    that bracket them go through `RemoteLearner`, the proxy a Host holds.
    """
    import asyncio

    from rlstack.runner.host import Partition, Regime
    from rlstack.runner.remote import RemoteLearner
    from rlstack.runner.residents import LearnerBuild, Resident, ResidentBirth

    batch = a_fixed_batch()
    birth = ResidentBirth(
        label="sleep-probe:learner",
        partition=Partition(METAL, tuple(range(FSDP)), 0.9, GPU.split(":")[0]),
        regime=Regime("learner", "training", BASE, FSDP),
        build=LearnerBuild(), store=a_store().address())
    resident = Resident.spawn(birth)
    told: dict = {"hello": resident.hello}
    print(f"[door] hello {json.dumps(resident.hello)}", flush=True)
    try:
        if not resident.hello.get("sleeps"):
            raise SystemExit(
                f"the resident says it cannot sleep: "
                f"{resident.hello.get('sleep_refusal')!r} — no host would wire "
                f"the alternation hooks, which is #82's whole claim")
        learner = RemoteLearner(resident.transport, fsdp=FSDP)
        learner.install("sleeper", one_lora_tenant())
        told["loaded_mib"] = gpu_memory_used()
        before = rails_of(learner.forward_backward("sleeper", batch))

        async def a_switch() -> None:
            await resident.sleep()
            told["asleep_mib"] = gpu_memory_used()
            await resident.wake()

        asyncio.run(a_switch())
        told["returned_mib"] = [held - slept for held, slept
                                in zip(told["loaded_mib"], told["asleep_mib"])]
        after = rails_of(learner.forward_backward("sleeper", batch))
        told["before"], told["after"] = before, after
        told["loss_delta"] = abs(after["loss"] - before["loss"])
        print(f"[door] returned {told['returned_mib']} MiB, loss delta "
              f"{told['loss_delta']}", flush=True)
        if told["loss_delta"] != 0.0:
            raise SystemExit("the forward moved across a door-driven cycle")
        learner.uninstall("sleeper")
    finally:
        teardown = resident.stop()
        told["teardown_graceful"] = teardown.graceful
    return told


@app.local_entrypoint()
def learner_sleep() -> None:
    """Can a SHARDED learner hand its devices back (#82)? — WRITTEN, UNRUN.

    Needs the fsdp=2 shape: `RLSTACK_STRESS_GPU=L4:2 RLSTACK_STRESS_FSDP=2`,
    which is this venue's default. It takes NO metal through the desk — the
    probe runs on its own GPU function — so it releases nothing: there is
    nothing of this door's to release, and under one desk a blanket sweep
    would be someone else's outage (ADR 0007, Q6).
    """
    print(json.dumps(sleep_probe.remote(), indent=1), flush=True)


# ---------------------------------------------------------------------------
# the bases: one L4 each, no desk — schema, the gate, parity, a generation
# ---------------------------------------------------------------------------

@app.function(image=gpu_image, gpu="L4", volumes={"/hf": hf_cache},
              timeout=1800, secrets=SECRETS)
def probe_base(base: str) -> dict:
    """Everything a base has to survive before a run: the schema compiles and
    resolves the sites the specs use; Phase 0 accepts a lora+steer bank; the
    engine builds with lora and steer served; zero steer is the base bit for
    bit on both sides; lora and steer each agree across the bridge; a
    windowed generation records its window. Each stage records its own
    failure and the next stages are skipped, so a base that cannot even
    compile a schema says so in one line."""
    import asyncio
    import traceback

    out: dict = {"base": base, "stages": {}}

    def stage(name: str, fn):
        try:
            out["stages"][name] = fn()
            return True
        except Exception as failure:
            out["stages"][name] = {
                "FAILED": f"{type(failure).__name__}: {str(failure)[:300]}",
                "where": traceback.format_exc().strip().splitlines()[-3:]}
            return False

    def config():
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base)
        out["hidden"] = int(getattr(cfg, "hidden_size", 0) or 0)
        return {"arch": list(getattr(cfg, "architectures", []) or []),
                "hidden": out["hidden"],
                "layers": int(getattr(cfg, "num_hidden_layers", 0) or 0)}

    if not stage("config", config):
        return out

    def schema():
        from rlstack.policy.siteschema import hf_schema, resolve
        s = hf_schema(base)
        weighted = resolve(s.sites, LORA_SITE)
        boundaries = resolve(s.sites, STEER_SITE)
        out["_schema"] = s
        got = {"sites": len(s.sites), "lora_sites": len(weighted),
               "steer_sites": len(boundaries),
               "first_weighted_path": weighted[0].path if weighted else None}
        if not weighted or not boundaries:
            raise ValueError(f"the schema names no site the specs use: {got}")
        return got

    if not stage("schema", schema):
        return out

    def gate():
        from rlstack import (
            AlgoSpec, ExperimentSpec, GenSpec, HostSpec, LearnerMember,
            OptimSpec, Plans, PolicySpec, PoolMember, Schedule, Seeds,
            Topology, lora, steer, validate,
        )
        spec = ExperimentSpec(
            policy=PolicySpec(base=base, bank={
                "pi": lora(LORA_SITE, r=RANK),
                "nudge": steer(STEER_SITE, d=out["hidden"])}),
            gen=GenSpec(envs=("dapo_math",), tasks=(TRAIN_TASKS,)),
            plans=Plans(train="cas://plan/train", rollout="cas://plan/roll"),
            algo=AlgoSpec(loss="grpo", post=("final_answer", "grpo_advantage"),
                          optim=OptimSpec("adamw", lr=1e-5), schedule=Schedule()),
            topology=Topology(hosts=(HostSpec((PoolMember("main"),)),
                                     HostSpec((LearnerMember(),)))),
            seeds=Seeds(master=1))
        issues = validate(spec, out["_schema"])
        if issues:
            raise ValueError("; ".join(f"[{i.code}] {i.path}" for i in issues))
        return {"issues": 0}

    if not stage("gate", gate):
        return out

    def parity():
        import torch

        from rlstack import Message, Role, SteerWindow
        from rlstack.data.flatten import TokenBatch
        from rlstack.policy.adapters import lora_torch, steer_torch
        from rlstack.policy.adapters.replay import ReplayRows, row_plan
        from rlstack.policy.adapters.steer import STEER_RECORD
        from rlstack.policy.compile import compile_bundle
        from rlstack.policy.siteschema import resolve
        from rlstack.runner.engines.vllm_engine import VllmEngine
        from rlstack.runner.learners.torch_learner import TorchLearner, _doc_spans
        from rlstack.spec.specs import SamplingSpec

        schema = out["_schema"]
        boundaries = resolve(schema.sites, STEER_SITE)
        weighted = resolve(schema.sites, LORA_SITE)
        engine = VllmEngine(base, gpu_memory_utilization=0.45, max_model_len=512,
                            max_rank=RANK, serves=("lora", "steer"))
        learner = TorchLearner()
        learner._ensure_base(base)
        model = learner._model
        hidden = out["hidden"]

        zero = steer_torch.build(boundaries, {"d": hidden, "init_std": 0.0})
        nudge = steer_torch.build(boundaries, {"d": hidden, "seed": 3,
                                               "init_std": 0.05})
        delta = lora_torch.build(weighted, {"r": RANK, "seed": 101})
        generator = torch.Generator().manual_seed(101)
        for path in delta.b:
            delta.b[path].data = torch.randn(*delta.b[path].shape,
                                             generator=generator) / (RANK * 40)
        steer_torch.install(model, zero)
        steer_torch.install(model, nudge)
        lora_torch.install(model, delta)

        def steer_slot(state):
            return {meta.path: state for meta in boundaries}
        lora_slot = {meta.path: delta for meta in weighted}
        banks = {"base": ({}, {}),
                 "zero": (steer_slot(zero), {"nudge": steer_torch.emit(zero)}),
                 "steer": (steer_slot(nudge), {"nudge": steer_torch.emit(nudge)}),
                 "lora": (lora_slot, {"pi": lora_torch.emit(delta)})}
        bundles = {}
        for name, (_, payloads) in banks.items():
            bundle = compile_bundle(payloads, {n: 0 for n in payloads},
                                    servable=payloads,
                                    adapter_types={"pi": "lora", "nudge": "steer"})
            engine.add_bundle(bundle)
            bundles[name] = bundle

        def replayed(slot, prompt, answer, window=(0, None)):
            context = list(engine.tokenize(prompt))
            ids = context + list(engine.tokenize(answer))
            batch = TokenBatch(token_ids=tuple(ids), loss_mask=(1,) * len(ids),
                               behavior_logprobs=(0.0,) * len(ids),
                               segment_ids=(0,) * len(ids), doc_starts=(0,))
            plan = ReplayRows(slots=(slot,),
                              index=torch.zeros(1, dtype=torch.long,
                                                device=learner.device),
                              facts=(({STEER_RECORD: list(window)},),))
            with torch.no_grad():
                with row_plan(model).route(plan):
                    got = learner._batched_logprobs(batch, _doc_spans(batch))
            return tuple(float(x) for x in got[len(context):])

        async def scored(name, prompt, answer, directives=()):
            return await engine.score_tokens([Message(Role.USER, prompt)],
                                             engine.tokenize(answer),
                                             bundles[name].bundle_id, directives)

        def gap(a, b):
            return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

        def worst(a, b):
            return max(abs(x - y) for x, y in zip(a, b))

        pairs = [("What is 17 + 26? Answer with the number only.", " 43"),
                 ("Name the largest planet in the solar system.",
                  " Jupiter is the largest.")]

        async def measure():
            got: dict = {}
            # list comprehensions, not generator expressions: a generator
            # expression holding an `await` is an async generator, which
            # max() cannot consume (found on the first matrix run)
            got["zero_engine_max_delta"] = max([
                worst(await scored("zero", p, a), await scored("base", p, a))
                for p, a in pairs])
            got["zero_trainer_max_delta"] = max([
                worst(replayed(banks["zero"][0], p, a), replayed({}, p, a))
                for p, a in pairs])
            for name in ("base", "lora", "steer"):
                got[f"{name}_gap"] = max([
                    gap(await scored(name, p, a), replayed(banks[name][0], p, a))
                    for p, a in pairs])
            prompt, answer = pairs[1]
            n = len(engine.tokenize(prompt))
            windowed = await scored("steer", prompt, answer,
                                    directives=(SteerWindow(start=n),))
            got["window_gap"] = gap(windowed, replayed(banks["steer"][0], prompt,
                                                       answer, window=(n, None)))
            events = [e async for e in engine.sample_tokens(
                [Message(Role.USER, prompt)],
                SamplingSpec(temperature=0.0, max_tokens=8), (),
                bundles["steer"].bundle_id, seed=1,
                directives=(SteerWindow(start=n),))]
            got["generated"] = "".join(e.text_delta for e in events[:-1])[:60]
            got["recorded_window"] = events[-1].turn_extras.get(STEER_RECORD)
            return got

        result = asyncio.run(measure())
        engine.shutdown()
        verdict = {
            "zero_is_the_base_both_sides": result["zero_engine_max_delta"] == 0.0
            and result["zero_trainer_max_delta"] == 0.0,
            "lora_agrees": result["lora_gap"] < 0.15,
            "steer_agrees": result["steer_gap"] < 0.15,
            "window_agrees": result["window_gap"] < 0.15,
            "window_recorded": result["recorded_window"] == [n_of(engine, pairs[1][0]), None],
        }
        return {**{k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in result.items()}, "verdict": verdict}

    def n_of(engine, prompt):
        return len(engine.tokenize(prompt))

    stage("parity", parity)
    out.pop("_schema", None)
    return out


@app.local_entrypoint()
def bases(only: str = "") -> None:
    """Every base in the matrix, one L4 each, in parallel. A base that fails
    a stage reports the stage; the matrix never aborts on one base."""
    chosen = [(b, note) for b, note in BASES if not only or only in b]
    calls = {b: probe_base.spawn(b) for b, _ in chosen}
    results = {}
    for b, _ in chosen:
        try:
            results[b] = calls[b].get(timeout=1800)
        except Exception as failure:
            results[b] = {"base": b, "stages": {"call": {"FAILED": str(failure)[:300]}}}
    for b, note in chosen:
        r = results[b]
        print(f"\n=== {b}  ({note})")
        for stage_name, got in r["stages"].items():
            if isinstance(got, dict) and "FAILED" in got:
                print(f"  {stage_name}: FAILED — {got['FAILED']}")
                for line in got.get("where", []):
                    print(f"      {line[:160]}")
            elif stage_name == "parity":
                print(f"  parity: verdict {json.dumps(got['verdict'])}")
                print(f"          gaps base/lora/steer/window "
                      f"{got['base_gap']}/{got['lora_gap']}/{got['steer_gap']}/"
                      f"{got['window_gap']}; zero max|Δ| engine "
                      f"{got['zero_engine_max_delta']} trainer "
                      f"{got['zero_trainer_max_delta']}")
                print(f"          generated {got['generated']!r}, "
                      f"recorded {got['recorded_window']}")
            else:
                print(f"  {stage_name}: {json.dumps(got)}")
    print()
    print(json.dumps({b: {s: ("FAILED" if isinstance(g, dict) and "FAILED" in g
                              else ("ok" if s != "parity" else g["verdict"]))
                          for s, g in r["stages"].items()}
                      for b, r in results.items()}, indent=1))


# ---------------------------------------------------------------------------
# the fleet doors
# ---------------------------------------------------------------------------

def submit(name: str, row: dict, anchor: str | None = None) -> dict:
    """One spec through THE desk, named in the log line."""
    print(f"[submit] {name}:", flush=True)
    return submit_spec(row, SUBDIR, anchor)


def await_runs(runs: dict[str, str], targets: dict[str, int],
               timeout_s: float = 3600.0, until_any: bool = False) -> dict:
    """Poll the ledgers until every run reaches its target (or, with
    until_any, until any run has committed at least one update)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        progress = ledgers.remote(list(runs.values()))
        line = {name: progress[rid]["committed"] for name, rid in runs.items()}
        print(f"[ledger] {json.dumps(line)}", flush=True)
        if until_any and any(n >= 1 for n in line.values()):
            return progress
        if all(line[name] >= targets[name] for name in runs):
            return progress
        time.sleep(30)
    raise SystemExit("the runs did not finish within the deadline")


def report_rails(runs: dict[str, str], progress: dict) -> dict:
    rails = {}
    for name, rid in runs.items():
        gaps = [round(t.get("logprob_gap", -1.0), 4) for t in progress[rid]["train"]]
        losses = [round(t.get("loss", 0.0), 5) for t in progress[rid]["train"]]
        print(f"[{name}] {rid}: logprob_gap per update {gaps}, loss {losses}",
              flush=True)
        rails[name] = {"run_id": rid, "logprob_gap": gaps, "loss": losses}
    return rails


@app.local_entrypoint()
def topology(master: int = 41) -> None:
    """Three tenants — lora, steer, soft_prompt — through the desk onto ONE
    tp=TP serving host and ONE fsdp=FSDP learner, two updates each, then the
    release. What this shows: every adapter type's rollout lowering on a
    sharded engine and its replay lowering on a sharded learner, three
    tenants deep."""
    call = metal_handle(APP).serve.spawn()
    print(f"[topology] {METAL} serving: call {call.object_id} "
          f"(gpu {GPU}, tp {TP}, fsdp {FSDP})", flush=True)
    try:
        print(json.dumps(wait_for_metal(METAL), indent=1), flush=True)
        names = ["lora", "steer", "soft_prompt"]
        rows = build_specs.remote(names, [UPDATES] * 3,
                                  [master, master + 1, master + 2])
        runs, pools = {}, {}
        for name in names:
            reply = submit(name, rows[name])
            runs[name] = reply["run_id"]
            pools[name] = reply["pools"]
        joined = len({json.dumps(p, sort_keys=True) for p in pools.values()}) == 1
        print(f"[join] {'ONE serving host, ONE learner for all three' if joined else 'NOT joined: ' + json.dumps(pools)}",
              flush=True)
        progress = await_runs(runs, {n: UPDATES for n in names})
        rails = report_rails(runs, progress)
        print(json.dumps({"joined": joined, "tp": TP, "fsdp": FSDP,
                          "rails": rails}, indent=1), flush=True)
    finally:
        take_down(MINE, call, "topology check done")


@app.local_entrypoint()
def latejoin(master: int = 51) -> None:
    """A runs first (lora, UPDATES_A updates); once A has committed at
    least one update, B (steer) is submitted and JOINS A's serving host and
    learner mid-run. What this shows: additive install on a learner that is
    training, requests of two tenants batching on one engine, A's rails
    undisturbed by B's arrival, both plans finishing."""
    call = metal_handle(APP).serve.spawn()
    print(f"[latejoin] {METAL} serving: call {call.object_id}", flush=True)
    try:
        print(json.dumps(wait_for_metal(METAL), indent=1), flush=True)
        rows = build_specs.remote(["lora", "steer"], [UPDATES_A, UPDATES],
                                  [master, master + 1])
        a = submit("A=lora", rows["lora"])
        runs = {"A": a["run_id"]}
        progress = await_runs(runs, {"A": 1}, until_any=True)
        a_before = progress[runs["A"]]["committed"]
        print(f"[latejoin] A has committed {a_before}; B joins now", flush=True)
        b = submit("B=steer", rows["steer"])
        runs["B"] = b["run_id"]
        joined = a["pools"] == b["pools"]
        print(f"[join] {'B joined A: one serving host, one learner' if joined else 'NOT joined: ' + json.dumps([a['pools'], b['pools']])}",
              flush=True)
        learner_address = fleet()["listings"][a["pools"]["learner"]]["address"]
        roster = host_status.remote(learner_address).get("tenants", {})
        print(f"[roster] the learner host after B's arrival: "
              f"{json.dumps(roster)}", flush=True)
        progress = await_runs(runs, {"A": UPDATES_A, "B": UPDATES})
        rails = report_rails(runs, progress)
        print(json.dumps({"joined": joined, "a_committed_before_b": a_before,
                          "roster_after_b": roster, "rails": rails}, indent=1),
              flush=True)
    finally:
        take_down(MINE, call, "latejoin check done")


@app.local_entrypoint()
def remote_learner(master: int = 61) -> None:
    """THE RUNNER AWAY FROM ITS LEARNER (ADR 0006 Part A) — WRITTEN, UNRUN.

    One lora tenant submitted with `anchor="main"`, so the desk delivers the
    frame to the SERVING listing and threads the learner's address as a
    route: the Trainer runs beside the sampling host and every learner verb
    — install, one per microbatch, optim_step, emit, uninstall — crosses the
    host door to the fsdp=FSDP listing and is admitted at ITS arbiter.

    What it would show that fakes cannot: the wire cost of a TokenBatch per
    microbatch and an emit per update between two real containers, the
    admitted frames interleaving with that host's own alternation, and the
    custody rows on the learner host's journal for a run whose tenancy lives
    somewhere else. It ends like every other door here: release every metal
    through the desk and assert the plane is empty (ADR 0003).

    Samarth runs metal; this has never been run.
    """
    call = metal_handle(APP).serve.spawn()
    print(f"[remote_learner] {METAL} serving: call {call.object_id}", flush=True)
    try:
        print(json.dumps(wait_for_metal(METAL), indent=1), flush=True)
        rows = build_specs.remote(["lora"], [UPDATES], [master])
        reply = submit("anchored-on-main", rows["lora"], anchor="main")
        runs = {"anchored-on-main": reply["run_id"]}
        pools = reply["pools"]
        away = pools["main"] != pools["learner"]
        print(f"[anchor] frame at {reply['host']}, main on {pools['main']}, "
              f"learner on {pools['learner']} — "
              f"{'the runner is away from its learner' if away else 'ONE HOST: no wire exercised'}",
              flush=True)
        if reply["host"] != pools["main"]:
            raise SystemExit(f"anchored on {reply['host']}, not on main's host")
        progress = await_runs(runs, {"anchored-on-main": UPDATES})
        rails = report_rails(runs, progress)
        # the tenancy lives at the anchor; the learner host holds only custody
        anchor_roster = host_status.remote(
            fleet()["listings"][pools["main"]]["address"]
        ).get("tenants", {})
        custody = learner_custody.remote(pools["learner"])
        print(json.dumps({"away": away, "pools": pools,
                          "anchor_roster": anchor_roster,
                          "learner_custody": custody, "rails": rails},
                         indent=1), flush=True)
        if reply["run_id"] not in anchor_roster:
            raise SystemExit("the anchor host does not carry the tenancy")
        if [row["event"] for row in custody
                if row.get("run_id") == reply["run_id"]] != [
                    "learner-attach", "learner-detach"]:
            raise SystemExit("the learner host did not journal this tenancy's "
                             "two ends")
    finally:
        take_down(MINE, call, "remote-learner check done")


@app.local_entrypoint()
def down(call_id: str = "") -> None:
    """Hand THIS venue's metal back by hand and, given the keepalive's call
    id, watch it return."""
    take_down(MINE, modal.FunctionCall.from_id(call_id) if call_id else None,
              "released by hand")
