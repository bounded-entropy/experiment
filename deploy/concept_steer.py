"""A steering vector distilled from a prompt-conditioned teacher (ADR 0005).

    modal run deploy/concept_steer.py::prompts                   # the corpus -> cas uris
    modal deploy deploy/desk.py                                  # THE desk, once, for every venue
    modal deploy deploy/concept_steer.py                         # this venue's metal
    modal run deploy/desk.py::recipe --metal concept-a100 \
        --engine max_model_len=4096,max_bundles=32,max_rank=16,serves=steer,\
enforce_eager,enable_sleep_mode --learner checkpoint_activations
    modal run deploy/concept_steer.py::up                        # boot the metal
    modal run deploy/concept_steer.py::distill_set --train-tasks cas://<sha>
    modal run deploy/concept_steer.py::train --layer 10 --teacher-run <rid>
    modal run deploy/concept_steer.py::measure --run-id <rid> --heldout-tasks cas://<sha>
    modal run deploy/concept_steer.py::export --run-id <rid> --version 64
    modal run deploy/desk.py::status / ::sweep                   # the fleet, at the desk

NOTHING IN THIS FILE HAS RUN. It is written against ADR 0005's answered
questions and left for Samarth's metal: no number below has been observed, the
32B student's memory and pace are unmeasured, the eager-mode serving cost is
unmeasured, and the corpus builder's chat-template assertion is only checked
when `prompts` actually runs on the volume.

THE EXPERIMENT, in two runs and one measurement. `distill_set` submits a
GENERATION-ONLY run (ADR 0006 Part B): an EMPTY bank, so `main` is the bare
Qwen3-32B, sampling the `conditioned_teacher` environment over prompts whose
`meta["hint"]` is the happiness system block. Its sealed `rollouts/` ARE the
trajectory set and its run_id is the set's identity. `train --layer` then
submits ONE SFT arm per anchor boundary: the same base with one `d=5120`
steer at `resid_pre.<layer>`, `gen=None`, and a train plan of Replay leaves
naming that teacher run's rollouts. The three arms differ in the bank and in
nothing else. `measure` runs the distillation number from outside the run
(#70): the student samples held-out prompts and
`conditioned_teacher_logprobs` -> `reverse_kl` says how far it still is from
the teacher. `export` copies the vector's safetensors to the volume for the
paper's ICL harness, which is where the actual hypothesis is tested (Q10).

THE METAL (Q7): one A100-80GB:2, one HostSpec ALTERNATING `main` at tp=2 with
the learner at fsdp=2 — a 32B is ~32.5 GiB per device either way, so serving
and training cannot co-reside on 80 GB with a 1000-token activation budget
and take turns instead (which is why every training spec declares
`max_policy_lag=0`, and why an SFT run does not care). The engine is built
with `enable_sleep_mode`, which is what makes an alternating partition really
hand the device back. The serving build also pays the steer's demands (our
worker class, eager mode) because the gate holds the steer's boundary against
the main engine's inventory even for a run that samples nothing.

WHAT THE ALTERNATION BUYS, AND WHAT IT IS STILL WAITING ON. Since #82 a
sharded learner sleeps too: `FsdpTorchLearner.sleeps` is a PROBE of the
pinned torch, not `ranks.width == 1`, so at fsdp=2 the arbiter wires the
learner's evict/wake exactly as it wires the engine's and ONE resident is
live at a time. The partition then sizes for the unit's LARGEST member — the
number both members are already entitled to — instead of for their sum.

That holds ONCE THE METAL PROBE PASSES, and it has not been run:
`deploy/stress_fleet.py::learner_sleep` is what shows the memory actually
comes back off both cards and that a forward is bit-identical across the
cycle. Until it does, the co-residence numbers stand as the fallback — a 32B
is ~32.5 GiB per device either way, serving and training cannot co-reside on
80 GB with a 1000-token activation budget, and the answer if the learner's
sleep does not land is two HostSpecs on wider metal or an engine share small
enough to sit beside a resident learner.

Everything semantics-bearing is in the specs below — the banks, the loss, the
plans, the corpus (I5); everything else here is the chassis' (ADR 0007). THE
DOORS NEVER RELEASE: this is a campaign venue, and idle metal is the desk's
to collect (ADR 0003, ADR 0007 Q6).
"""

from __future__ import annotations

import json
import os

import modal

from modal_venue import (
    a_store, cpu_image_for, desk, export_blob, follow, gpu_image_for,
    hf_cache, metal_class, metal_handle, progress_function, run_suite,
    store_volume, submit_and_follow, wait_for_metal,
)

APP = "rlstack-concept-steer"
app = modal.App(APP)

# The card, as one flag. A 32B at tp=2/fsdp=2 needs ~80 GB per device pair;
# the list is the scheduler's choice of which of those frees first, and GB is
# what keeps that honest — 34 GB is 34 GB on either card (ADR 0001).
GPUS = [g for g in os.environ.get("RLSTACK_CONCEPT_GPU", "A100-80GB:2,H100:2"
                                  ).split(",") if g]
WIDTH = 2                       # tp for the pool, fsdp for the learner

cpu_image = cpu_image_for()
gpu_image = gpu_image_for(with_tests=True)

# the corpus builder's own layer, kept AFTER the pinned one so the pins stay
# cached with the campaign images (deploy/tasks_dapo.py's rule, #60)
tasks_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==5.16.1", "huggingface_hub", "safetensors",
                 "numpy")
    .pip_install("pyarrow", "jinja2")   # jinja2: apply_chat_template needs it, found on the venue
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("rlstack", "modal_venue")
)

BASE = "Qwen/Qwen3-32B"
HIDDEN = 5120                   # Qwen3-32B's width — a boundary has no shape
SUBDIR = "concept"

# The paper's own anchors for Qwen3-32B (10 / 32 / 54 of 64, Appendix B), and
# `resid_pre.<n>` is the stream LEAVING layer n — the output of
# model.layers[n], which is where the harness adds (Q8, confirmed).
ANCHORS = (10, 32, 54)
ENTRY = "v"                     # the bank's one name; `export` copies v@<version>
ALPHA = 0.1                     # nsteer: the injection STARTS at this fraction of ||h_t|| per token
LORA_RANK = 4                   # mlp_lora: a rank-4 delta on one layer's three MLP matrices
ADAPTERS = ("steer", "nsteer", "mlp_lora")
"""The arms' families: a free vector at the boundary, a norm-scaled direction
there (alpha trained, unit direction), and a small LoRA on the MLP of the
same layer — the paper's anchors, three ways of touching one layer."""
ALGOS = ("sft", "opd")
"""How an arm learns: SFT over the teacher's sealed set (ADR 0005), or ON-POLICY
distillation — the student samples its own rollouts and the conditioned
teacher scores those very tokens (opd: sampled-token reverse KL)."""

CONCEPT = "happiness"
SPLIT_SEED = 5                  # the split draw's seed; a task's split is h(this, id)

# The set: 64 waves of 32 trajectories, ~1024 tokens each (Q6, Q9)
WAVES, PER_WAVE, MAX_TOKENS = 64, 32, 1024

# COUNTS ARE DRAWN, NOT DEALT (split_tasks): the plan needs EXACTLY
# WAVES * PER_WAVE = 2048 prompts, and a draw of 2048/8850 lands within about
# +/-40 of that, so it would come up short half the time and refuse the
# build. The split therefore ASKS for headroom and the plan takes the first
# 2048 in set order, leaving the rest.
TRAIN_ASK, HELDOUT_ASK = 2304, 160
LR = 5e-3                       # a 5120-wide vector reaches useful norm in ~100 steps (Q9)
MICROBATCH_TOKENS = 4096

# The partition treaty on one 80 GB card, in GB TOTAL across a member's shards
# (ADR 0001) — the per-device need is this / WIDTH. A 32B in bf16 is ~32.5 GiB
# per device at either width; serving wants that plus KV and the learner wants
# it plus checkpointed activations, one 5120-wide vector and its moments. The
# two members ALTERNATE, so the unit's carve is the LARGER of them.
# UNMEASURED, and the shakeout's first job (ADR 0005 non-promises): no number
# here has been observed on metal.
MAIN_GB = 44.0 * WIDTH          # 44 GB/device: ~35 weights + ~9 KV at 4096 ctx
LEARNER_GB = 64.0 * WIDTH       # 64 GB/device — MEASURED on the venue: the
                                # learner rank sits at 46.3 GiB (shard, recompute
                                # state, the vector's moments) and the logprob
                                # pass over a 4096-token microbatch takes 8.2 GiB
                                # more in fp32 logits; 48 died there at update 1

METAL = "concept-a100"
IDLE_S = 90.0                   # the desk's clock (ADR 0003, cut to a minute and a half on 2026-09-05); the venue's scaledown is no shorter


def proposed_recipe():
    """WHAT THIS METAL IS FOR, PROPOSED (ADR 0007, Q4). The recipe is the
    desk's declaration now — `deploy/desk.py::recipe` is where an operator
    says it and the journal is where it lives — but a venue that knows what
    its metal exists to serve may say so at registration, and the desk
    journals the proposal as its own `recipe` event.

    It serves `steer`: those demands (our worker class, eager mode, the V1
    runner) are paid by the build and refused at construction if they cannot
    be. `enable_sleep_mode` is what makes the alternating partition really
    hand the device back, and the learner checkpoints activations because a
    32B shard's recompute is cheaper than its memory."""
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    return Builds(
        engine=EngineBuild(max_model_len=4096, max_bundles=32, max_rank=16,
                           serves=("steer", "nsteer", "lora"), enforce_eager=True,
                           enable_sleep_mode=True),
        learner=LearnerBuild(checkpoint_activations=True))


MetalS = metal_class(app, APP, METAL, GPUS, gpu_image, module=__name__,
                     idle_s=IDLE_S, recipe=proposed_recipe())


# ---------------------------------------------------------------------------
# the science: two specs and one measurement, written out (I5)
# ---------------------------------------------------------------------------

def topology():
    """ONE HOSTSPEC, TWO MEMBERS: `main` at tp=2 ALTERNATING with the learner
    at fsdp=2 on one partition (Q7). A 32B is ~32.5 GiB per device either
    way, so the two cannot co-reside on 80 GB with a 1000-token activation
    budget; alternating implies max_policy_lag=0, which an SFT run does not
    care about because nothing it trains on is its own.

    Since #82 BOTH members hand the device back — the sharded learner's sleep
    is a probed build fact, not a refusal — so the members' declared GB is the
    largest one's rather than their sum. Unproven on metal
    (`stress_fleet.py::learner_sleep`); the header says what the fallback is."""
    from rlstack import HostSpec, LearnerMember, PoolMember, Topology

    return Topology(hosts=(HostSpec((
        PoolMember("main", tp=WIDTH, vram_gb=MAIN_GB),
        LearnerMember(fsdp=WIDTH, vram_gb=LEARNER_GB))),))


def serving_topology():
    """The teacher's: `main` alone, because a generation-only run has no
    learner to alternate with (ADR 0006 Part B)."""
    from rlstack import HostSpec, PoolMember, Topology

    return Topology(hosts=(HostSpec((
        PoolMember("main", tp=WIDTH, vram_gb=MAIN_GB),)),))


def teacher_rollout_plan(task_ids):
    """WAVES waves of PER_WAVE prompts, each sampled ONCE under the
    conditioned teacher: a trajectory set is a set of prompts answered, not a
    group structure — nothing here computes an advantage."""
    from rlstack import GroupPlan, RunPlan, Sample, WavePlan

    def wave(u: int):
        return WavePlan(tuple(
            GroupPlan(task_id, (Sample(task_id, "conditioned_teacher"),))
            for task_id in task_ids[u * PER_WAVE:(u + 1) * PER_WAVE]))
    return RunPlan(tuple(wave(u) for u in range(WAVES)))


def sft_train_plan(teacher_run: str):
    """Update u trains on the teacher's rollout u, whole — one group per
    wave, this run's own key, the teacher's rows by ref (ADR 0006's
    `store://<run_id>/rollouts/<r>#<i>`)."""
    from rlstack import GroupPlan, Replay, RunPlan, WavePlan

    return RunPlan(tuple(
        WavePlan((GroupPlan(f"distill-{u}", tuple(
            Replay(f"store://{teacher_run}/rollouts/{u}#{i}")
            for i in range(PER_WAVE))),))
        for u in range(1, WAVES + 1)))


def teacher_spec(store, train_tasks: str):
    """THE TRAJECTORY SET AS A RUN (ADR 0005, Q4): an EMPTY bank, so `main`
    is the bare Qwen3-32B and the hint is the whole of the conditioning; no
    algo, so no Trainer, no ledger, and the extent is the rollout plan. Its
    sealed rollouts are the set and its run_id is the set's identity."""
    from rlstack import (
        ExperimentSpec, GenSpec, Plans, PolicySpec, SamplingSpec, Seeds,
        encode, load_tasks,
    )

    ids = [task.id for task in load_tasks(store, train_tasks)]
    if len(ids) < WAVES * PER_WAVE:
        raise ValueError(
            f"{train_tasks} holds {len(ids)} prompts; the plan wants "
            f"{WAVES * PER_WAVE}")
    plan = store.cas_put(encode(teacher_rollout_plan(ids)))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank={}),
        gen=GenSpec(envs=("conditioned_teacher",), tasks=(train_tasks,),
                    sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                          max_tokens=MAX_TOKENS)),
        plans=Plans(train=None, rollout=plan),
        algo=None,
        topology=serving_topology(),
        seeds=Seeds(master=5))


def student_spec(store, teacher_run: str, layer: int, adapter: str = "steer"):
    """ONE ARM: the same base with ONE steer at one anchor boundary, SFT over
    the teacher's rows, sampling nothing of its own. The arms share plan
    bytes and differ in the bank alone — `main` is declared and idle,
    because the gate holds the steer's boundary against its inventory.

    `adapter` picks the family: "steer", a free vector from zero (ADR 0005);
    "nsteer", a learned DIRECTION injected at ALPHA times each token's own
    residual norm (the paper's calibration), starting from a seeded random
    direction because a norm-scaled steer has no identity."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, OptimSpec, Plans, PolicySpec, Schedule,
        Seeds, encode,
    )

    entry = bank_entry(adapter, layer)
    plan = store.cas_put(encode(sft_train_plan(teacher_run)))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank={ENTRY: entry}),
        gen=None,
        plans=Plans(train=plan, rollout=None),
        algo=AlgoSpec(loss="sft", post=(),
                      optim=OptimSpec("adamw", lr=LR),
                      schedule=Schedule(microbatch_tokens=MICROBATCH_TOKENS,
                                        max_policy_lag=0)),
        topology=topology(),
        seeds=Seeds(master=11))


def bank_entry(adapter: str, layer: int):
    """ONE LAYER, THREE WAYS: the bank's one entry for an arm. `steer` and
    `nsteer` sit at the boundary the paper adds at (`resid_pre.<layer>`, the
    output of model.layers[layer]); `mlp_lora` is a rank-LORA_RANK delta on
    that layer's gate, up and down projections — the same layer's own
    computation, changed a little, instead of its output, pushed."""
    from rlstack import lora, nsteer, steer

    if adapter == "steer":
        return steer(f"resid_pre.{layer}", d=HIDDEN)
    if adapter == "nsteer":
        return nsteer(f"resid_pre.{layer}", d=HIDDEN, alpha=ALPHA)
    if adapter == "mlp_lora":
        return lora(f"layers.{layer}.mlp.*", r=LORA_RANK)
    raise ValueError(f"adapter must be one of {ADAPTERS}; got {adapter!r}")


def onpolicy_rollout_plan(task_ids):
    """The student's OWN waves: WAVES x PER_WAVE prompts, each sampled once
    by the student under `single_turn` — no hint, no advantage; the teacher's
    opinion of those tokens is the post pipeline's, not the plan's."""
    from rlstack import GroupPlan, RunPlan, Sample, WavePlan

    def wave(u: int):
        return WavePlan(tuple(
            GroupPlan(task_id, (Sample(task_id, "single_turn"),))
            for task_id in task_ids[u * PER_WAVE:(u + 1) * PER_WAVE]))
    return RunPlan(tuple(wave(u) for u in range(WAVES)))


def onpolicy_train_plan():
    """Update u trains on this run's OWN rollout u, whole (ADR 0006's
    `self://rollouts/<u>`) — the ordinary paced loop."""
    from rlstack import RunPlan, WaveRef

    return RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                         for u in range(1, WAVES + 1)))


def opd_topology():
    """TWO HOSTS: the student's unit (`main` at tp=2 alternating with the
    learner at fsdp=2, as every arm) and the TEACHER as its own host — the
    bare base under the hint, scored over the wire. Two pools of one base
    are two engine residents at a carve (one per regime), and two 32B
    engines do not fit beside a learner on one 80 GB pair; a second HostSpec
    is a second placement unit, so the desk puts the teacher on the other
    metal. This is the measurement's arrangement, declared."""
    from rlstack import HostSpec, LearnerMember, PoolMember, Topology

    return Topology(hosts=(
        HostSpec((PoolMember("main", tp=WIDTH, vram_gb=MAIN_GB),
                  LearnerMember(fsdp=WIDTH, vram_gb=LEARNER_GB))),
        HostSpec((PoolMember("teacher", tp=WIDTH, vram_gb=MAIN_GB),)),
    ))


def opd_spec(store, train_tasks: str, layer: int, adapter: str = "nsteer"):
    """ON-POLICY DISTILLATION, one arm: the student samples the train prompts
    under its own adapter (no hint), the conditioned teacher scores each
    sampled token under the hint through the `teacher` pool, and `opd` — the
    sampled-token reverse KL — pulls the student toward the teacher on the
    student's own distribution rather than the teacher's. No sealed set is
    replayed; the extent is this run's own 64 waves."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, GenSpec, OptimSpec, Plans, PolicySpec,
        SamplingSpec, Schedule, Seeds, encode, load_tasks,
    )

    ids = [task.id for task in load_tasks(store, train_tasks)]
    if len(ids) < WAVES * PER_WAVE:
        raise ValueError(
            f"{train_tasks} holds {len(ids)} prompts; the plan wants "
            f"{WAVES * PER_WAVE}")
    rollout = store.cas_put(encode(onpolicy_rollout_plan(ids)))
    train = store.cas_put(encode(onpolicy_train_plan()))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank={ENTRY: bank_entry(adapter, layer)}),
        gen=GenSpec(envs=("single_turn",), tasks=(train_tasks,),
                    sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                          max_tokens=MAX_TOKENS)),
        plans=Plans(train=train, rollout=rollout),
        algo=AlgoSpec(loss="opd", post=("conditioned_teacher_logprobs",),
                      optim=OptimSpec("adamw", lr=LR),
                      schedule=Schedule(microbatch_tokens=MICROBATCH_TOKENS,
                                        max_policy_lag=0)),
        topology=opd_topology(),
        seeds=Seeds(master=13))


def the_measurement(task_ids):
    """The distillation number, OUTSIDE the run (#70): the student answers
    held-out prompts under its own vector, the conditioned teacher scores
    those very tokens, and `reverse_kl` is how many nats apart they are."""
    from rlstack import Measurement

    return Measurement(
        name="distill", env="single_turn", task_ids=tuple(task_ids),
        samples=1, every=8,
        post=("conditioned_teacher_logprobs", "reverse_kl"),
        seed=3, temperature=1.0, max_tokens=MAX_TOKENS)


# ---------------------------------------------------------------------------
# the volume-side functions: the corpus, the specs, the ledgers, the export
# ---------------------------------------------------------------------------

@app.function(image=tasks_image, volumes={"/store": store_volume,
                                          "/hf": hf_cache}, timeout=3600)
def build_prompts(concept: str = CONCEPT, seed: int = SPLIT_SEED) -> dict:
    """The corpus into the cas — the same `build_task_sets` the CLI calls, so
    a local build and this one print the same table. The chat-template
    assertion runs here, per row, and refuses the set rather than shipping a
    teacher conditioned on text no chat model reads."""
    from rlstack.__main__ import build_task_sets
    from rlstack.data.tasks.concept_prompts import prompt_splits

    uris = build_task_sets(
        a_store(), "concept_prompts",
        prompt_splits(train=TRAIN_ASK, heldout=HELDOUT_ASK), seed)
    store_volume.commit()
    return uris


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def canonical(kind: str, train_tasks: str = "", teacher_run: str = "",
              layer: int = 0, adapter: str = "steer") -> dict:
    """One spec as its canonical row, for the client to submit. The plan goes
    in the cas here, which is why this runs on the volume."""
    from rlstack import canonical_json

    store = a_store()
    spec = (teacher_spec(store, train_tasks) if kind == "teacher"
            else opd_spec(store, train_tasks, layer, adapter) if kind == "opd"
            else student_spec(store, teacher_run, layer, adapter))
    store_volume.commit()
    return json.loads(canonical_json(spec))


progress = progress_function(app, cpu_image, module=__name__)
"""Each run's extent progress, off the store — the chassis' one reader."""


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=1200)
def measure_once(run_id: str, heldout_tasks: str, address: str) -> dict:
    """One idempotent measuring pass against the serving host.

    "teacher" and "main" route to the SAME engine object under different
    bundles: the student's restored version for `main`, a payload-free base
    bundle for `teacher` — which is exactly what the conditioned teacher is,
    the bare Qwen3-32B told about happiness by the hint. One pool, two names
    (the Routes contract already says one engine may back many).
    """
    import asyncio

    from rlstack import load_tasks, measure_run
    from rlstack.runner.remote import RemotePool, transport_for

    store_volume.reload()
    store = a_store()
    tasks = {t.id: t for t in load_tasks(store, heldout_tasks)}
    pool = RemotePool(transport_for(address), base=BASE, tp=WIDTH)
    fresh = asyncio.run(measure_run(
        store, run_id, the_measurement(sorted(tasks)), pool, tasks,
        pools={"teacher": pool}))
    store_volume.commit()
    told = store.read_measurements(run_id).get("distill", {})
    return {"measured": fresh,
            "points": [{"update": p["update"], **p["means"]}
                       for p in told.get("points", [])]}


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def export_vector(run_id: str, version: int) -> dict:
    """The trained vector out of the run's blobs and onto the volume, for the
    paper's harness. `adapters/<name>@<v>.bin` IS `steer_torch.emit`'s
    safetensors, keyed by boundary PATH (`model.layers.10`), so the harness
    adds the tensor under that key at the output of that layer (Q8)."""
    store_volume.reload()
    return export_blob(a_store(), run_id, f"adapters/{ENTRY}@{version}.bin",
                       f"exports/{run_id}/{ENTRY}@{version}.safetensors")


# ---------------------------------------------------------------------------
# the doors — submit and follow, never release (ADR 0007, Q6)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def prompts(concept: str = CONCEPT, seed: int = SPLIT_SEED) -> None:
    """THE CORPUS. No metal: the builder needs a tokenizer and a download,
    not a GPU. Prints the uris the specs below pin."""
    print(json.dumps(build_prompts.remote(concept, seed), indent=1))


@app.local_entrypoint()
def up() -> None:
    """Boot the metal (spawning the keepalive is the knock) and wait until it
    has registered itself with the desk."""
    call = metal_handle(APP).serve.spawn()
    print(f"[up] {METAL} serving: call {call.object_id}")
    print(json.dumps(wait_for_metal(METAL), indent=2))


@app.local_entrypoint()
def distill_set(train_tasks: str = "", timeout_s: float = 14400.0) -> None:
    """THE TEACHER'S TRAJECTORY SET, as a generation-only run. Prints the
    run_id the three arms name in their plans."""
    if not train_tasks:
        raise SystemExit("--train-tasks <cas uri from ::prompts>")
    metal_handle(APP).serve.spawn()
    print(json.dumps(wait_for_metal(METAL), indent=1), flush=True)
    run_id = submit_and_follow(progress, canonical.remote("teacher",
                                                          train_tasks),
                               SUBDIR, timeout_s)
    print(f"[set] the teacher's rollouts are run {run_id} — "
          f"pass it to ::train --teacher-run", flush=True)


@app.local_entrypoint()
def train(layer: int = 0, teacher_run: str = "",
          timeout_s: float = 14400.0, adapter: str = "steer") -> None:
    """ONE ARM: the steer at `resid_pre.<layer>`, SFT over the teacher's set.
    Three arms run back to back on ONE booted metal — which is exactly what
    the campaign door not releasing buys (Q6)."""
    if layer not in ANCHORS:
        raise SystemExit(f"--layer must be one of {list(ANCHORS)}")
    if not teacher_run:
        raise SystemExit("--teacher-run <run_id from ::distill_set>")
    metal_handle(APP).serve.spawn()
    print(json.dumps(wait_for_metal(METAL), indent=1), flush=True)
    run_id = submit_and_follow(
        progress, canonical.remote("student", "", teacher_run, layer, adapter),
        SUBDIR, timeout_s)
    print(f"[arm] resid_pre.{layer} trained as run {run_id}", flush=True)


@app.local_entrypoint()
def follow_run(run_id: str = "", timeout_s: float = 14400.0) -> None:
    """RE-ATTACH to a run already on the metal and follow it to its extent.
    A driver that died (this venue's first teacher run lost its follower to
    a chassis bug) leaves the run untouched: the tenancy is the desk's and
    the daemons are the host's, and only the watching stopped."""
    if not run_id:
        raise SystemExit("--run-id <rid>")
    print(f"[follow] {follow(progress, run_id, timeout_s)} reached its extent",
          flush=True)


@app.local_entrypoint()
def measure(run_id: str = "", heldout_tasks: str = "",
            adapter: str = "steer") -> None:
    """THE DISTILLATION NUMBER, outside the run: one idempotent pass that
    backfills every 8th committed version it has not measured."""
    import asyncio

    from rlstack.runner.desk import Demand

    if not run_id or not heldout_tasks:
        raise SystemExit("--run-id <rid> --heldout-tasks <cas uri>")
    # a PURE CLIENT's door: demands in, addresses out — the measurer joins
    # the serving pool and is thereafter just admitted traffic, which the
    # idle rule counts, so the metal is not released under its own sampling.
    # RESOLVE FIRST: the pool that serves the arms' steer is whichever host
    # the desk lists for it — the arms' own, on any metal — and booting THIS
    # venue's metal to wait on it would stand a second pair up for nothing
    # (found on the venue). Only when nothing covers the demand is the knock
    # this venue's to make.
    def resolve():
        return asyncio.run(desk().resolve((
            Demand(pool="main", capability="inference", base=BASE, shape=WIDTH,
                   vram_gb=MAIN_GB, group=0, adapter_types=(adapter,)),)))
    placed = resolve()
    if not placed.get("placed"):
        metal_handle(APP).serve.spawn()
        print(json.dumps(wait_for_metal(METAL), indent=1), flush=True)
        placed = resolve()
    print(f"[measure] placed: {json.dumps(placed, default=str)[:400]}",
          flush=True)
    if not placed.get("placed"):
        raise SystemExit(f"no serving host: {placed}")
    print(json.dumps(measure_once.remote(run_id, heldout_tasks,
                                         placed["pools"]["main"]), indent=1))


@app.local_entrypoint()
def export(run_id: str = "", version: int = 0) -> None:
    """The vector out to the volume, for the paper's ICL harness (Q10). No
    metal: blobs are store reads."""
    if not run_id:
        raise SystemExit("--run-id <rid> --version <n>")
    print(json.dumps(export_vector.remote(run_id, version), indent=1))


@app.function(image=gpu_image, timeout=1800)
def run_tests() -> str:
    """The fakes suite inside the image, where the torch-gated cases run."""
    return run_suite()
