"""THE GSM-SYMBOLIC CAMPAIGN on one card, through the desk: five arms, one of
them empirical Bayes.

    modal deploy deploy/desk.py                      # THE desk, once, for every venue
    modal deploy deploy/gsm_a100.py                  # this venue: the metal, the screen, the cron
    modal run deploy/gsm_a100.py::up                 # boot the metal; it registers itself
    modal run deploy/gsm_a100.py::screen_now         # the base over every family -> the ten learnable
    modal run deploy/gsm_a100.py::campaign           # the five arms through the desk
    modal run deploy/gsm_a100.py::campaign --arms gsm-grpo,gsm-elbo   # a subset of them
    modal run deploy/gsm_a100.py::status             # this metal, its listings, each arm's extent
    modal run deploy/gsm_a100.py::measure_now        # one measurement pass, by hand
    modal run deploy/desk.py::status                 # the whole fleet, at the desk
    modal run deploy/desk.py::release --metal gsm-a  # hand the card back sooner than idle (guarded)

THE VENUE: one metal (gsm-a) on whichever card frees first (GPUS). Two
HostSpecs, two carves on its single device — serve SERVE_GB and learner
LEARN_GB, in GB because the card is not known until the container measures
it (ADR 0001) — so one engine and one learner share the card and the five
tenants share both. The metal is the chassis' (`metal_class`, ADR 0007): it
boots BARE, measures its card, registers with THE ONE desk carrying this
file's recipe as a PROPOSAL (Q4), holds its shift open while its hosts carry
work, and is released by the desk for sitting idle (ADR 0003). No desk and no
fleet journal live here any more: `deploy/desk.py` is every venue's desk,
which is what lets this campaign share a plane with whatever else is listed
(Q2). THE DOORS NEVER RELEASE — a campaign venue's idle metal is the desk's
to collect (Q6); `deploy/desk.py::release --metal gsm-a` is the hand that
gives it back sooner, and it is refused while an arm is still running.

THE SCREEN, then THE EXPERIMENT. `screen` samples the BASE model over every
GSM-Symbolic template (a template = a task family: one procedure, 50
parametric instances) and writes the ten families where Qwen3-0.6B lands
LOW BUT NONZERO — enough variance for a GRPO group to carry signal, enough
headroom to see learning. The campaign then trains on TWO instances of ONE
chosen family and evaluates the sweep: held-out instances of that family
(near transfer — same procedure, different strings) plus instances of the
other nine (far transfer). Five arms:

    grpo      lora r=16, plain GRPO                      — the baseline
    spectral  spectral k=16, plain GRPO                  — SVF, no latent
    slatent   spectral_latent k=16, gated latent KL      — SVF with the latent
    elbo      spectral_latent k=16, LEARNED prior, ELBO  — empirical Bayes:
                                                          the prior's scale
                                                          trains, no beta
    sdpo      lora r=16, the reflect loop, sdpo          — the loop teaches

The `measure` cron backfills TWO measurements per arm every EVAL_EVERY
updates, always under the plain math_single_turn environment, sdpo included:
`near` — the NEAR_EVAL held-out instances of the train family, the same
procedure under different strings, which is the within-class generalization
the campaign is about — and `heldout`, the whole eval set, near plus far.

Everything semantics-bearing is below — the task sets, the plans, the five
specs, the measurement (I5); everything else is the chassis' (ADR 0007). Every
spec value is byte-identical to the rows the pre-chassis file built
(tests/test_venues.py, against tests/venue_spec_rows.json), so an arm placed
by the old venue's desk resubmitted here is a resume by identity (I3).
"""

from __future__ import annotations

import asyncio
import json

import modal

from modal_venue import (
    a_store, cpu_image_for, desk, gpu_image_for, hf_cache, metal_class,
    metal_handle, progress_function, store_volume, submit_spec,
    wait_for_metal,
)

APP = "rlstack-gsm-a100"
app = modal.App(APP)

cpu_image = cpu_image_for()
gpu_image = gpu_image_for()

# the campaign builder's own layer, kept AFTER the pinned one so the pins
# stay cached with the campaign images (deploy/tasks_dapo.py's rule, #60):
# the tokenizer's chat template is what turns a question into a prompt
tasks_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==5.16.1", "huggingface_hub", "safetensors",
                 "numpy")
    .pip_install("jinja2")   # apply_chat_template needs it (found on the venue)
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("rlstack", "modal_venue")
)

BASE = "Qwen/Qwen3-0.6B"
SITE = "layers.*.self_attn.*"
ENV = "math_single_turn"       # the RL arms' environment, and every measurement's
SUBDIR = "gsm"

UPDATES = 200                  # the four RL arms' plan length
LOOPS = 120                    # the sdpo arm: one loop = attempt + 2 reflects
GROUP_SIZE = 8
MAX_TOKENS = 512               # worked arithmetic needs room the DSLs did not
SEED = 11

PLORA = {"latent": 64, "prior_std": 0.05, "members": 4}
# the elbo arm's learned prior: ONE scalar chased by the KL alone. At the
# entry's 1e-3 Adam moves its log-scale at most 0.2 in UPDATES steps (e^0.2
# in scale) — too little to reach any fixed point the posterior finds — so
# the `prior` group gets its own rate (CONTEXT #83)
PRIOR_LR = 1e-2
SPECTRAL_K = 16

TRAIN_INSTANCES = (0, 1)       # two examples, the campaign's whole trainset
NEAR_EVAL = 10                 # held-out instances of the train family
FAR_EVAL = 4                   # instances of each other chosen family

SCREEN_INSTANCES = 6           # instances probed per family by the screen
SCREEN_SAMPLES = 4             # completions per probed instance
SCREEN_TEMPERATURE = 0.8
SCREEN_BAND = (0.0, 0.40)      # low but NONZERO: 0 < accuracy <= 0.40

EVAL_EVERY = 5
EVAL_SAMPLES = 2
EVAL_TEMPERATURE = 0.2
RUNS_KEY = "measurements/gsm/runs.json"      # {run_id: {...}} — the cron's list
SCREEN_KEY = "measurements/gsm/screen.json"  # the screen's verdict
ROWS_KEY = "measurements/gsm/rows-main.json"  # the dataset, fetched ONCE

# Memory in GB, TOTAL per member (one shard each here): what the spec
# declares and the desk books; the metal converts to its partition's
# fraction against the card it measured (fraction_for_gb, ADR 0001)
SERVE_GB = 18.0
LEARN_GB = 16.0

# ANY of these unblocks the venue — 0.6B at these budgets fits every card
# here, and the scheduler takes whichever frees first (a stale A100 queue
# sat 45+ minutes; the iteration loop pays for GPU loyalty). GB is what makes
# this honest: 18 GB is 18 GB on an L40S and on an H100
GPUS = ["A100-40GB", "L40S", "A100-80GB", "H100", "L4"]
# L4 LAST, for the measurement: after the arms finish, a 0.6B's 18 GB pool
# fits a 24 GB card, and a night when every big card is out (yu-masala,
# 2026-09-04: ninety minutes waiting) should not be a night without curves.
# A campaign that lands there is refused at the carve (34 GB > 24) and parks
# with a boot instruction, which is the honest answer on that card.

METAL = "gsm-a"
IDLE_S = 90.0                # the desk's clock; the venue's scaledown is no shorter


def proposed_recipe():
    """WHAT THIS METAL IS FOR, PROPOSED (ADR 0007, Q4): the desk journals it as
    its own `recipe` event and `deploy/desk.py::recipe` is the last word. The
    engine serves every adapter type the five arms name — the join refuses a
    spec whose bank names one it does not (Q4a) — with room for the latent
    arms' `members` and the five tenants' bundles; the learner is the default
    build."""
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    return Builds(
        engine=EngineBuild(max_model_len=4096, max_bundles=32, max_rank=16,
                           max_members=PLORA["members"],
                           serves=("lora", "plora", "spectral",
                                   "spectral_latent")),
        learner=LearnerBuild())


MetalS = metal_class(app, APP, METAL, GPUS, gpu_image, module=__name__,
                     idle_s=IDLE_S, recipe=proposed_recipe())


# ---------------------------------------------------------------------------
# the science: the task sets, the plans, the five arms, the measurement (I5)
# ---------------------------------------------------------------------------

def topology():
    """TWO HOSTSPECS on one card: the serving host and the learner host are
    two dedicated hosts — two carves, two honest bookings (ADR 0001: side by
    side is two hosts; one HostSpec with both members would ALTERNATE them
    and force lag 0). The frame lands on the learner's host and its Trainer
    reaches `main` through the metal's in-process switchboard, never a
    self-call (#77, the chassis' rule) — proven on fakes (test_chassis),
    UNPROVEN on this venue's metal until it boots."""
    from rlstack import HostSpec, LearnerMember, PoolMember, Topology

    return Topology(hosts=(
        HostSpec((PoolMember("main", tp=1, vram_gb=SERVE_GB),)),
        HostSpec((LearnerMember(fsdp=1, vram_gb=LEARN_GB),))))


def chat_formatter():
    """Qwen3's chat template with thinking OFF, so the visible completion IS
    the worked reasoning — the one thing in this file that needs the
    tokenizer, which is why `campaign_specs` takes it as a value."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE)

    def chat(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    return chat


def gsm_rows(store) -> list:
    """The dataset rows, fetched from the datasets-server ONCE EVER and
    cached on the store — re-fetching 5,000 rows per build drew an HTTP 429
    (observed live), and the rows are immutable content anyway."""
    from rlstack.data.tasks.gsm_symbolic import fetch_rows

    try:
        return json.loads(store._read(ROWS_KEY))
    except FileNotFoundError:
        rows = fetch_rows("main")
        store._write(ROWS_KEY, json.dumps(rows, sort_keys=True).encode())
        store_volume.commit()
        return rows


def gsm_task_sets(store, verdict: dict, chat):
    """(train uri, eval uri, train ids) for the screen's chosen families:
    TRAIN_INSTANCES of the train family to learn from; NEAR_EVAL of its other
    instances plus FAR_EVAL of each other chosen family held out."""
    from rlstack.data.tasks.base import write_tasks
    from rlstack.data.tasks.gsm_symbolic import gsm_eval_tasks, gsm_train_tasks

    rows = gsm_rows(store)
    train = gsm_train_tasks(rows, verdict["train_family"], TRAIN_INSTANCES,
                            chat)
    held = gsm_eval_tasks(rows, verdict["train_family"], TRAIN_INSTANCES,
                          verdict["chosen"], near=NEAR_EVAL, far=FAR_EVAL,
                          chat=chat)
    return (write_tasks(store, train), write_tasks(store, held),
            [t.id for t in train])


def rl_plans(store, task_ids: list[str]):
    """The four RL arms' plan: UPDATES waves of one GROUP_SIZE group per
    train task, update u training on rollout u."""
    from rlstack import GroupPlan, Plans, RunPlan, Sample, WavePlan, WaveRef, encode

    wave = WavePlan(tuple(
        GroupPlan(task, tuple(Sample(task, ENV) for _ in range(GROUP_SIZE)))
        for task in task_ids))
    return Plans(
        train=store.cas_put(encode(RunPlan(tuple(
            WaveRef(f"self://rollouts/{u}") for u in range(1, UPDATES + 1))))),
        rollout=store.cas_put(encode(RunPlan((wave,) * UPDATES))))


def loop_plans(store, task_ids: list[str]):
    """The sdpo arm's plan: LOOPS loops of an attempt and two reflects — each
    reflect a wave of Derive leaves naming the previous wave's rollouts
    through the `reflect` maker into the `reflect_retry` environment — with
    update u training on loop u's LAST wave, whole."""
    from rlstack import (
        Derive, GroupPlan, Plans, RunPlan, Sample, WavePlan, WaveRef, encode,
    )

    attempt = WavePlan(tuple(
        GroupPlan(task, tuple(Sample(task, ENV) for _ in range(GROUP_SIZE)))
        for task in task_ids))

    def reflected(source_wave: int) -> WavePlan:
        return WavePlan(tuple(
            GroupPlan(task, tuple(
                Derive(f"self://rollouts/{source_wave}#{g * GROUP_SIZE + i}",
                       "reflect", "reflect_retry")
                for i in range(GROUP_SIZE)))
            for g, task in enumerate(task_ids)))

    waves = []
    for loop in range(LOOPS):
        base = 3 * loop
        waves += [attempt, reflected(base + 1), reflected(base + 2)]
    return Plans(
        train=store.cas_put(encode(RunPlan(tuple(
            WaveRef(f"self://rollouts/{3 * (u + 1)}")
            for u in range(LOOPS))))),
        rollout=store.cas_put(encode(RunPlan(tuple(waves)))))


ARM_TAGS = {
    "gsm-grpo": (["gsm", "grpo", "lora-baseline"],
                 "plain LoRA GRPO on the GSM-Symbolic train family"),
    "gsm-spectral": (["gsm", "spectral", "svf"],
                     "SVF: trainable singular-value gains, top-k served"),
    "gsm-slatent": (["gsm", "spectral", "latent", "svf", "gated"],
                    "SVF gains generated from a latent, gated KL"),
    "gsm-elbo": (["gsm", "spectral", "latent", "svf", "elbo", "learned-prior"],
                 "SVF gains generated from a latent, prior LEARNED, the "
                 "ELBO (empirical Bayes, no beta)"),
    "gsm-sdpo": (["gsm", "sdpo", "reflect-loop"],
                 "iterative reflect loop, final-turn cloning"),
}
"""Each arm's name and taxonomy for annotations.jsonl — what the observer
shows beside the run id."""


def campaign_specs(store, chat) -> tuple[dict, str]:
    """THE FIVE ARMS AS VALUES, plus the eval-set uri: ({name ->
    ExperimentSpec}, eval uri). `chat` is the prompt formatter —
    `chat_formatter()` on the venue, a stand-in in the suite — and the
    screen's verdict at SCREEN_KEY says which families."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, GenSpec, OptimSpec, PolicySpec,
        SamplingSpec, Schedule, Seeds, lora,
    )
    from rlstack.policy.adapters.spectral import spectral
    from rlstack.policy.adapters.spectral_latent import spectral_latent

    verdict = json.loads(store._read(SCREEN_KEY))
    train_uri, eval_uri, train_ids = gsm_task_sets(store, verdict, chat)
    plans = rl_plans(store, train_ids)
    loop = loop_plans(store, train_ids)

    def arm(bank, loss, post, lr, overrides=None, plans=plans, envs=(ENV,),
            makers=(), lag=1):
        return ExperimentSpec(
            policy=PolicySpec(base=BASE, bank=bank),
            gen=GenSpec(envs=envs, tasks=(train_uri, eval_uri),
                        sampling=SamplingSpec(temperature=1.0,
                                              max_tokens=MAX_TOKENS),
                        makers=makers),
            plans=plans,
            algo=AlgoSpec(loss=loss, post=post,
                          optim=OptimSpec("adamw", lr=lr, weight_decay=0.0,
                                          overrides=dict(overrides or {})),
                          schedule=Schedule(microbatch_tokens=4096,
                                            max_policy_lag=lag)),
            topology=topology(),
            seeds=Seeds(master=SEED))

    latent = dict(k=SPECTRAL_K, latent=PLORA["latent"],
                  members=PLORA["members"], prior_std=PLORA["prior_std"])
    specs = {
        "gsm-grpo": arm(
            {"pi": lora(SITE, r=16)}, "grpo",
            ("final_answer", "grpo_advantage"), 1e-4),
        "gsm-spectral": arm(
            {"pi": spectral(SITE, k=SPECTRAL_K)}, "grpo",
            ("final_answer", "grpo_advantage"), 1e-3),
        "gsm-slatent": arm(
            {"pi": spectral_latent(SITE, **latent)},
            "grpo_latent_kl_gated",
            ("final_answer", "grpo_advantage", "group_accuracy"),
            1e-3, {"pi.mapper": {"weight_decay": 1e-2}}),
        # slatent's twin with the prior's scale TRAINED and the KL priced by
        # the bound itself: same adapter, same latent, same members; the
        # posterior and the prior are fit together, once per trajectory, no
        # beta and no gate (CONTEXT #83). Watch spectral_prior_std against
        # spectral_sigma_mean in the ledger
        "gsm-elbo": arm(
            {"pi": spectral_latent(SITE, prior="learned", **latent)},
            "grpo_elbo",
            ("final_answer", "grpo_advantage"),
            1e-3, {"pi.mapper": {"weight_decay": 1e-2},
                   "pi.prior": {"lr": PRIOR_LR}}),
        "gsm-sdpo": arm(
            {"pi": lora(SITE, r=16)}, "sdpo", ("final_answer",), 1e-4,
            plans=loop, envs=(ENV, "reflect_retry"), makers=("reflect",),
            lag=0),
    }
    return specs, eval_uri


def near_ids(eval_ids: list[str], train_family: int) -> list[str]:
    """The eval tasks of the TRAIN family — a task id carries its family
    (`gsm-symbolic/t<family>-i<instance>`, gsm_symbolic's rule), so the
    within-class subset is read off the ids and never off the rows."""
    prefix = f"gsm-symbolic/t{train_family:02d}-"
    return [task_id for task_id in eval_ids if task_id.startswith(prefix)]


def gsm_measurements(eval_ids: list[str], train_family: int) -> tuple:
    """TWO held-out numbers, OUTSIDE the run (#70), on one cadence: `near`,
    the held-out instances of the train family — the same procedure under
    different strings, the within-class generalization the campaign is
    about — and `heldout`, the whole eval set, near plus the far families.
    Every EVAL_EVERY updates, under the PLAIN environment — sdpo's
    iteration-0 behaviour included, which is the honest metric for a loop
    that could otherwise learn to sandbag its first attempt."""
    from rlstack import Measurement

    def measurement(name: str, task_ids: list[str]):
        return Measurement(
            name=name, env=ENV, task_ids=tuple(task_ids),
            samples=EVAL_SAMPLES, every=EVAL_EVERY, post=("final_answer",),
            seed=7, temperature=EVAL_TEMPERATURE, max_tokens=MAX_TOKENS)

    return (measurement("near", near_ids(eval_ids, train_family)),
            measurement("heldout", eval_ids))


# ---------------------------------------------------------------------------
# the volume-side functions: the screen, the specs, the roster, the ledgers
# ---------------------------------------------------------------------------

@app.function(image=gpu_image, gpu=GPUS,
              volumes={"/store": store_volume, "/hf": hf_cache},
              timeout=7200)
def screen() -> dict:
    """Sample the BASE 0.6B over SCREEN_INSTANCES x SCREEN_SAMPLES of all
    100 families, grade with the campaign's own marker reader, and write the
    ten families inside SCREEN_BAND (low but NONZERO) plus the train pick
    (the median of the chosen — the middle of the learnable band, where a
    group of 8 most reliably splits). Direct vLLM: a screening pass is
    measurement of the base model, not a run — no desk, no store artifacts
    beyond the verdict."""
    from collections import defaultdict

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from rlstack.data.tasks.gsm_symbolic import ASK, gold_of
    from rlstack.training.post.final_answer import stated_answer

    store = a_store()
    rows = gsm_rows(store)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    probes = []                      # (family, prompt, gold)
    for row in rows:
        if int(row["instance"]) >= SCREEN_INSTANCES:
            continue
        gold = gold_of(row)
        if gold is None:
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": f"{row['question'].strip()}\n\n{ASK}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        probes.append((int(row["id"]), prompt, gold))
    print(f"[screen] {len(probes)} probes over "
          f"{len({f for f, _, _ in probes})} families", flush=True)

    llm = LLM(model=BASE, max_model_len=2048,
              gpu_memory_utilization=0.9, dtype="bfloat16")
    params = SamplingParams(temperature=SCREEN_TEMPERATURE, top_p=0.95,
                            max_tokens=MAX_TOKENS, n=SCREEN_SAMPLES, seed=SEED)
    outs = llm.generate([p for _, p, _ in probes], params)

    hits = defaultdict(list)
    for (family, _, gold), out in zip(probes, outs):
        for completion in out.outputs:
            claimed = stated_answer(completion.text)
            hits[family].append(float(claimed == str(gold)))
    accuracy = {f: sum(v) / len(v) for f, v in sorted(hits.items())}
    low, high = SCREEN_BAND
    landable = sorted((f for f, a in accuracy.items() if low < a <= high),
                      key=lambda f: accuracy[f])
    chosen = landable[:10]
    if len(chosen) < 10:
        raise RuntimeError(
            f"only {len(chosen)} families inside {SCREEN_BAND}: {accuracy}")
    train_family = sorted(chosen, key=lambda f: accuracy[f])[len(chosen) // 2]
    verdict = {"accuracy": accuracy, "chosen": chosen,
               "train_family": train_family,
               "band": list(SCREEN_BAND), "samples": SCREEN_SAMPLES,
               "instances": SCREEN_INSTANCES}
    store._write(SCREEN_KEY, json.dumps(verdict, sort_keys=True).encode())
    store_volume.commit()
    print(f"[screen] chosen {chosen}, train family {train_family}, "
          f"accuracies {[round(accuracy[f], 3) for f in chosen]}", flush=True)
    return verdict


@app.function(image=tasks_image, cpu=8.0, memory=16384,
              volumes={"/store": store_volume, "/hf": hf_cache},
              timeout=3600)
def build_campaign() -> dict:
    """The five specs as canonical rows, for the client to submit. The task
    sets and the plans go into the cas here, which is why this runs on the
    volume."""
    from rlstack import canonical_json

    store = a_store()
    specs, eval_uri = campaign_specs(store, chat_formatter())
    store_volume.commit()
    return {"specs": {name: json.loads(canonical_json(spec))
                      for name, spec in specs.items()},
            "eval": eval_uri}


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def write_roster(roster: dict) -> None:
    """The campaign's roster at RUNS_KEY — the cron's list — and each run
    tagged (name + taxonomy) in annotations.jsonl for the observer."""
    store = a_store()
    store._write(RUNS_KEY, json.dumps(roster, sort_keys=True).encode())
    for rid, entry in roster.items():
        tags, note = ARM_TAGS[entry["arm"]]
        store.annotate_run(rid, name=entry["arm"], tags=tags, note=note)
    store_volume.commit()


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def read_roster() -> dict:
    """{run_id: {arm, eval}} as `campaign` wrote it, or {} before it ran."""
    store_volume.reload()
    try:
        return json.loads(a_store()._read(RUNS_KEY))
    except FileNotFoundError:
        return {}


progress = progress_function(app, cpu_image, module=__name__)
"""Each run's extent progress, off the store — the chassis' one reader."""


# ---------------------------------------------------------------------------
# the measurement cron
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, volumes={"/store": store_volume},
              schedule=modal.Period(minutes=10), timeout=7200,
              max_containers=1)
async def measure() -> None:
    """The measurement pass, on a clock. A MEASUREMENT IS A PURE CLIENT: it
    asks the desk to RESOLVE one `main` demand — a join onto the campaign's
    standing pool while the arms run, a carve of a pool-only host on this
    venue's metal once they are done and the container has turned over (a
    fresh metal lists nothing, and a measurement with nothing to join would
    otherwise skip forever). The old venue never provisioned, because its
    desk once carved a second engine beside a shared card's first and OOMed
    every tenancy; today a carve is BOOKED in GB against the measured card
    and a live pool is JOINED, never doubled. Joined, the measuring is
    admitted traffic, which the idle rule counts.

    ONE PASS AT A TIME (`max_containers=1`, no concurrency): a firing that
    lands while the last is still backfilling QUEUES behind it instead of
    measuring the same points beside it — `measure_run` is idempotent per
    point only against what was on the store when the pass began, so two
    passes side by side would write every missing point twice. The timeout
    is a full backfill's, not a tick's: four arms x two measurements x forty
    points through one shared pool was measured at longer than 3000 s."""
    from rlstack import load_tasks, measure_run
    from rlstack.runner.desk import Demand
    from rlstack.runner.remote import RemotePool, transport_for

    await store_volume.reload.aio()
    store = a_store()
    try:
        listed = json.loads(store._read(RUNS_KEY))
    except FileNotFoundError:
        print("[measure] no campaign roster yet", flush=True)
        return
    train_family = int(json.loads(store._read(SCREEN_KEY))["train_family"])
    placed = await desk().resolve((Demand(
        pool="main", capability="inference", base=BASE, shape=1,
        vram_gb=SERVE_GB, group=0,
        adapter_types=("lora", "spectral", "spectral_latent")),))
    if not placed.get("placed"):
        print(f"[measure] no serving pool: {json.dumps(placed, default=str)[:300]}",
              flush=True)
        return
    pool = RemotePool(transport_for(placed["pools"]["main"]), base=BASE, tp=1)
    for rid, entry in sorted(listed.items()):
        tasks = {t.id: t for t in load_tasks(store, entry["eval"])}
        for measurement in gsm_measurements(sorted(tasks), train_family):
            fresh = await measure_run(store, rid, measurement, pool, tasks)
            print(f"[measure] {rid} ({entry['arm']}) {measurement.name}: "
                  f"{fresh}", flush=True)
    await store_volume.commit.aio()


# ---------------------------------------------------------------------------
# the doors — submit and leave running, never release (ADR 0007, Q6)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def up() -> None:
    """Boot the metal (spawning the keepalive is the knock) and wait until it
    has registered itself with the desk."""
    call = metal_handle(APP).serve.spawn()
    print(f"[up] {METAL} serving: call {call.object_id}")
    print(json.dumps(wait_for_metal(METAL), indent=2))


@app.local_entrypoint()
def screen_now() -> None:
    """THE SCREEN, by hand: prints the verdict the campaign will read."""
    print(json.dumps(screen.remote(), indent=1)[:2000])


@app.local_entrypoint()
def campaign(arms: str = "") -> None:
    """THE ARMS through the desk, submitted and LEFT RUNNING: the roster
    lands at RUNS_KEY, each accepted run is tagged, and the cron measures from
    here on. `--arms` names a subset, comma-separated; unsaid is all five.
    A refused arm is reported and the rest proceed — what the desk accepted
    is already on the metal, and the roster says which. This door never
    follows and never releases: the tenants share the card for hours, and
    idle metal is the desk's to collect."""
    metal_handle(APP).serve.spawn()
    print(json.dumps(wait_for_metal(METAL), indent=1), flush=True)
    told = build_campaign.remote()
    chosen = [arm for arm in arms.split(",") if arm] or sorted(told["specs"])
    unknown = sorted(set(chosen) - set(told["specs"]))
    if unknown:
        raise SystemExit(f"--arms names no such arm: {unknown}; the arms "
                         f"are {sorted(told['specs'])}")
    roster: dict = {}
    for name in sorted(chosen):
        try:
            reply = submit_spec(told["specs"][name], SUBDIR)
        except SystemExit as refused:
            print(f"[{name}] REFUSED: {refused}", flush=True)
            continue
        roster[reply["run_id"]] = {"arm": name, "eval": told["eval"]}
        print(f"[{name}] run {reply['run_id']} anchored at "
              f"{reply.get('host')}", flush=True)
    write_roster.remote(roster)
    print(f"[campaign] roster of {len(roster)} runs written to {RUNS_KEY}",
          flush=True)


@app.local_entrypoint()
def status() -> None:
    """This venue's slice of the fleet — the metal, the listings on it — and
    how far each arm has got."""
    told = desk().status()
    roster = read_roster.remote()
    extents = progress.remote(sorted(roster)) if roster else {}
    print(json.dumps({
        "metal": told["metal"].get(METAL),
        "listings": {name: row for name, row in told["listings"].items()
                     if row.get("metal") == METAL},
        "runs": {rid: {**roster[rid], **extents.get(rid, {})}
                 for rid in sorted(roster)}}, indent=2))


@app.local_entrypoint()
def measure_now() -> None:
    """One measurement pass, by hand — the cron's body, now."""
    measure.remote()
