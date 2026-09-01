"""ONE A100 the DESK owns, and the GSM-Symbolic campaign running on it.

    MODAL_PROFILE=yu-masala-workspace modal deploy deploy/gsm_a100.py
    ... run deploy/gsm_a100.py::up             # the metal registers its plane
    ... run deploy/gsm_a100.py::screen_now     # find the learnable families
    ... run deploy/gsm_a100.py::campaign       # the four arms through the desk
    ... run deploy/gsm_a100.py::status         # listings, residual, roster
    ... run deploy/gsm_a100.py::measure_now    # one measurement pass, by hand
    ... run deploy/gsm_a100.py::stop --call-id <id>     # kill the shift

THE VENUE: one A100-40GB metal (gsm-a). Both units carve on its single
device — serve 0.45 + learner 0.40 compose additively (#51: sub-GPU hosts,
vLLM budgets against device total) — so one engine and one learner share the
card and the four tenants share both. Its desk is this app's own, with its
OWN fleet journal (fleet/gsm.jsonl): the volume is shared with the DSL venue
and two desks must not replay each other's listings.

THE SCREEN, then THE EXPERIMENT. `screen` samples the BASE model over every
GSM-Symbolic template (a template = a task family: one procedure, 50
parametric instances) and writes the ten families where Qwen3-0.6B lands
LOW BUT NONZERO — enough variance for a GRPO group to carry signal, enough
headroom to see learning. The campaign then trains on TWO instances of ONE
chosen family and evaluates the sweep: held-out instances of that family
(near transfer — same procedure, different strings) plus instances of the
other nine (far transfer). Four arms:

    grpo      lora r=16, plain GRPO                      — the baseline
    spectral  spectral k=16, plain GRPO                  — SVF, no latent
    slatent   spectral_latent k=16, gated latent KL      — SVF with the latent
    sdpo      lora r=16, the reflect loop, sdpo          — the loop teaches

The `measure` cron backfills measurements/<rid>/heldout every 5 updates,
always under the plain math_single_turn environment, sdpo included.
"""

import json
import time

import modal

APP = "rlstack-gsm-a100"
app = modal.App(APP)

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlstack-hf-cache", create_if_missing=True)

gpu_image = (
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
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .add_local_python_source("rlstack", "rlstack_engine"))
build_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "transformers==5.16.1", "safetensors",
                 "numpy", "huggingface_hub")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-0.6B"
STORE = "modal://rlstack-store"
SITE = "layers.*.self_attn.*"

UPDATES = 200                  # the three RL arms' plan length
LOOPS = 120                    # the sdpo arm: one loop = attempt + 2 reflects
GROUP_SIZE = 8
MAX_TOKENS = 512               # worked arithmetic needs room the DSLs did not
SEED = 11

PLORA = {"latent": 64, "prior_std": 0.05, "members": 4}
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

SERVE_FRACTION = 0.45
LEARN_FRACTION = 0.40

# ANY of these unblocks the venue — 0.6B at fractional budgets fits every
# card here, and the scheduler takes whichever frees first (a stale A100
# queue sat 45+ minutes; the iteration loop pays for GPU loyalty)
GPUS = ["A100-40GB", "L40S", "A100-80GB", "H100"]

METALS = {"gsm-a": {"cls": "MetalG", "scheme": "gsma"}}


# ---------------------------------------------------------------------------
# the venue's transports (I5): desk-by-name, host-by-address, scheme-routed
# ---------------------------------------------------------------------------

def desk_handle():
    return modal.Cls.from_name(APP, "Desk")()


def metal_cls_for(address: str) -> str:
    scheme = address.split("://", 1)[0]
    for row in METALS.values():
        if row["scheme"] == scheme:
            return row["cls"]
    raise KeyError(f"no metal serves scheme {scheme!r} (address {address!r})")


def blocking_ask(fn):
    """One blocking Modal call on its OWN thread (the dsl venue's lesson:
    a blocking portal call from a loop thread wedges the loop)."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as one:
        return one.submit(fn).result()


class DeskTransport:
    def __init__(self) -> None:
        self._handle = None

    def handle(self):
        if self._handle is None:
            self._handle = desk_handle()
        return self._handle

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.handle().desk.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return blocking_ask(lambda: self.handle().desk_ask.remote(
            verb, payload))


class MetalTransport:
    """Host-addressed frames, routed to the owning metal container by
    scheme."""

    def __init__(self, address: str) -> None:
        self.address = address
        self._handle = None

    def handle(self):
        if self._handle is None:
            self._handle = modal.Cls.from_name(
                APP, metal_cls_for(self.address))()
        return self._handle

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.handle().host.remote.aio(self.address, verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return blocking_ask(lambda: self.handle().host_ask.remote(
            self.address, verb, payload))


class MetalPlaneTransport:
    """Metal-addressed frames (carve/decarve/residual), same scheme router."""

    def __init__(self, address: str) -> None:
        self.address = address
        self._handle = None

    def handle(self):
        if self._handle is None:
            self._handle = modal.Cls.from_name(
                APP, metal_cls_for(self.address))()
        return self._handle

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.handle().metal.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return blocking_ask(lambda: self.handle().metal_ask.remote(
            verb, payload))


# ---------------------------------------------------------------------------
# the desk: its own container, its OWN fleet journal on the shared volume
# ---------------------------------------------------------------------------

@app.cls(image=cpu_image, volumes={"/store": store_volume},
         timeout=3600, min_containers=1, max_containers=1,
         scaledown_window=1200)
@modal.concurrent(max_inputs=32)
class Desk:
    @modal.enter()
    def bring_up(self) -> None:
        from rlstack import ModalVolumeStore
        from rlstack.runner.campaign import Campaigns
        from rlstack.runner.desk import Desk
        from rlstack.runner.remote import RemoteHost, RemoteMetal

        class GsmDeskStore(ModalVolumeStore):
            """Two desks share this volume; this one journals to its OWN
            fleet file so neither replays the other's listings. cas blobs
            the desk writes (sliced plans) must be readable by the metal
            container before the adopt that names them."""

            def append_fleet_event(self, entry) -> None:
                self._append_line("fleet/gsm.jsonl", json.dumps(
                    entry, sort_keys=True, separators=(",", ":")))

            def read_fleet_log(self):
                try:
                    raw = self._read("fleet/gsm.jsonl").decode()
                except FileNotFoundError:
                    return []
                out = []
                for line in raw.strip().splitlines():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
                return out

        self.desk = Desk.from_journal(
            GsmDeskStore("/store", volume=store_volume, locator=STORE),
            connect=lambda address: RemoteHost(MetalTransport(address)),
            connect_metal=lambda address: RemoteMetal(
                MetalPlaneTransport(address)))
        self.door = Campaigns(self.desk)
        print(f"[desk] rebuilt from journal: {sorted(self.desk.listings)} "
              f"/ metal plane: {sorted(self.desk.metal_remotes)}")

    @modal.method()
    async def desk(self, verb: str, payload: dict) -> dict:
        return await self.door.serve(verb, payload)

    @modal.method()
    def desk_ask(self, verb: str, payload: dict) -> dict:
        return self.door.answer(verb, payload)


# ---------------------------------------------------------------------------
# the metal: one bare single-GPU container, every host a desk-issued carve
# ---------------------------------------------------------------------------

def bring_up_metal(name: str):
    """The metal container's whole state: its books, its factories, its
    router — dsl_a100's proven body, one metal instead of two."""
    from rlstack import ModalVolumeStore
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.desk import Metal as OwnedMetal, MetalService
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.learners.torch_learner import TorchLearner

    scheme = METALS[name]["scheme"]
    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    service = MetalService(
        OwnedMetal(name, "A100-40GB", 1, 40.0), store=store,
        engine_factory=lambda regime, partition: VllmEngine(
            regime.base, tp=regime.shape,
            gpu_memory_utilization=partition.memory,
            max_model_len=4096, max_bundles=32, max_rank=16,
            max_members=PLORA["members"], cas_get=store.cas_get,
            serves=("lora", "plora", "spectral", "spectral_latent")),
        learner_factory=lambda regime, partition: TorchLearner(
            device=f"cuda:{partition.devices[0]}"),
        address_of=lambda host_name: f"{scheme}://{host_name}",
        schema_for=hf_schema,
        dial=lambda address: MetalTransport(address),
        release=lambda host: [engine.shutdown() for engine in host.engines])
    print(f"[{name}] up, bare; residual {service.residual()}")
    return store, service


async def metal_shift(name: str, service) -> None:
    """The standing shift: register the plane, then hold the door open, a
    stats task following every carved host."""
    import asyncio

    from rlstack.runner.remote import RemoteDesk

    scheme = METALS[name]["scheme"]
    fleet = RemoteDesk(DeskTransport())
    try:
        await fleet.register_metal(name, "A100-40GB", 1, 40.0,
                                   f"{scheme}://metal")
        print(f"[{name}] registered on the metal plane")
    except Exception as taken:
        print(f"[{name}] not re-registered: {taken}")
    stats: dict[str, asyncio.Task] = {}
    tick = 0
    try:
        while True:
            for host_service in list(service.services.values()):
                host = host_service.host
                if host.name not in stats:
                    stats[host.name] = asyncio.create_task(host.run_stats())
            await asyncio.sleep(60)
            tick += 1
            if tick % 5 == 0:
                await store_volume.commit.aio()
    finally:
        for task in stats.values():
            task.cancel()


def roster_of(store, service) -> dict:
    out = {}
    for host_service in service.services.values():
        for rid, tenancy in sorted(host_service.host.roster.items()):
            entries = store.peek_ledger(rid)
            out[rid] = {"status": tenancy.status,
                        "host": host_service.host.name,
                        "committed": int(entries[-1]["update"])
                        if entries else 0}
    return out


@app.cls(image=gpu_image, gpu=GPUS,
         volumes={"/store": store_volume, "/hf": hf_cache},
         timeout=86400, scaledown_window=900, max_containers=1)
@modal.concurrent(max_inputs=64)
class MetalG:
    @modal.enter()
    def bring_up(self) -> None:
        self.store, self.metal_service = bring_up_metal("gsm-a")

    @modal.method()
    async def host(self, address: str, verb: str, payload: dict) -> dict:
        return await self.metal_service.service_for(address).serve(verb,
                                                                   payload)

    @modal.method()
    def host_ask(self, address: str, verb: str, payload: dict) -> dict:
        # NO reload here, deliberately: a reload invalidates the whole mount
        # under every in-flight writer on THIS container (grpo/sdpo died
        # mid-mkdir of the measure pass's first add_bundle, observed live) —
        # and cross-container cas blobs are already served by the store's
        # read fall-through to the volume's committed view.
        return self.metal_service.service_for(address).answer(verb, payload)

    @modal.method()
    async def metal(self, verb: str, payload: dict) -> dict:
        return await self.metal_service.serve(verb, payload)

    @modal.method()
    def metal_ask(self, verb: str, payload: dict) -> dict:
        return self.metal_service.answer(verb, payload)

    @modal.method()
    def roster(self) -> dict:
        return roster_of(self.store, self.metal_service)

    @modal.method()
    async def serve(self) -> None:
        await metal_shift("gsm-a", self.metal_service)

    @modal.exit()
    def bring_down(self) -> None:
        for service in self.metal_service.services.values():
            for engine in service.host.engines:
                engine.shutdown()


# ---------------------------------------------------------------------------
# the screen: the base model over every family, low-but-nonzero found
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

    from rlstack import ModalVolumeStore
    from rlstack.data.tasks.gsm_symbolic import ASK, gold_of
    from rlstack.training.post.final_answer import stated_answer

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
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
          f"{len({f for f, _, _ in probes})} families")

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
    from rlstack import ModalVolumeStore

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    store._write(SCREEN_KEY, json.dumps(verdict, sort_keys=True).encode())
    store_volume.commit()
    print(f"[screen] chosen {chosen}, train family {train_family}, "
          f"accuracies {[round(accuracy[f], 3) for f in chosen]}")
    return verdict


# ---------------------------------------------------------------------------
# the campaign: task sets, plans, and the four arms as values
# ---------------------------------------------------------------------------

def chat_formatter():
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
    except Exception:
        rows = fetch_rows("main")
        store._write(ROWS_KEY, json.dumps(rows, sort_keys=True).encode())
        store_volume.commit()
        return rows


def gsm_task_sets(store, verdict: dict):
    """(train uri, eval uri, train ids) for the screen's chosen families."""
    from rlstack.data.tasks.base import write_tasks
    from rlstack.data.tasks.gsm_symbolic import (
        gsm_eval_tasks, gsm_train_tasks,
    )

    chat = chat_formatter()
    rows = gsm_rows(store)
    train = gsm_train_tasks(rows, verdict["train_family"],
                            TRAIN_INSTANCES, chat)
    held = gsm_eval_tasks(rows, verdict["train_family"], TRAIN_INSTANCES,
                          verdict["chosen"], near=NEAR_EVAL, far=FAR_EVAL,
                          chat=chat)
    return (write_tasks(store, train), write_tasks(store, held),
            [t.id for t in train])


def rl_plans(store, env: str, task_ids: list[str]):
    from rlstack import GroupPlan, Plans, RunPlan, Sample, WavePlan, WaveRef, encode

    wave = WavePlan(tuple(
        GroupPlan(task, tuple(Sample(task, env) for _ in range(GROUP_SIZE)))
        for task in task_ids))
    return Plans(
        train=store.cas_put(encode(RunPlan(tuple(
            WaveRef(f"self://rollouts/{u}") for u in range(1, UPDATES + 1))))),
        rollout=store.cas_put(encode(RunPlan((wave,) * UPDATES))))


def loop_plans(store, env: str, task_ids: list[str]):
    from rlstack import (
        Derive, GroupPlan, Plans, RunPlan, Sample, WavePlan, WaveRef, encode,
    )

    attempt = WavePlan(tuple(
        GroupPlan(task, tuple(Sample(task, env) for _ in range(GROUP_SIZE)))
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
    "gsm-sdpo": (["gsm", "sdpo", "reflect-loop"],
                 "iterative reflect loop, final-turn cloning"),
}


def campaign_specs(store) -> tuple[dict, str]:
    """The four arms as values plus the eval-set uri:
    ({name -> ExperimentSpec}, eval uri)."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup, GpuSet,
        LearnerMember, OptimSpec, PolicySpec, PoolMember, SamplingSpec,
        Schedule, Seeds, lora,
    )
    from rlstack.policy.adapters.spectral import spectral
    from rlstack.policy.adapters.spectral_latent import spectral_latent

    verdict = json.loads(store._read(SCREEN_KEY))
    train_uri, eval_uri, train_ids = gsm_task_sets(store, verdict)
    env = "math_single_turn"
    plans = rl_plans(store, env, train_ids)
    loop = loop_plans(store, env, train_ids)

    def arm(bank, loss, post, lr, overrides={}, plans=plans,
            envs=(env,), makers=(), lag=1):
        return ExperimentSpec(
            policy=PolicySpec(base=BASE, bank=bank),
            gen=GenSpec(envs=envs, tasks=(train_uri, eval_uri),
                        sampling=SamplingSpec(temperature=1.0,
                                              max_tokens=MAX_TOKENS),
                        makers=makers),
            plans=plans,
            algo=AlgoSpec(loss=loss, post=post,
                          optim=OptimSpec("adamw", lr=lr,
                                          weight_decay=0.0,
                                          overrides=overrides),
                          schedule=Schedule(microbatch_tokens=4096,
                                            max_policy_lag=lag)),
            # ONE group, BOTH members: placement makes a single carve wearing
            # both regimes (the stress-matrix shape), so routes are empty and
            # the tenancy samples through its LOCAL pool — the two-carve
            # variant made the anchor self-dial its own container for every
            # sample, the unproven edge this venue wedged on
            gpu_config=GpuConfig(groups=(
                GpuGroup(gpus=GpuSet(n=1), members=(
                    PoolMember("main", tp=1, fraction=SERVE_FRACTION),
                    LearnerMember(fsdp=1, fraction=LEARN_FRACTION))),)),
            seeds=Seeds(master=SEED))

    specs = {
        "gsm-grpo": arm(
            {"pi": lora(SITE, r=16)}, "grpo",
            ("final_answer", "grpo_advantage"), 1e-4),
        "gsm-spectral": arm(
            {"pi": spectral(SITE, k=SPECTRAL_K)}, "grpo",
            ("final_answer", "grpo_advantage"), 1e-3),
        "gsm-slatent": arm(
            {"pi": spectral_latent(SITE, k=SPECTRAL_K,
                                   latent=PLORA["latent"],
                                   members=PLORA["members"],
                                   prior_std=PLORA["prior_std"])},
            "grpo_latent_kl_gated",
            ("final_answer", "grpo_advantage", "group_accuracy"),
            1e-3, {"pi.mapper": {"weight_decay": 1e-2}}),
        "gsm-sdpo": arm(
            {"pi": lora(SITE, r=16)}, "sdpo", ("final_answer",), 1e-4,
            plans=loop, envs=(env, "reflect_retry"), makers=("reflect",),
            lag=0),
    }
    return specs, eval_uri


@app.function(image=build_image, cpu=8.0, memory=16384,
              volumes={"/store": store_volume, "/hf": hf_cache},
              timeout=3600)
def build_campaign() -> dict:
    from rlstack import ModalVolumeStore
    from rlstack.spec.canonical import canonical_json

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    specs, eval_uri = campaign_specs(store)
    rows = {name: json.loads(canonical_json(spec))
            for name, spec in specs.items()}
    store_volume.commit()
    return {"specs": rows, "eval": eval_uri}


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def write_roster(roster: dict) -> None:
    from rlstack import ModalVolumeStore

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    store._write(RUNS_KEY, json.dumps(roster, sort_keys=True).encode())
    for rid, entry in roster.items():
        tags, note = ARM_TAGS[entry["arm"]]
        store.annotate_run(rid, name=entry["arm"], tags=tags, note=note)
    store_volume.commit()


# ---------------------------------------------------------------------------
# the measurement cron
# ---------------------------------------------------------------------------

def gsm_measurement(eval_ids: list[str]):
    from rlstack import Measurement

    return Measurement(
        name="heldout", env="math_single_turn", task_ids=tuple(eval_ids),
        samples=EVAL_SAMPLES, every=EVAL_EVERY, post=("final_answer",),
        seed=7, temperature=EVAL_TEMPERATURE, max_tokens=MAX_TOKENS)


@app.function(image=cpu_image, volumes={"/store": store_volume},
              schedule=modal.Period(minutes=10), timeout=3000)
async def measure() -> None:
    from rlstack import ModalVolumeStore, load_tasks, measure_run
    from rlstack.runner.remote import RemoteDesk, RemotePool
    from rlstack.runner.desk import Demand

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    try:
        listed = json.loads(store._read(RUNS_KEY))
    except Exception:
        print("[measure] no campaign roster yet")
        return
    fleet = RemoteDesk(DeskTransport())
    placed = await fleet.resolve([Demand(
        pool="main", capability="inference", base=BASE, shape=1,
        memory=SERVE_FRACTION, group=0, sharing="concurrent")])
    if not placed.get("placed"):
        print(f"[measure] no pool to measure through: {placed}")
        return
    pool = RemotePool(MetalTransport(placed["pools"]["main"]), base=BASE, tp=1)
    for rid, entry in sorted(listed.items()):
        tasks = {t.id: t for t in load_tasks(store, entry["eval"])}
        told = await measure_run(
            store, rid, gsm_measurement(sorted(tasks)), pool, tasks)
        print(f"[measure] {rid} ({entry['arm']}): {told}")
    await store_volume.commit.aio()


@app.function(image=cpu_image, schedule=modal.Period(minutes=15),
              timeout=1200)
async def reaper() -> None:
    from rlstack.runner.remote import RemoteDesk

    print(json.dumps(await RemoteDesk(DeskTransport()).reap(probes=3,
                                                            wait=30.0)))


# ---------------------------------------------------------------------------
# the doors
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def up() -> None:
    """Start the metal's shift and wait until the desk holds its plane."""
    from rlstack.runner.remote import RemoteDesk

    for name, row in METALS.items():
        call = modal.Cls.from_name(APP, row["cls"])().serve.spawn()
        print(f"[up] {name} serving: call {call.object_id}")
        print(f"[up] kill it later with: modal run deploy/gsm_a100.py::stop "
              f"--call-id {call.object_id}")
    fleet = RemoteDesk(DeskTransport())
    for _ in range(90):
        held = fleet.status().get("metal", {})
        if all(name in held for name in METALS):
            break
        time.sleep(10)
    print(json.dumps(fleet.status(), indent=2))


@app.local_entrypoint()
async def screen_now() -> None:
    print(json.dumps(await screen.remote.aio(), indent=1)[:2000])


@app.local_entrypoint()
async def campaign() -> None:
    """The four arms through the desk; the roster lands at RUNS_KEY and each
    accepted run is tagged (name + taxonomy) in annotations.jsonl."""
    from rlstack.runner.remote import RemoteDesk, spec_from_json

    told = await build_campaign.remote.aio()
    rows, eval_uri = told["specs"], told["eval"]
    fleet = RemoteDesk(DeskTransport())
    roster: dict = {}
    for name in sorted(rows):
        reply = await fleet.submit(spec_from_json(rows[name]), subdir="gsm")
        print(f"[{name}] accepted={reply.get('accepted')} "
              f"run={reply.get('run_id')} host={reply.get('host')} "
              f"reply={json.dumps(reply, default=str)[:300]}")
        if reply.get("accepted"):
            roster[reply["run_id"]] = {"arm": name, "eval": eval_uri}
    await write_roster.remote.aio(roster)
    print(f"[campaign] roster of {len(roster)} runs written to {RUNS_KEY}")


@app.local_entrypoint()
def status() -> None:
    from rlstack.runner.remote import RemoteDesk, RemoteMetal

    fleet = RemoteDesk(DeskTransport())
    told = fleet.status()
    residuals = {}
    rosters = {}
    for name, row in METALS.items():
        scheme = row["scheme"]
        try:
            residuals[name] = RemoteMetal(
                MetalPlaneTransport(f"{scheme}://metal")).residual()
            rosters.update(modal.Cls.from_name(APP, row["cls"])().roster.remote())
        except Exception as silent:
            residuals[name] = f"silent: {silent}"
    print(json.dumps({
        "listings": told["listings"], "metal": told["metal"],
        "residual": residuals,
        "liveness": fleet.liveness(),
        "roster": rosters}, indent=2))


@app.local_entrypoint()
async def measure_now() -> None:
    await measure.remote.aio()


@app.local_entrypoint()
def stop(call_id: str) -> None:
    modal.FunctionCall.from_id(call_id).cancel()
    print(f"cancelled {call_id}")
