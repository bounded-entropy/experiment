"""Two A100s the DESK owns end to end, and the DSL campaign running on them.

    MODAL_PROFILE=yu-masala-workspace modal deploy deploy/dsl_a100.py
    ... run deploy/dsl_a100.py::up            # bare metal registers its plane
    ... run deploy/dsl_a100.py::campaign      # all twelve arms through the desk
    ... run deploy/dsl_a100.py::status        # listings, residual, roster
    ... run deploy/dsl_a100.py::measure_now   # one measurement pass, by hand
    ... run deploy/dsl_a100.py::stop --call-id <id>    # kill the metal

THE VENUE (fleet_a100's, at two devices): the desk is its own warm CPU
container; the A100 container wears ONLY a MetalService over BOTH devices —
no standing hosts, the whole metal is residual — so every host on it is a
desk-issued carve. The first submit carves the serve host (device 0) and the
learner host (device 1); the other eleven join them: one engine, one
learner, twelve tenants.

THE EXPERIMENT: two invented tool DSLs (the Stamp Office and the Glyph
Exchange — the rulebook fits in the prompt, the verbs exist nowhere in
pretraining), each trained on its TWO train requests only and measured on
the full generalization sweep every 5 updates. Six arms per DSL:

    grpo      lora r=16, plain GRPO                      — the baseline
    svd       plora k=8, gated latent KL, SVD basis      — the frozen frame
    nosvd     plora k=8, gated latent KL, random basis   — the frame control
    spectral  spectral k=16, plain GRPO                  — SVF, no latent
    slatent   spectral_latent k=16, gated latent KL      — SVF with the latent
    sdpo      lora r=16, the reflect loop, sdpo          — the loop teaches

The `measure` cron backfills measurements/<rid>/heldout for every campaign
run — ALWAYS under the plain single-turn environment, sdpo included: the
question is what reached the weights, not the context.
"""

import json
import time

import modal

APP = "rlstack-dsl-a100"
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

BASE = "Qwen/Qwen3-0.6B"
STORE = "modal://rlstack-store"
SITE = "layers.*.self_attn.*"

UPDATES = 200                  # the five RL arms' plan length
LOOPS = 120                    # the sdpo arm: one loop = attempt + 2 reflects
GROUP_SIZE = 8
MAX_TOKENS = 256
SEED = 11

PLORA = {"k": 8, "latent": 64, "prior_std": 0.05, "members": 4}
RANDOM_BASIS_SEED = 7
SPECTRAL_K = 16

EVAL_EVERY = 5
EVAL_SAMPLES = 2
EVAL_TEMPERATURE = 0.2
RUNS_KEY = "measurements/dsl/runs.json"    # {run_id: family} — the cron's list

SERVE_FRACTION = 0.85
LEARN_FRACTION = 0.85

METAL_NAME = "modal-2xa100"
METAL_ADDRESS = "dsl://metal"


# ---------------------------------------------------------------------------
# the venue's transports (I5): desk-by-name, host-by-address
# ---------------------------------------------------------------------------

def desk_handle():
    return modal.Cls.from_name(APP, "Desk")()


def metal_handle():
    return modal.Cls.from_name(APP, "Metal")()


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
        return self.handle().desk_ask.remote(verb, payload)


class MetalTransport:
    """One host inside the metal container, addressed per frame; lazy lookup
    so a desk rebuilt from an old journal holds dead listings harmlessly."""

    def __init__(self, address: str) -> None:
        self.address = address
        self._handle = None

    def handle(self):
        if self._handle is None:
            self._handle = metal_handle()
        return self._handle

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.handle().host.remote.aio(self.address, verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return self.handle().host_ask.remote(self.address, verb, payload)


class MetalPlaneTransport:
    def __init__(self, address: str) -> None:
        self.address = address
        self._handle = None

    def handle(self):
        if self._handle is None:
            self._handle = metal_handle()
        return self._handle

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.handle().metal.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return self.handle().metal_ask.remote(verb, payload)


# ---------------------------------------------------------------------------
# the desk: a warm CPU container whose whole memory is the fleet journal
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

        class DeskStore(ModalVolumeStore):
            """cas blobs the desk writes (sliced plans) must be readable by
            the metal container before the adopt that names them."""

            def _write(self, key: str, data: bytes) -> None:
                super()._write(key, data)
                if key.startswith("cas/"):
                    self._persist()

        self.desk = Desk.from_journal(
            DeskStore("/store", volume=store_volume, locator=STORE),
            connect=lambda address: RemoteHost(MetalTransport(address)),
            connect_metal=lambda address: RemoteMetal(
                MetalPlaneTransport(address)))
        self.door = Campaigns(self.desk)
        print(f"[desk] rebuilt from journal: {sorted(self.desk.listings)} "
              f"/ metal plane: {sorted(self.desk.metal_remotes)}")

    @modal.method()
    async def desk(self, verb: str, payload: dict) -> dict:
        store_volume.reload()
        return await self.door.serve(verb, payload)

    @modal.method()
    def desk_ask(self, verb: str, payload: dict) -> dict:
        return self.door.answer(verb, payload)


# ---------------------------------------------------------------------------
# the metal: two bare devices, every host on them a desk-issued carve
# ---------------------------------------------------------------------------

@app.cls(image=gpu_image, gpu="A100-40GB:2",
         volumes={"/store": store_volume, "/hf": hf_cache},
         timeout=86400, scaledown_window=900, max_containers=1)
@modal.concurrent(max_inputs=64)
class Metal:
    @modal.enter()
    def bring_up(self) -> None:
        from rlstack import ModalVolumeStore
        from rlstack.policy.siteschema import hf_schema
        from rlstack.runner.desk import Metal as OwnedMetal, MetalService
        from rlstack.runner.engines.vllm_engine import VllmEngine
        from rlstack.runner.learners.torch_learner import TorchLearner
        from rlstack.runner.remote import LocalTransport

        self.store = ModalVolumeStore("/store", volume=store_volume,
                                      locator=STORE)
        self.metal_service = MetalService(
            OwnedMetal(METAL_NAME, "A100-40GB", 2, 40.0), store=self.store,
            engine_factory=lambda regime, partition: VllmEngine(
                regime.base, tp=regime.shape,
                gpu_memory_utilization=partition.memory,
                max_model_len=2048, max_bundles=32, max_rank=16,
                max_members=PLORA["members"], cas_get=self.store.cas_get,
                serves=("lora", "plora", "spectral", "spectral_latent")),
            learner_factory=lambda regime, partition: TorchLearner(),
            address_of=lambda name: f"dsl://{name}",
            schema_for=hf_schema,
            dial=lambda address: LocalTransport(
                self.metal_service.service_for(address)),
            release=lambda host: [engine.shutdown()
                                  for engine in host.engines])
        print(f"[metal] up: {METAL_NAME} bare; "
              f"residual {self.metal_service.residual()}")

    @modal.method()
    async def host(self, address: str, verb: str, payload: dict) -> dict:
        if verb == "adopt":
            store_volume.reload()
        return await self.metal_service.service_for(address).serve(verb,
                                                                   payload)

    @modal.method()
    def host_ask(self, address: str, verb: str, payload: dict) -> dict:
        return self.metal_service.service_for(address).answer(verb, payload)

    @modal.method()
    async def metal(self, verb: str, payload: dict) -> dict:
        if verb == "carve":
            store_volume.reload()
        return await self.metal_service.serve(verb, payload)

    @modal.method()
    def metal_ask(self, verb: str, payload: dict) -> dict:
        return self.metal_service.answer(verb, payload)

    @modal.method()
    def build_campaign_here(self) -> dict:
        """Every arm as a canonical spec row, built INSIDE the container so
        the task sets, plans and plora factor artifacts land on the store
        this container reads. Also returns the eval-set uris the measurement
        cron reads (the cron's image carries no tokenizer)."""
        from rlstack.spec.canonical import canonical_json

        specs, evals = campaign_specs(self.store)
        rows = {name: json.loads(canonical_json(spec))
                for name, spec in specs.items()}
        store_volume.commit()
        return {"specs": rows, "evals": evals}

    @modal.method()
    def roster(self) -> dict:
        out = {}
        for service in self.metal_service.services.values():
            for rid, tenancy in sorted(service.host.roster.items()):
                entries = self.store.peek_ledger(rid)
                out[rid] = {"status": tenancy.status,
                            "host": service.host.name,
                            "committed": int(entries[-1]["update"])
                            if entries else 0}
        return out

    @modal.method()
    async def serve(self) -> None:
        """The container's standing shift: register the metal plane, then
        hold the door open while carved hosts work. Stopping THIS call (or
        `modal app stop`) kills the metal."""
        import asyncio

        from rlstack.runner.remote import RemoteDesk

        fleet = RemoteDesk(DeskTransport())
        try:
            await fleet.register_metal(METAL_NAME, "A100-40GB", 2, 40.0,
                                       METAL_ADDRESS)
            print(f"[metal] registered {METAL_NAME} on the metal plane")
        except Exception as taken:
            print(f"[metal] {METAL_NAME} not re-registered: {taken}")
        stats: dict[str, asyncio.Task] = {}
        tick = 0
        try:
            while True:
                for service in list(self.metal_service.services.values()):
                    host = service.host
                    if host.name not in stats:
                        stats[host.name] = asyncio.create_task(
                            host.run_stats())
                await asyncio.sleep(60)
                tick += 1
                if tick % 5 == 0:
                    store_volume.commit()
        finally:
            for task in stats.values():
                task.cancel()

    @modal.exit()
    def bring_down(self) -> None:
        for service in self.metal_service.services.values():
            for engine in service.host.engines:
                engine.shutdown()


# ---------------------------------------------------------------------------
# the campaign: task sets, plans, and the twelve arms as values
# ---------------------------------------------------------------------------

FAMILIES = {
    "stamp": {"env": "stamp_office", "post": "stamp_grade"},
    "glyph": {"env": "glyph_exchange", "post": "glyph_grade"},
}


def chat_formatter():
    """The base's own template, thinking OFF (dapo_math's rule): applied at
    task-build time, because a prompt is content and hashes into identity."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE)

    def chat(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    return chat


def family_tasks(store, family: str):
    """(train uri, eval uri, train ids, eval ids) for one DSL family."""
    from rlstack.data.tasks.base import write_tasks
    from rlstack.data.tasks.glyph_exchange import (
        glyph_eval_tasks, glyph_train_tasks,
    )
    from rlstack.data.tasks.stamp_office import (
        stamp_eval_tasks, stamp_train_tasks,
    )

    chat = chat_formatter()
    builders = {"stamp": (stamp_train_tasks, stamp_eval_tasks),
                "glyph": (glyph_train_tasks, glyph_eval_tasks)}
    train_of, eval_of = builders[family]
    train, held = train_of(chat), eval_of(chat)
    return (write_tasks(store, train), write_tasks(store, held),
            [t.id for t in train], [t.id for t in held])


def rl_plans(store, env: str, task_ids: list[str]):
    """The five RL arms' shared shape: every update rolls both train tasks,
    GROUP_SIZE completions each."""
    from rlstack import GroupPlan, Plans, RunPlan, Sample, WavePlan, WaveRef, encode

    wave = WavePlan(tuple(
        GroupPlan(task, tuple(Sample(task, env) for _ in range(GROUP_SIZE)))
        for task in task_ids))
    return Plans(
        train=store.cas_put(encode(RunPlan(tuple(
            WaveRef(f"self://rollouts/{u}") for u in range(1, UPDATES + 1))))),
        rollout=store.cas_put(encode(RunPlan((wave,) * UPDATES))))


def loop_plans(store, env: str, task_ids: list[str]):
    """The sdpo arm's shape: each loop is attempt + two reflect waves (every
    trajectory of the previous wave reflected), training only on the
    loop-final waves — LOOPS updates over 3 * LOOPS rollouts."""
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


def ensure_factors(store, basis: str) -> str:
    """The frozen plora artifact for this basis, built if the store has not
    seen it — content addressing makes the rebuild free to be wrong about."""
    from rlstack.policy.adapters.plora_factors import (
        build_factors, hf_weight_reader,
    )
    from rlstack.policy.siteschema import hf_schema, resolve

    sites = resolve(hf_schema(BASE).sites, SITE)
    return store.cas_put(build_factors(
        BASE, sites, PLORA["k"], hf_weight_reader(BASE), basis=basis,
        basis_seed=RANDOM_BASIS_SEED if basis == "random" else 0))


def campaign_specs(store) -> tuple[dict, dict]:
    """All twelve arms as values plus the measurement's map:
    ({family}-{arm} -> ExperimentSpec, {family: eval uri})."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup, GpuSet,
        LearnerMember, OptimSpec, PolicySpec, PoolMember, SamplingSpec,
        Schedule, Seeds, lora, plora,
    )
    from rlstack.policy.adapters.spectral import spectral
    from rlstack.policy.adapters.spectral_latent import spectral_latent

    factors = {"svd": ensure_factors(store, "svd"),
               "random": ensure_factors(store, "random")}

    specs: dict = {}
    evals: dict = {}
    for family, shape in FAMILIES.items():
        train_uri, eval_uri, train_ids, _ = family_tasks(store, family)
        evals[family] = eval_uri
        env, grade = shape["env"], shape["post"]
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
                gpu_config=GpuConfig(groups=(
                    GpuGroup(gpus=GpuSet(n=1), members=(
                        PoolMember("main", tp=1, fraction=SERVE_FRACTION),)),
                    GpuGroup(gpus=GpuSet(n=1), members=(
                        LearnerMember(fsdp=1, fraction=LEARN_FRACTION),)))),
                seeds=Seeds(master=SEED))

        specs[f"{family}-grpo"] = arm(
            {"pi": lora(SITE, r=16)}, "grpo", (grade, "grpo_advantage"), 1e-4)
        specs[f"{family}-svd"] = arm(
            {"pi": plora(SITE, k=PLORA["k"], latent=PLORA["latent"],
                         members=PLORA["members"],
                         prior_std=PLORA["prior_std"],
                         factors=factors["svd"])},
            "grpo_latent_kl_gated", (grade, "grpo_advantage", "group_accuracy"),
            3e-4, {"pi.mapper": {"weight_decay": 1e-2}})
        specs[f"{family}-nosvd"] = arm(
            {"pi": plora(SITE, k=PLORA["k"], latent=PLORA["latent"],
                         members=PLORA["members"],
                         prior_std=PLORA["prior_std"],
                         factors=factors["random"], basis="random",
                         basis_seed=RANDOM_BASIS_SEED)},
            "grpo_latent_kl_gated", (grade, "grpo_advantage", "group_accuracy"),
            3e-4, {"pi.mapper": {"weight_decay": 1e-2}})
        specs[f"{family}-spectral"] = arm(
            {"pi": spectral(SITE, k=SPECTRAL_K)}, "grpo",
            (grade, "grpo_advantage"), 1e-3)
        specs[f"{family}-slatent"] = arm(
            {"pi": spectral_latent(SITE, k=SPECTRAL_K,
                                   latent=PLORA["latent"],
                                   members=PLORA["members"],
                                   prior_std=PLORA["prior_std"])},
            "grpo_latent_kl_gated", (grade, "grpo_advantage", "group_accuracy"),
            1e-3, {"pi.mapper": {"weight_decay": 1e-2}})
        specs[f"{family}-sdpo"] = arm(
            {"pi": lora(SITE, r=16)}, "sdpo", (grade,), 1e-4,
            plans=loop, envs=(env, "reflect_retry"), makers=("reflect",),
            lag=0)
    return specs, evals


# ---------------------------------------------------------------------------
# measurement: the generalization sweep, backfilled on a cadence
# ---------------------------------------------------------------------------

def family_measurement(family: str, eval_ids: list[str]):
    from rlstack import Measurement

    return Measurement(
        name="heldout", env=FAMILIES[family]["env"],
        task_ids=tuple(eval_ids), samples=EVAL_SAMPLES, every=EVAL_EVERY,
        post=(FAMILIES[family]["post"],), seed=7,
        temperature=EVAL_TEMPERATURE, max_tokens=MAX_TOKENS)


@app.function(image=cpu_image, volumes={"/store": store_volume},
              schedule=modal.Period(minutes=10), timeout=3000)
async def measure() -> None:
    """One idempotent pass per campaign run: restore each every-5th committed
    version, sample the family's WHOLE eval sweep under the PLAIN environment
    (sdpo included — iteration-0 behavior is the honest metric), grade, and
    append what is missing. measure_run skips done points, so the cron only
    ever pays for new updates."""
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
        family = entry["family"]
        tasks = {t.id: t for t in load_tasks(store, entry["eval"])}
        # phrasing 0 only per point: a third of the sweep, every color and
        # every direction still covered — the full 3-phrasing sweep stays in
        # the set for a deeper pass later
        measured_ids = sorted(i for i in tasks if i.endswith("-p0"))
        told = await measure_run(
            store, rid, family_measurement(family, measured_ids), pool, tasks)
        print(f"[measure] {rid} ({family}): {told}")
    store_volume.commit()


# ---------------------------------------------------------------------------
# the reaper: liveness delisting on a cadence
# ---------------------------------------------------------------------------

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
    """Start the metal's shift and wait until the desk holds the metal."""
    from rlstack.runner.remote import RemoteDesk

    call = metal_handle().serve.spawn()
    print(f"[up] metal serving: call {call.object_id}")
    print(f"[up] kill it later with: modal run deploy/dsl_a100.py::stop "
          f"--call-id {call.object_id}")
    fleet = RemoteDesk(DeskTransport())
    for _ in range(90):
        if METAL_NAME in fleet.status().get("metal", {}):
            break
        time.sleep(10)
    print(json.dumps(fleet.status(), indent=2))


@app.local_entrypoint()
async def campaign() -> None:
    """All twelve arms through the desk. The first submit CARVES the serve
    and learner hosts (the engine builds inside that frame — minutes); the
    rest JOIN them. The accepted run ids land in the store's campaign roster,
    which is what the measurement cron reads."""
    from rlstack.runner.remote import RemoteDesk, spec_from_json

    told = metal_handle().build_campaign_here.remote()
    rows, evals = told["specs"], told["evals"]
    fleet = RemoteDesk(DeskTransport())
    roster: dict = {}
    for name in sorted(rows):
        reply = await fleet.submit(spec_from_json(rows[name]))
        family = name.split("-")[0]
        print(f"[{name}] accepted={reply.get('accepted')} "
              f"run={reply.get('run_id')} host={reply.get('host')}")
        if reply.get("accepted"):
            roster[reply["run_id"]] = {"family": family,
                                       "eval": evals[family], "arm": name}
    write_roster.remote(roster)
    print(f"[campaign] roster of {len(roster)} runs written to {RUNS_KEY}")


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def write_roster(roster: dict) -> None:
    from rlstack import ModalVolumeStore

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    store._write(RUNS_KEY, json.dumps(roster, sort_keys=True).encode())
    store_volume.commit()


@app.local_entrypoint()
def status() -> None:
    from rlstack.runner.remote import RemoteDesk, RemoteMetal

    fleet = RemoteDesk(DeskTransport())
    told = fleet.status()
    print(json.dumps({
        "listings": told["listings"], "metal": told["metal"],
        "residual": RemoteMetal(
            MetalPlaneTransport(METAL_ADDRESS)).residual(),
        "liveness": fleet.liveness(),
        "roster": metal_handle().roster.remote()}, indent=2))


@app.local_entrypoint()
async def measure_now() -> None:
    """One measurement pass, by hand — the same body the cron runs."""
    await measure.remote.aio()


@app.local_entrypoint()
def stop(call_id: str) -> None:
    modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
    print(f"[stop] cancelled {call_id} (containers terminated)")
