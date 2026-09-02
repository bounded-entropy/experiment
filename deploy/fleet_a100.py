"""One A100 the DESK owns end to end, and the gated-KL question running on it.

    MODAL_PROFILE=yu-masala-workspace modal deploy deploy/fleet_a100.py
    ... run deploy/fleet_a100.py::up           # boot the metal; it registers itself
    ... run deploy/fleet_a100.py::pair         # BOTH arms through the desk
    ... run deploy/fleet_a100.py::status       # listings, residual, roster
    ... run deploy/fleet_a100.py::reap         # the janitor's sweep, by hand
    ... run deploy/fleet_a100.py::migrate --run-ids <rid>   # code-refresh pass
    ... run deploy/fleet_a100.py::stop --call-id <id>       # release the metal

THE VENUE: the desk is its own warm CPU container (memory = the fleet
journal, rebuilt every boot); the A100 container wears ONLY a MetalService —
no standing hosts, the whole device is residual — so every host on it is a
desk-issued CARVE, booked at this container's door before the build. The
metal MEASURES its card and REGISTERS ITSELF at bring-up (ADR 0001); the
`reaper` sweeps on a schedule: probe, retry (a knock boots a stopped-but-
deployed container, so a mere reboot reads as recovered), and what stays
silent is decarved + delisted, its metal knocked and its runs rerouted or
parked for the reborn metal's own registration to retry. The three
transports below are this venue's whole contribution (I5).

THE EXPERIMENT (the pair): one screened DAPO problem trained 400 updates —
deliberate overfitting pressure — measured every 5 updates on 10 held-out
problems. Arm "lora" is the sweep's best plain arm (grpo); arm "gated" is
plora with grpo_latent_kl_gated (#66: no prior pull until a group is
solved). The question: does held-out eval DEGRADE for plain lora but
STABILIZE when the KL is earned? First submit carves serve+learn hosts;
the second joins them — one engine, one learner, both arms.
"""

import json
import time

import modal

APP = "rlstack-fleet-a100"
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
TRAIN_TASKS = "cas://09499d32b51e5e1b2a644b1c65e01b44aa42ff1a5bfac78ead41f98f89f09c93"
TRAIN_TASK_ID = "dapo-math-17k/a6d38312-86c7-4022-b8d2-adcf19fa0c3a"

SITE = "layers.*.self_attn.*"
GROUPS_PER_WAVE = 2
GROUP_SIZE = 8
MAX_TOKENS = 512

# THE PAIR (one question, two arms): does held-out eval DEGRADE under plain
# one-task GRPO but STABILIZE when the prior's pull is earned? Same base,
# same screened train task, same engine, same seed — the arm is the only
# difference. Hyperparameters are each family's best from the 40-arm sweep.
EVAL_TASKS = "cas://82ae4626dbb59a2c50e2b13cbe7250c5f1ddd02dfb81edc7495efb77759d420b"
EVAL_EVERY = 5
EVAL_HELD_OUT = 10                   # problems measured, fixed for the run
EVAL_SAMPLES = 4                     # completions per problem per point
SEED = 11
LORA = {"r": 16, "lr": 1e-4}
PLORA = {"k": 8, "latent": 64, "prior_std": 0.05, "lr": 3e-4}
MEMBERS = 4

# Memory in GB, TOTAL per member (one shard each): the spec declares it,
# the desk books it, the metal converts against the card it measured
SERVE_GB = 16.8
LEARN_GB = 20.0

METAL_NAME = "modal-a100"
METAL_ADDRESS = "a100://metal"       # the metal PLANE: carve/decarve/residual


# ---------------------------------------------------------------------------
# the venue's transports (I5): desk-by-name, host-by-address
# ---------------------------------------------------------------------------

def desk_handle():
    return modal.Cls.from_name(APP, "Desk")()


def metal_handle():
    return modal.Cls.from_name(APP, "Metal")()


class DeskTransport:
    """The Transport contract over the desk's cls handle, looked up by name —
    any process in the workspace reaches the SAME standing desk."""

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
    """The Transport contract to ONE host inside the metal container: the
    address rides every frame and the container routes it. Lazy lookup, so a
    desk rebuilt from an old journal can hold listings whose container is
    gone — alive() discovers that, nothing crashes."""

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
    """The Transport contract to the metal PLANE of the same container —
    carve/decarve/residual, the verbs that create and free hosts rather than
    talk to one. One metal today; the address rides anyway so a second metal
    is a second cls, not a new desk."""

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
        from rlstack.runner.desk import Desk
        from rlstack.runner.remote import RemoteHost, RemoteMetal

        class DeskStore(ModalVolumeStore):
            """The desk's one extra durable point: cas blobs it writes
            (migrate's sliced plans) must be READABLE by the metal container
            before the adopt that names them — so they commit as written."""

            def _write(self, key: str, data: bytes) -> None:
                super()._write(key, data)
                if key.startswith("cas/"):
                    self._persist()

        self.desk = Desk.from_journal(
            DeskStore("/store", volume=store_volume, locator=STORE),
            host_for=lambda address: RemoteHost(MetalTransport(address)),
            metal_for=lambda address: RemoteMetal(
                MetalPlaneTransport(address)))
        # the composed door: the blind desk plus its spec-aware sidecar
        # (campaign.py) — one Transport surface, migrate included
        from rlstack.runner.campaign import Campaigns
        self.door = Campaigns(self.desk)
        print(f"[desk] rebuilt from journal: {sorted(self.desk.listings)} "
              f"/ metal plane: {sorted(self.desk.metal_remotes)}")

    @modal.method()
    async def desk(self, verb: str, payload: dict) -> dict:
        store_volume.reload()      # other containers' ledgers, seen fresh
        return await self.door.serve(verb, payload)

    @modal.method()
    def desk_ask(self, verb: str, payload: dict) -> dict:
        return self.door.answer(verb, payload)


# ---------------------------------------------------------------------------
# the metal: the sweep's two fractional hosts, listed over the wire
# ---------------------------------------------------------------------------

@app.cls(image=gpu_image, gpu="A100-40GB",
         volumes={"/store": store_volume, "/hf": hf_cache},
         timeout=86400, scaledown_window=900, max_containers=1)
@modal.concurrent(max_inputs=32)
class Metal:
    @modal.enter()
    async def bring_up(self) -> None:
        import asyncio

        from rlstack import ModalVolumeStore
        from rlstack.policy.siteschema import hf_schema
        from rlstack.runner.desk import MetalService
        from rlstack.runner.remote import LocalTransport
        from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

        self.store = ModalVolumeStore("/store", volume=store_volume,
                                      locator=STORE)
        ensure_tasks(self.store)

        # THE METAL PLANE, and NOTHING standing: the container wears only its
        # books and its recipe, so the WHOLE device is residual and every host
        # on it is a desk-issued carve — the space is the desk's to give.
        # The recipe (ADR 0002) pays for both adapter families up front (the
        # sweep's rule: plora's demands are the wider sizing); every resident
        # a carve births is a child process pinned and capped to its partition.
        # The card is MEASURED, never typed (ADR 0001, Q6).
        self.metal_service = MetalService(
            MetalService.measure(METAL_NAME), store=self.store,
            builds=Builds(
                engine=EngineBuild(max_model_len=1536, max_bundles=16,
                                   max_rank=16, max_members=MEMBERS,
                                   serves=("lora", "plora")),
                learner=LearnerBuild()),
            address_of=lambda name: f"a100://{name}",
            schema_for=hf_schema,
            transport_for=lambda address: LocalTransport(
                self.metal_service.service_for(address)))
        metal = self.metal_service.metal
        print(f"[metal] up: {METAL_NAME} bare: {metal.gpu} x{metal.devices} "
              f"at {metal.vram_gb:g} GB; residual {self.metal_service.residual()}")
        # the duties run on this container's loop from birth (ADR 0001, Q5a/
        # Q5b): an async enter, so the announce is a task and never a
        # blocking wait on the desk — whose reply may carve on THIS container
        self.duties = asyncio.create_task(self.metal_duties())

    async def announce(self) -> None:
        """The metal REGISTERS ITSELF the moment it exists: measured card,
        plane address, recipe. A reborn container announces the same way and
        the desk reaps the previous generation's corpses and retries their
        runs; a name taken at ANOTHER address is refused, loudly."""
        from rlstack.runner.remote import RemoteDesk

        metal = self.metal_service.metal
        told = await RemoteDesk(DeskTransport()).register_metal(
            metal.name, metal.gpu, metal.devices, metal.vram_gb, METAL_ADDRESS,
            builds=self.metal_service.builds.row())
        print(f"[metal] registered on the metal plane: {json.dumps(told)}")

    async def metal_duties(self) -> None:
        """The metal's own duties on its own loop: announce, then a stats
        task following every carved host and the volume commit tick. Nothing
        here depends on a spawned input surviving a preempt."""
        import asyncio

        try:
            await self.announce()
        except Exception as refused:
            print(f"[metal] REGISTRATION REFUSED: {refused}", flush=True)
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
                    await store_volume.commit.aio()
        finally:
            for task in stats.values():
                task.cancel()

    @modal.method()
    async def host(self, address: str, verb: str, payload: dict) -> dict:
        if verb == "adopt":
            store_volume.reload()  # a migrated child's plans are desk-written
        return await self.metal_service.service_for(address).serve(verb,
                                                                   payload)

    @modal.method()
    def host_ask(self, address: str, verb: str, payload: dict) -> dict:
        return self.metal_service.service_for(address).answer(verb, payload)

    @modal.method()
    async def metal(self, verb: str, payload: dict) -> dict:
        if verb == "carve":
            store_volume.reload()  # a carve may read desk-written cas blobs
        return await self.metal_service.serve(verb, payload)

    @modal.method()
    def metal_ask(self, verb: str, payload: dict) -> dict:
        return self.metal_service.answer(verb, payload)

    @modal.method()
    def build_pair_here(self, updates: int) -> dict:
        """Both arms as canonical spec rows, built INSIDE the container so
        the plans' cas blobs (and the SVD factors) land on the store this
        container reads."""
        from rlstack.spec.canonical import canonical_json

        rows = {name: json.loads(canonical_json(spec))
                for name, spec in pair_specs(self.store, updates).items()}
        store_volume.commit()
        return rows

    @modal.method()
    def roster(self) -> dict:
        """Every tenancy on every host this container's books hold — carved
        hosts included, which is all of them now."""
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
        """The KEEPALIVE only: an input in flight holds the container open
        while its hosts carry work, and a spawned input reschedules onto a
        fresh container after a preempt — where bring_up has already
        registered and started the duties. Stopping THIS call releases the
        metal."""
        import asyncio

        while True:
            await asyncio.sleep(60)

    @modal.exit()
    def bring_down(self) -> None:
        self.duties.cancel()
        # every resident down the ladder (ADR 0002): a learner's chorus and
        # an engine's core end with the container, bounded
        for teardown in self.metal_service.shutdown():
            if not teardown.graceful:
                print(teardown.line(), flush=True)


# ---------------------------------------------------------------------------
# the arm (the sweep's spec_for, lora branch only) and the task set
# ---------------------------------------------------------------------------

def pair_plans(store, updates: int):
    """The shared plans: every update rolls the ONE screened problem
    (GROUPS_PER_WAVE groups of GROUP_SIZE — the overfitting pressure is the
    point). Measurement has no plan here (#70): the held-out reading is
    pair_eval.py's Measurement, outside the runs."""
    from rlstack import GroupPlan, Plans, RunPlan, Sample, WavePlan, WaveRef, encode

    wave = WavePlan(tuple(
        GroupPlan(f"{TRAIN_TASK_ID}#{g}",
                  tuple(Sample(TRAIN_TASK_ID, "dapo_math")
                        for _ in range(GROUP_SIZE)))
        for g in range(GROUPS_PER_WAVE)))
    return Plans(
        train=store.cas_put(encode(RunPlan(tuple(
            WaveRef(f"self://rollouts/{u}") for u in range(1, updates + 1))))),
        rollout=store.cas_put(encode(RunPlan((wave,) * updates))))


def pair_specs(store, updates: int):
    """The two arms as values: identical everywhere but the bank and the
    loss. Returns {"lora": spec, "gated": spec}."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, GenSpec, Topology, HostSpec,
        LearnerMember, OptimSpec, PolicySpec, PoolMember,
        SamplingSpec, Schedule, Seeds, lora, plora,
    )

    plans = pair_plans(store, updates)
    factors = ensure_factors(store)

    def arm(bank, loss, post, lr, overrides):
        return ExperimentSpec(
            policy=PolicySpec(base=BASE, bank=bank),
            gen=GenSpec(envs=("dapo_math",), tasks=(TRAIN_TASKS, EVAL_TASKS),
                        sampling=SamplingSpec(temperature=1.0,
                                              max_tokens=MAX_TOKENS)),
            plans=plans,
            algo=AlgoSpec(loss=loss, post=post,
                          optim=OptimSpec("adamw", lr=lr, weight_decay=0.0,
                                          overrides=overrides),
                          schedule=Schedule(microbatch_tokens=512,
                                            max_policy_lag=1)),
            topology=Topology(hosts=(
                HostSpec((PoolMember("main", tp=1, vram_gb=SERVE_GB),)),
                HostSpec((LearnerMember(fsdp=1, vram_gb=LEARN_GB),)))),
            seeds=Seeds(master=SEED))

    return {
        "lora": arm({"pi": lora(SITE, r=LORA["r"])},
                    "grpo", ("final_answer", "grpo_advantage"),
                    LORA["lr"], {}),
        "gated": arm({"pi": plora(SITE, k=PLORA["k"], latent=PLORA["latent"],
                                  members=MEMBERS,
                                  prior_std=PLORA["prior_std"],
                                  factors=factors)},
                     "grpo_latent_kl_gated",
                     ("final_answer", "grpo_advantage", "group_accuracy"),
                     PLORA["lr"], {"pi.mapper": {"weight_decay": 1e-2}}),
    }


def ensure_factors(store) -> str:
    """The frozen SVD artifact for PLORA's k, built if this store has not
    seen it — content addressing makes the rebuild free to be wrong about."""
    from rlstack.policy.adapters.plora_factors import (
        build_factors, hf_weight_reader,
    )
    from rlstack.policy.siteschema import hf_schema, resolve

    sites = resolve(hf_schema(BASE).sites, SITE)
    return store.cas_put(
        build_factors(BASE, sites, PLORA["k"], hf_weight_reader(BASE)))


def ensure_tasks(store) -> None:
    try:
        store.cas_get(TRAIN_TASKS)
    except FileNotFoundError:
        from rlstack.__main__ import build_task_sets
        build_task_sets(store, "dapo_math", {"train": 0.98, "eval": 0.02}, 17)


# ---------------------------------------------------------------------------
# the reaper: liveness delisting on a cadence
# ---------------------------------------------------------------------------

@app.function(image=cpu_image, schedule=modal.Period(minutes=15),
              timeout=1200)
async def reaper() -> None:
    """The janitor's sweep, on the clock: every listing probed by the desk,
    the silent retried (on Modal the knock itself boots a stopped-but-
    deployed container — a call queues until bring_up answers, so a mere
    reboot reads as alive/recovered, never reaped), and only a torn-down
    deployment stays silent long enough to be decarved and delisted."""
    from rlstack.runner.remote import RemoteDesk

    print(json.dumps(await RemoteDesk(DeskTransport()).reap(probes=3,
                                                             wait=30.0)))


# ---------------------------------------------------------------------------
# the doors
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def up() -> None:
    """Boot the metal (spawning the keepalive is the knock) and wait until it
    has registered itself — nothing stands: listings appear only when
    placements carve them."""
    from rlstack.runner.remote import RemoteDesk

    call = metal_handle().serve.spawn()
    print(f"[up] metal serving: call {call.object_id}")
    print(f"[up] kill it later with: modal run deploy/fleet_a100.py::stop "
          f"--call-id {call.object_id}")
    desk = RemoteDesk(DeskTransport())
    for _ in range(90):
        if METAL_NAME in desk.status().get("metal", {}):
            break
        time.sleep(10)
    print(json.dumps(desk.status(), indent=2))


@app.local_entrypoint()
async def pair(updates: int = 400) -> None:
    """THE EXPERIMENT: both arms through the desk, one frame each. The first
    submit CARVES the serve and learner hosts out of the bare metal (the
    engine builds inside that frame — minutes); the second JOINS the same
    hosts, so both arms share one engine and one learner."""
    from rlstack.runner.remote import RemoteDesk, spec_from_json

    rows = metal_handle().build_pair_here.remote(updates)
    desk = RemoteDesk(DeskTransport())
    for name in ("lora", "gated"):
        reply = await desk.submit(spec_from_json(rows[name]))
        print(f"[{name}]", json.dumps(reply, indent=2))


@app.local_entrypoint()
def status() -> None:
    from rlstack.runner.remote import RemoteDesk, RemoteMetal

    desk = RemoteDesk(DeskTransport())
    told = desk.status()
    print(json.dumps({
        "listings": told["listings"], "metal": told["metal"],
        "residual": RemoteMetal(
            MetalPlaneTransport(METAL_ADDRESS)).residual(),
        "liveness": desk.liveness(),
        "roster": metal_handle().roster.remote()}, indent=2))


@app.local_entrypoint()
async def reap(probes: int = 3, wait: float = 30.0) -> None:
    """The janitor's sweep, by hand — same verb the scheduled reaper runs."""
    from rlstack.runner.remote import RemoteDesk

    print(json.dumps(await RemoteDesk(DeskTransport()).reap(probes, wait),
                     indent=2))


@app.local_entrypoint()
async def migrate(run_ids: str, optim: str = "load",
                  full_plan: bool = False) -> None:
    """The refresh's second half: warm-fork the named runs onto whatever
    code is deployed NOW."""
    from rlstack.runner.remote import RemoteDesk

    report = await RemoteDesk(DeskTransport()).migrate(
        run_ids.split(","), optim=optim, remaining_only=not full_plan)
    print(json.dumps(report, indent=2))


@app.local_entrypoint()
def stop(call_id: str) -> None:
    """Kill the metal's standing shift mid-run — the refresh test's crash."""
    modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
    print(f"[stop] cancelled {call_id} (containers terminated)")
