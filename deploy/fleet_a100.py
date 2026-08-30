"""One A100 the DESK owns end to end, and the gated-KL question running on it.

    MODAL_PROFILE=yu-masala-workspace modal deploy deploy/fleet_a100.py
    ... run deploy/fleet_a100.py::up           # bare metal registers its plane
    ... run deploy/fleet_a100.py::pair         # BOTH arms through the desk
    ... run deploy/fleet_a100.py::status       # listings, residual, roster
    ... run deploy/fleet_a100.py::reap         # the janitor's sweep, by hand
    ... run deploy/fleet_a100.py::migrate --run-ids <rid>   # code-refresh pass
    ... run deploy/fleet_a100.py::stop --call-id <id>       # kill the metal

THE VENUE: the desk is its own warm CPU container (memory = the fleet
journal, rebuilt every boot); the A100 container wears ONLY a MetalService —
no standing hosts, the whole device is residual — so every host on it is a
desk-issued CARVE, booked at this container's door before the build. The
`reaper` sweeps on a schedule: probe, retry (a knock boots a stopped-but-
deployed container, so a mere reboot reads as recovered), and only what
stays silent is decarved + delisted. The three transports below are this
venue's whole contribution (I5).

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

SERVE_FRACTION = 0.42
LEARN_FRACTION = 0.50

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
        return await self.handle().fleet.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return self.handle().fleet_ask.remote(verb, payload)


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
        from rlstack.runner.fleet import FleetService
        from rlstack.runner.remote import RemoteHost, RemoteMetal

        class DeskStore(ModalVolumeStore):
            """The desk's one extra durable point: cas blobs it writes
            (migrate's sliced plans) must be READABLE by the metal container
            before the adopt that names them — so they commit as written."""

            def _write(self, key: str, data: bytes) -> None:
                super()._write(key, data)
                if key.startswith("cas/"):
                    self._persist()

        self.desk = FleetService.from_journal(
            DeskStore("/store", volume=store_volume, locator=STORE),
            connect=lambda address: RemoteHost(MetalTransport(address)),
            connect_metal=lambda address: RemoteMetal(
                MetalPlaneTransport(address)))
        print(f"[desk] rebuilt from journal: {sorted(self.desk.listings)} "
              f"/ metal plane: {sorted(self.desk.metal_remotes)}")

    @modal.method()
    async def fleet(self, verb: str, payload: dict) -> dict:
        store_volume.reload()      # other containers' ledgers, seen fresh
        return await self.desk.serve(verb, payload)

    @modal.method()
    def fleet_ask(self, verb: str, payload: dict) -> dict:
        return self.desk.answer(verb, payload)


# ---------------------------------------------------------------------------
# the metal: the sweep's two fractional hosts, listed over the wire
# ---------------------------------------------------------------------------

@app.cls(image=gpu_image, gpu="A100-40GB",
         volumes={"/store": store_volume, "/hf": hf_cache},
         timeout=86400, scaledown_window=900, max_containers=1)
@modal.concurrent(max_inputs=32)
class Metal:
    @modal.enter()
    def bring_up(self) -> None:
        from rlstack import ModalVolumeStore
        from rlstack.policy.siteschema import hf_schema
        from rlstack.runner.engines.vllm_engine import VllmEngine
        from rlstack.runner.fleet import Metal as OwnedMetal, MetalService
        from rlstack.runner.learners.torch_learner import TorchLearner
        from rlstack.runner.remote import LocalTransport

        self.store = ModalVolumeStore("/store", volume=store_volume,
                                      locator=STORE)
        ensure_tasks(self.store)

        # THE METAL PLANE, and NOTHING standing: the container wears only its
        # books and factories, so the WHOLE device is residual and every host
        # on it is a desk-issued carve — the space is the desk's to give.
        # The factory pays for both adapter families up front (the sweep's
        # rule: plora's demands are the wider sizing).
        self.metal_service = MetalService(
            OwnedMetal(METAL_NAME, "A100-40GB", 1, 40.0), store=self.store,
            engine_factory=lambda regime, partition: VllmEngine(
                regime.base, tp=regime.shape,
                gpu_memory_utilization=partition.memory,
                max_model_len=1536, max_bundles=16, max_rank=16,
                max_members=MEMBERS, cas_get=self.store.cas_get,
                serves=("lora", "plora")),
            learner_factory=lambda regime, partition: TorchLearner(),
            address_of=lambda name: f"a100://{name}",
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
        """The container's standing shift: register the metal plane, then
        hold the door open while carved hosts work — a stats task follows
        every host the books grow. Stopping THIS call kills the metal."""
        import asyncio

        from rlstack.runner.remote import RemoteFleet

        fleet = RemoteFleet(DeskTransport())
        try:
            await fleet.register_metal(METAL_NAME, "A100-40GB", 1, 40.0,
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
# the arm (the sweep's spec_for, lora branch only) and the task set
# ---------------------------------------------------------------------------

def pair_plans(store, updates: int):
    """The shared plans: every update rolls the ONE screened problem
    (GROUPS_PER_WAVE groups of GROUP_SIZE — the overfitting pressure is the
    point), and every EVAL_EVERY updates one eval wave measures the SAME
    EVAL_HELD_OUT problems, EVAL_SAMPLES deep. One wave per eval POINT, the
    #61 lesson."""
    from rlstack import GroupPlan, Plans, RunPlan, Sample, WavePlan, WaveRef, encode
    from rlstack.data.tasks import load_tasks

    wave = WavePlan(tuple(
        GroupPlan(f"{TRAIN_TASK_ID}#{g}",
                  tuple(Sample(TRAIN_TASK_ID, "dapo_math")
                        for _ in range(GROUP_SIZE)))
        for g in range(GROUPS_PER_WAVE)))
    held_out = [t.id for t in load_tasks(store, EVAL_TASKS)][:EVAL_HELD_OUT]
    measured = WavePlan(tuple(
        GroupPlan(task, tuple(Sample(task, "dapo_math")
                              for _ in range(EVAL_SAMPLES)))
        for task in held_out))
    return Plans(
        train=store.cas_put(encode(RunPlan(tuple(
            WaveRef(f"self://rollouts/{u}") for u in range(1, updates + 1))))),
        rollout=store.cas_put(encode(RunPlan((wave,) * updates))),
        eval=store.cas_put(encode(RunPlan(
            (measured,) * (updates // EVAL_EVERY)))))


def pair_specs(store, updates: int):
    """The two arms as values: identical everywhere but the bank and the
    loss. Returns {"lora": spec, "gated": spec}."""
    from rlstack import (
        AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
        GpuSet, LearnerMember, OptimSpec, PolicySpec, PoolMember,
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
            eval=EvalSpec(every=EVAL_EVERY, post=("final_answer",)),
            gpu_config=GpuConfig(groups=(
                GpuGroup(gpus=GpuSet(n=1), members=(
                    PoolMember("main", tp=1, fraction=SERVE_FRACTION),)),
                GpuGroup(gpus=GpuSet(n=1), members=(
                    LearnerMember(fsdp=1, fraction=LEARN_FRACTION),)))),
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
    from rlstack.runner.remote import RemoteFleet

    print(json.dumps(await RemoteFleet(DeskTransport()).reap(probes=3,
                                                             wait=30.0)))


# ---------------------------------------------------------------------------
# the doors
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def up() -> None:
    """Start the metal's shift and wait until the desk holds the metal —
    nothing stands: listings appear only when placements carve them."""
    from rlstack.runner.remote import RemoteFleet

    call = metal_handle().serve.spawn()
    print(f"[up] metal serving: call {call.object_id}")
    print(f"[up] kill it later with: modal run deploy/fleet_a100.py::stop "
          f"--call-id {call.object_id}")
    fleet = RemoteFleet(DeskTransport())
    for _ in range(90):
        if METAL_NAME in fleet.status().get("metal", {}):
            break
        time.sleep(10)
    print(json.dumps(fleet.status(), indent=2))


@app.local_entrypoint()
async def pair(updates: int = 400) -> None:
    """THE EXPERIMENT: both arms through the desk, one frame each. The first
    submit CARVES the serve and learner hosts out of the bare metal (the
    engine builds inside that frame — minutes); the second JOINS the same
    hosts, so both arms share one engine and one learner."""
    from rlstack.runner.remote import RemoteFleet, spec_from_json

    rows = metal_handle().build_pair_here.remote(updates)
    fleet = RemoteFleet(DeskTransport())
    for name in ("lora", "gated"):
        reply = await fleet.submit(spec_from_json(rows[name]))
        print(f"[{name}]", json.dumps(reply, indent=2))


@app.local_entrypoint()
def status() -> None:
    from rlstack.runner.remote import RemoteFleet, RemoteMetal

    fleet = RemoteFleet(DeskTransport())
    told = fleet.status()
    print(json.dumps({
        "listings": told["listings"], "metal": told["metal"],
        "residual": RemoteMetal(
            MetalPlaneTransport(METAL_ADDRESS)).residual(),
        "liveness": fleet.liveness(),
        "roster": metal_handle().roster.remote()}, indent=2))


@app.local_entrypoint()
async def reap(probes: int = 3, wait: float = 30.0) -> None:
    """The janitor's sweep, by hand — same verb the scheduled reaper runs."""
    from rlstack.runner.remote import RemoteFleet

    print(json.dumps(await RemoteFleet(DeskTransport()).reap(probes, wait),
                     indent=2))


@app.local_entrypoint()
async def migrate(run_ids: str, optim: str = "load",
                  full_plan: bool = False) -> None:
    """The refresh's second half: warm-fork the named runs onto whatever
    code is deployed NOW."""
    from rlstack.runner.remote import RemoteFleet

    report = await RemoteFleet(DeskTransport()).migrate(
        run_ids.split(","), optim=optim, remaining_only=not full_plan)
    print(json.dumps(report, indent=2))


@app.local_entrypoint()
def stop(call_id: str) -> None:
    """Kill the metal's standing shift mid-run — the refresh test's crash."""
    modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
    print(f"[stop] cancelled {call_id} (containers terminated)")
