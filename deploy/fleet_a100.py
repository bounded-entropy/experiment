"""The code-refresh testbed: one STANDING DESK, one A100, one familiar arm.

    MODAL_PROFILE=yu-masala-workspace modal deploy deploy/fleet_a100.py
    ... run deploy/fleet_a100.py::up           # A100 serves, phones home
    ... run deploy/fleet_a100.py::submit       # the sweep's best lora arm, long
    ... run deploy/fleet_a100.py::status       # listings, residual, probes
    ... run deploy/fleet_a100.py::reap         # the janitor's sweep, by hand

then edit code and watch what a refresh really does:

    ... deploy deploy/fleet_a100.py            # new image, desk reborn
    ... run deploy/fleet_a100.py::stop --call-id <id>   # kill the old metal
    ... run deploy/fleet_a100.py::up           # new metal, same listings
    ... run deploy/fleet_a100.py::migrate --run-ids <rid>

The desk is ITS OWN warm CPU container for the first time — everything it
knows it re-reads from the fleet journal, so a redeploy reboots it into the
same fleet. The A100 stands the sweep's two fractional hosts and LISTS THEM
OVER THE WIRE (RemoteFleet.list_host) when `up` starts it serving; `submit`
goes client -> desk -> adopt, three processes, with the client's code claim
checked at the host. The three transports below are this venue's whole
contribution.

THE METAL PLANE stands by default: the container wears a MetalService (its
books hold the two standing hosts, so residual is honest: 1 - .42 - .50),
`up` phones home the metal's own registration beside the host listings, and
a placement no listing serves becomes a desk-issued CARVE on the ~8%% that
is free — booked at this container's door before the build, so carves never
double-promise. The `reaper` function sweeps on a schedule: probe, retry
(on Modal the knock itself boots a stopped-but-deployed container), and
only what stays silent is decarved + delisted with the reason journaled.
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
ARM = {"r": 16, "lr": 1e-4, "seed": 11}    # the sweep's best plain-lora arm
GROUPS_PER_WAVE = 2
GROUP_SIZE = 8
MAX_TOKENS = 512

SERVE_FRACTION = 0.42
LEARN_FRACTION = 0.50

SERVE_ADDRESS = "a100://serve"
LEARN_ADDRESS = "a100://train"
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
        from rlstack.runner.host import Host, Partition, Regime
        from rlstack.runner.learners.torch_learner import TorchLearner
        from rlstack.runner.remote import LocalTransport

        self.store = ModalVolumeStore("/store", volume=store_volume,
                                      locator=STORE)
        ensure_tasks(self.store)

        # THE METAL PLANE, by default: the container's books, the factories
        # a desk-issued carve builds with, and the address mint. dial routes
        # through the same table the venue's host verbs use, so a carved
        # host reaches its neighbors exactly like a standing one.
        self.metal_service = MetalService(
            OwnedMetal(METAL_NAME, "A100-40GB", 1, 40.0), store=self.store,
            engine_factory=lambda regime, partition: VllmEngine(
                regime.base, tp=regime.shape,
                gpu_memory_utilization=partition.memory,
                max_model_len=1536, max_bundles=8, max_rank=16,
                cas_get=self.store.cas_get, serves=("lora",)),
            learner_factory=lambda regime, partition: TorchLearner(),
            address_of=lambda name: f"a100://{name}",
            schema_for=hf_schema,
            dial=lambda address: LocalTransport(
                self.metal_service.service_for(address)),
            release=lambda host: [engine.shutdown()
                                  for engine in host.engines])

        engine = VllmEngine(BASE, tp=1, gpu_memory_utilization=SERVE_FRACTION,
                            max_model_len=1536, max_bundles=8, max_rank=16,
                            cas_get=self.store.cas_get, serves=("lora",))
        self.serve_host = Host(
            "a100-serve", engines=(engine,), learner=None, store=self.store,
            partition=Partition(METAL_NAME, (0,), SERVE_FRACTION,
                                "A100-40GB"),
            regimes=(Regime("serve-tp1", "inference", BASE, 1),))
        self.metal_service.adopt_born(self.serve_host, SERVE_ADDRESS)
        self.learn_host = Host(
            "a100-train", engines=(), learner=TorchLearner(), store=self.store,
            partition=Partition(METAL_NAME, (0,), LEARN_FRACTION,
                                "A100-40GB"),
            regimes=(Regime("train-fsdp1", "training", BASE, 1),),
            schema_for=hf_schema,
            dial=lambda address: LocalTransport(
                self.metal_service.service_for(address)))
        self.metal_service.adopt_born(self.learn_host, LEARN_ADDRESS)
        print(f"[metal] up: {BASE} on one A100-40GB, two hosts; "
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
    def build_row_here(self, updates: int) -> dict:
        """The arm as a canonical spec row, built INSIDE the container so the
        plans' cas blobs land on the store this container reads."""
        from rlstack.spec.canonical import canonical_json

        row = json.loads(canonical_json(arm_spec(self.store, updates)))
        store_volume.commit()
        return row

    @modal.method()
    def roster(self) -> dict:
        out = {}
        for rid, tenancy in sorted(self.learn_host.roster.items()):
            entries = self.store.peek_ledger(rid)
            out[rid] = {"status": tenancy.status,
                        "committed": int(entries[-1]["update"])
                        if entries else 0}
        return out

    @modal.method()
    async def serve(self) -> None:
        """The container's standing shift: phone home, then hold the door
        open while tenancies work. Stopping THIS call is how the refresh
        test kills the metal mid-run."""
        import asyncio

        from rlstack.runner.remote import RemoteFleet

        fleet = RemoteFleet(DeskTransport())
        try:
            await fleet.register_metal(METAL_NAME, "A100-40GB", 1, 40.0,
                                       METAL_ADDRESS)
            print(f"[metal] registered {METAL_NAME} on the metal plane")
        except Exception as taken:
            print(f"[metal] {METAL_NAME} not re-registered: {taken}")
        for name, host, address in (
                ("a100-serve", self.serve_host, SERVE_ADDRESS),
                ("a100-train", self.learn_host, LEARN_ADDRESS)):
            try:
                await fleet.list_host(name, host.regimes, address,
                                      partition=host.partition.row(),
                                      metal=METAL_NAME)
                print(f"[metal] listed {name} at {address}")
            except Exception as taken:
                print(f"[metal] {name} not re-listed: {taken}")
        stats = [asyncio.create_task(h.run_stats())
                 for h in (self.serve_host, self.learn_host)]
        try:
            while True:
                await asyncio.sleep(300)
                store_volume.commit()
        finally:
            for task in stats:
                task.cancel()

    @modal.exit()
    def bring_down(self) -> None:
        self.serve_host.engine_for(BASE, 1).shutdown()


# ---------------------------------------------------------------------------
# the arm (the sweep's spec_for, lora branch only) and the task set
# ---------------------------------------------------------------------------

def arm_spec(store, updates: int):
    from rlstack import (
        AlgoSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup, GpuSet,
        GroupPlan, LearnerMember, OptimSpec, Plans, PolicySpec, PoolMember,
        RunPlan, Sample, SamplingSpec, Schedule, Seeds, WavePlan, WaveRef,
        encode, lora,
    )

    wave = WavePlan(tuple(
        GroupPlan(f"{TRAIN_TASK_ID}#{g}",
                  tuple(Sample(TRAIN_TASK_ID, "dapo_math")
                        for _ in range(GROUP_SIZE)))
        for g in range(GROUPS_PER_WAVE)))
    plans = Plans(
        train=store.cas_put(encode(RunPlan(tuple(
            WaveRef(f"self://rollouts/{u}") for u in range(1, updates + 1))))),
        rollout=store.cas_put(encode(RunPlan((wave,) * updates))))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank={"pi": lora(SITE, r=ARM["r"])}),
        gen=GenSpec(envs=("dapo_math",), tasks=(TRAIN_TASKS,),
                    sampling=SamplingSpec(temperature=1.0,
                                          max_tokens=MAX_TOKENS)),
        plans=plans,
        algo=AlgoSpec(loss="grpo", post=("final_answer", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=ARM["lr"],
                                      weight_decay=0.0),
                      schedule=Schedule(microbatch_tokens=512,
                                        max_policy_lag=1)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus=GpuSet(n=1), members=(
                PoolMember("main", tp=1, fraction=SERVE_FRACTION),)),
            GpuGroup(gpus=GpuSet(n=1), members=(
                LearnerMember(fsdp=1, fraction=LEARN_FRACTION),)))),
        seeds=Seeds(master=ARM["seed"]))


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
    """Start the metal's shift and wait until the desk lists both hosts."""
    from rlstack.runner.remote import RemoteFleet

    call = metal_handle().serve.spawn()
    print(f"[up] metal serving: call {call.object_id}")
    print(f"[up] kill it later with: modal run deploy/fleet_a100.py::stop "
          f"--call-id {call.object_id}")
    fleet = RemoteFleet(DeskTransport())
    for _ in range(90):
        if {"a100-serve", "a100-train"} <= set(
                fleet.status().get("listings", {})):
            break
        time.sleep(10)
    print(json.dumps({"status": fleet.status(),
                      "liveness": fleet.liveness()}, indent=2))


@app.local_entrypoint()
async def submit(updates: int = 400) -> None:
    """The arm, from THIS process through the desk: client -> desk -> adopt,
    with the client's code claim checked at the host."""
    from rlstack.runner.remote import RemoteFleet, spec_from_json

    row = metal_handle().build_row_here.remote(updates)
    reply = await RemoteFleet(DeskTransport()).submit(spec_from_json(row))
    print(json.dumps(reply, indent=2))


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
