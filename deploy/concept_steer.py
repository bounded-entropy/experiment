"""A steering vector distilled from a prompt-conditioned teacher (ADR 0005).

    modal run deploy/concept_steer.py::prompts                   # the corpus -> cas uris
    modal deploy deploy/concept_steer.py                         # the desk and the metal, standing
    modal run deploy/concept_steer.py::distill_set               # the teacher's rollouts (one run_id)
    modal run deploy/concept_steer.py::train --layer 10 --teacher-run <rid>
    modal run deploy/concept_steer.py::measure --run-id <rid>    # reverse KL, outside the run
    modal run deploy/concept_steer.py::export --run-id <rid> --version 64
    modal run deploy/concept_steer.py::status                    # the desk's inventory
    modal run deploy/concept_steer.py::sweep                     # release every metal it holds

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

ONE THING THE ALTERNATION DOES NOT BUY, stated before it is run: a SHARDED
learner reports that it cannot sleep — `FsdpTorchLearner.sleeps` is
`ranks.width == 1`, because offloading a DTensor chorus is its own proof (ADR
0002, non-promises) — so at fsdp=2 the arbiter wires no sleep hook for it and
only the ENGINE hands its share back. The partition's fraction is the unit's
LARGEST member either way, so a live learner and an awake engine would both
be entitled to it. Whether the two fit on one 80 GB card at these numbers is
the shakeout's first finding; the fallback is two HostSpecs on wider metal,
or an engine share small enough to sit beside a resident learner.

Everything semantics-bearing is in the specs below — the banks, the loss, the
plans, the corpus (I5); everything else here is venue.
"""

from __future__ import annotations

import json
import os
import time

import modal

APP = "rlstack-concept-steer"
app = modal.App(APP)

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlstack-hf-cache", create_if_missing=True)

# The card, as one flag. A 32B at tp=2/fsdp=2 needs ~80 GB per device pair;
# the list is the scheduler's choice of which of those frees first, and GB is
# what keeps that honest — 34 GB is 34 GB on either card (ADR 0001).
GPUS = [g for g in os.environ.get("RLSTACK_CONCEPT_GPU", "A100-80GB:2,H100:2"
                                  ).split(",") if g]
WIDTH = 2                       # tp for the pool, fsdp for the learner

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("safetensors", "numpy")
    .add_local_python_source("rlstack", "rlstack_engine")
)

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
    .add_local_dir("rlstack/observe/web", remote_path="/root/rlstack/observe/web")
    .add_local_dir("tests", remote_path="/root/tests")
)

# the corpus builder's own layer, kept AFTER the pinned one so the pins stay
# cached with the campaign images (deploy/tasks_dapo.py's rule, #60)
tasks_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==5.16.1", "huggingface_hub", "safetensors",
                 "numpy")
    .pip_install("pyarrow")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("rlstack")
)

BASE = "Qwen/Qwen3-32B"
HIDDEN = 5120                   # Qwen3-32B's width — a boundary has no shape
STORE = "modal://rlstack-store"
SUBDIR = "concept"

# The paper's own anchors for Qwen3-32B (10 / 32 / 54 of 64, Appendix B), and
# `resid_pre.<n>` is the stream LEAVING layer n — the output of
# model.layers[n], which is where the harness adds (Q8, confirmed).
ANCHORS = (10, 32, 54)
ENTRY = "v"                     # the bank's one name; `export` copies v@<version>

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
LEARNER_GB = 48.0 * WIDTH       # 48 GB/device: ~35 shard + recompute + Adam

METAL = "concept-a100"
SCHEME = "concept"
IDLE_S = 1800.0                 # the desk's clock; the venue's scaledown is no shorter
FLEET_LOG = "fleet/concept.jsonl"   # this app's OWN fleet journal


# ---------------------------------------------------------------------------
# the venue's transports (I5): desk-by-name, host-by-address, scheme-routed
# ---------------------------------------------------------------------------

def blocking_ask(fn):
    """One blocking Modal call on its OWN thread (a blocking portal call from
    a loop thread wedges the loop)."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as one:
        return one.submit(fn).result()


class DeskTransport:
    def __init__(self) -> None:
        self._handle = None

    def handle(self):
        if self._handle is None:
            self._handle = modal.Cls.from_name(APP, "Desk")()
        return self._handle

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.handle().desk.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return blocking_ask(lambda: self.handle().desk_ask.remote(verb, payload))


def metal_handle():
    return modal.Cls.from_name(APP, "MetalS")()


class MetalTransport:
    """Host-addressed frames, routed to the one metal container."""

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
        return blocking_ask(lambda: self.handle().host_ask.remote(
            self.address, verb, payload))


class MetalPlaneTransport:
    """Metal-addressed frames (carve / decarve / release / residual)."""

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
        return blocking_ask(lambda: self.handle().metal_ask.remote(verb, payload))


# ---------------------------------------------------------------------------
# the desk: its own container, its OWN fleet journal on the shared volume
# ---------------------------------------------------------------------------

def a_store():
    from rlstack import ModalVolumeStore

    class ConceptDeskStore(ModalVolumeStore):
        """Several desks share this volume; this one journals to its OWN
        fleet file so none replays another's listings."""

        def append_fleet_event(self, entry) -> None:
            self._append_line(FLEET_LOG, json.dumps(
                entry, sort_keys=True, separators=(",", ":")))

        def read_fleet_log(self):
            try:
                raw = self._read(FLEET_LOG).decode()
            except FileNotFoundError:
                return []
            out = []
            for line in raw.strip().splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
            return out

    return ConceptDeskStore("/store", volume=store_volume, locator=STORE)


@app.cls(image=cpu_image, volumes={"/store": store_volume},
         timeout=3600, min_containers=1, max_containers=1,
         scaledown_window=1200)
@modal.concurrent(max_inputs=32)
class Desk:
    @modal.enter()
    def bring_up(self) -> None:
        from rlstack.runner.campaign import Campaigns
        from rlstack.runner.desk import Desk
        from rlstack.runner.remote import RemoteHost, RemoteMetal

        self.desk = Desk.from_journal(
            a_store(),
            host_for=lambda address: RemoteHost(MetalTransport(address)),
            metal_for=lambda address: RemoteMetal(MetalPlaneTransport(address)),
            idle_s=IDLE_S)
        self.door = Campaigns(self.desk)
        print(f"[desk] rebuilt from journal: {sorted(self.desk.listings)} "
              f"/ metal plane: {sorted(self.desk.metal_remotes)} "
              f"/ released: {sorted(self.desk.released)}")

    @modal.method()
    async def desk(self, verb: str, payload: dict) -> dict:
        return await self.door.serve(verb, payload)

    @modal.method()
    def desk_ask(self, verb: str, payload: dict) -> dict:
        return self.door.answer(verb, payload)


# ---------------------------------------------------------------------------
# the metal: one container, every host a desk-issued carve, released by the desk
# ---------------------------------------------------------------------------

def bring_up_metal():
    """The metal container's whole state: its books, its recipe, its router.

    The recipe serves `steer` — its demands (our worker class, eager mode,
    the V1 runner) are paid by the build and refused at construction if they
    cannot be — and a plain learner. The card is MEASURED off the device
    (ADR 0001, Q6), never typed: GPUS lets either card answer and the books
    must speak the truth about the one that did.
    """
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.desk import MetalService
    from rlstack.runner.remote import LocalTransport
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    born: dict = {}

    def transport_for(address: str):
        """A host reaching a pool on THIS metal goes in-process: the wire is
        LocalTransport over the sibling host's own service. Never a Modal
        call to this container from inside this container — the adoption
        runs on the container's loop and would wait on an input it must
        itself dispatch (#77, found on the venue)."""
        if address.startswith(f"{SCHEME}://"):
            return LocalTransport(born["service"].service_for(address))
        return MetalTransport(address)

    service = MetalService(
        MetalService.measure(METAL), store=a_store(),
        builds=Builds(
            engine=EngineBuild(max_model_len=4096, max_bundles=8, max_rank=16,
                               serves=("steer",), enforce_eager=True,
                               enable_sleep_mode=True),
            learner=LearnerBuild(checkpoint_activations=True)),
        address_of=lambda host_name: f"{SCHEME}://{host_name}",
        schema_for=hf_schema,
        transport_for=transport_for)
    born["service"] = service
    print(f"[{METAL}] up, bare: {service.metal.gpu} x{service.metal.devices} "
          f"at {service.metal.vram_gb:g} GB; residual {service.residual()}")
    return service


async def announce(service) -> None:
    """The metal REGISTERS ITSELF the moment it exists (ADR 0001, Q5a), with
    its idle limit declared — the desk's clock for this metal."""
    from rlstack.runner.remote import RemoteDesk

    metal = service.metal
    told = await RemoteDesk(DeskTransport()).register_metal(
        metal.name, metal.gpu, metal.devices, metal.vram_gb,
        f"{SCHEME}://metal", builds=service.builds.row(), idle_s=IDLE_S)
    print(f"[{METAL}] registered on the metal plane: {json.dumps(told)}")


async def metal_duties(service) -> None:
    """Announce, then follow every carved host's stats and commit the volume
    — until the desk releases this metal, at which point the duties end with
    the shift."""
    import asyncio

    try:
        await announce(service)
    except Exception as refused:
        print(f"[{METAL}] REGISTRATION REFUSED: {refused}", flush=True)
    stats: dict[str, asyncio.Task] = {}
    tick = 0
    try:
        while not service.released.is_set():
            for host_service in list(service.services.values()):
                host = host_service.host
                if host.name not in stats:
                    stats[host.name] = asyncio.create_task(host.run_stats())
            await asyncio.sleep(30)
            tick += 1
            if tick % 2 == 0:
                await store_volume.commit.aio()
    finally:
        for task in stats.values():
            task.cancel()
        await store_volume.commit.aio()


@app.cls(image=gpu_image, gpu=GPUS,
         volumes={"/store": store_volume, "/hf": hf_cache},
         timeout=86400, scaledown_window=int(IDLE_S), max_containers=1)
@modal.concurrent(max_inputs=64)
class MetalS:
    @modal.enter()
    async def bring_up(self) -> None:
        self.stand_up()

    def stand_up(self) -> None:
        """A fresh metal on this container: books, recipe, router, duties."""
        import asyncio

        self.born = time.time()
        self.metal_service = bring_up_metal()
        self.duties = asyncio.create_task(metal_duties(self.metal_service))

    def live(self):
        """The metal every door answers through — reborn first if the desk
        RELEASED the one standing here. Standing a new metal up on the same
        container is exactly what a knock asks for (ADR 0003, Q4)."""
        if self.metal_service.released.is_set():
            print(f"[{METAL}] reborn on a released container", flush=True)
            self.stand_up()
        return self.metal_service

    @modal.method()
    async def host(self, address: str, verb: str, payload: dict) -> dict:
        return await self.live().service_for(address).serve(verb, payload)

    @modal.method()
    def host_ask(self, address: str, verb: str, payload: dict) -> dict:
        return self.live().service_for(address).answer(verb, payload)

    @modal.method()
    async def metal(self, verb: str, payload: dict) -> dict:
        return await self.live().serve(verb, payload)

    @modal.method()
    def metal_ask(self, verb: str, payload: dict) -> dict:
        return self.live().answer(verb, payload)

    @modal.method()
    async def serve(self) -> dict:
        """THE KEEPALIVE, as the shift (ADR 0003, Q3): the input in flight
        keeps this container from scaling down while its hosts carry work,
        and it RETURNS when the desk releases this metal."""
        service = self.live()
        await service.until_released()
        try:
            from modal.experimental import stop_fetching_inputs
            stop_fetching_inputs()
        except ImportError:
            pass                       # the idle scaledown is the backstop
        return {"released": True, "metal": METAL,
                "shift_s": round(time.time() - self.born, 1)}

    @modal.exit()
    def bring_down(self) -> None:
        self.duties.cancel()
        for teardown in self.metal_service.shutdown():
            if not teardown.graceful:
                print(teardown.line(), flush=True)


# ---------------------------------------------------------------------------
# the science: two specs and one measurement, written out (I5)
# ---------------------------------------------------------------------------

def topology():
    """ONE HOSTSPEC, TWO MEMBERS: `main` at tp=2 ALTERNATING with the learner
    at fsdp=2 on one partition (Q7). A 32B is ~32.5 GiB per device either
    way, so the two cannot co-reside on 80 GB with a 1000-token activation
    budget; alternating implies max_policy_lag=0, which an SFT run does not
    care about because nothing it trains on is its own."""
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


def student_spec(store, teacher_run: str, layer: int):
    """ONE ARM: the same base with ONE steer at one anchor boundary, SFT over
    the teacher's rows, sampling nothing of its own. The three arms share
    plan bytes and differ in the bank alone — `main` is declared and idle,
    because the gate holds the steer's boundary against its inventory."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, OptimSpec, Plans, PolicySpec, Schedule,
        Seeds, encode, steer,
    )

    plan = store.cas_put(encode(sft_train_plan(teacher_run)))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE,
                          bank={ENTRY: steer(f"resid_pre.{layer}", d=HIDDEN)}),
        gen=None,
        plans=Plans(train=plan, rollout=None),
        algo=AlgoSpec(loss="sft", post=(),
                      optim=OptimSpec("adamw", lr=LR),
                      schedule=Schedule(microbatch_tokens=MICROBATCH_TOKENS,
                                        max_policy_lag=0)),
        topology=topology(),
        seeds=Seeds(master=11))


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
              layer: int = 0) -> dict:
    """One spec as its canonical row, for the client to submit. The plan goes
    in the cas here, which is why this runs on the volume."""
    from rlstack import canonical_json

    store = a_store()
    spec = (teacher_spec(store, train_tasks) if kind == "teacher"
            else student_spec(store, teacher_run, layer))
    store_volume.commit()
    return json.loads(canonical_json(spec))


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def progress(run_ids: list[str]) -> dict:
    """Each run's extent progress, off the store — one predicate for both
    kinds of run (ADR 0006 Part B)."""
    from rlstack import run_progress

    store_volume.reload()
    store = a_store()
    out = {}
    for run_id in run_ids:
        told = run_progress(store, run_id)
        entries = store.peek_ledger(run_id)
        out[run_id] = {
            "extent": told.extent, "completed": told.completed,
            "planned": told.planned, "done": told.done,
            "train": [dict(e.get("train", {})) for e in entries[-3:]]}
    return out


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
    from rlstack.runner.remote import RemotePool

    store_volume.reload()
    store = a_store()
    tasks = {t.id: t for t in load_tasks(store, heldout_tasks)}
    pool = RemotePool(MetalTransport(address), base=BASE, tp=WIDTH)
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
    store = a_store()
    blob = f"{store.run_prefix(run_id)}/adapters/{ENTRY}@{version}.bin"
    key = f"exports/{run_id}/{ENTRY}@{version}.safetensors"
    data = store._read(blob)
    store._write(key, data)
    store_volume.commit()
    return {"run_id": run_id, "version": version, "bytes": len(data),
            "from": blob, "path": f"/store/{key}", "uri": f"{STORE}/{key}"}


# ---------------------------------------------------------------------------
# the doors
# ---------------------------------------------------------------------------

def desk():
    from rlstack.runner.remote import RemoteDesk
    return RemoteDesk(DeskTransport())


def wait_for_metal(timeout_s: float = 900.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        held = desk().status().get("metal", {})
        if METAL in held and held[METAL].get("plane"):
            return held[METAL]
        time.sleep(10)
    raise SystemExit(f"{METAL} did not register within {timeout_s:.0f}s")


def plane_is_empty() -> bool:
    """ADR 0003 / #77: nothing on the plane — every metal released."""
    held = desk().status().get("metal", {})
    standing = [name for name, row in held.items() if row.get("plane")]
    print(f"[plane] standing: {standing or 'none'}")
    return not standing


def release_everything(reason: str) -> list[str]:
    """Every metal the desk can still command, released by the desk."""
    import asyncio

    held = desk().status().get("metal", {})
    released = []
    for name, row in sorted(held.items()):
        if row.get("plane"):
            told = asyncio.run(desk().release(name, reason=reason))
            print(f"[release] {name}: {json.dumps(told)}")
            released.append(name)
    return released


def take_down(call, reason: str) -> dict:
    """Every door that acquires metal ends HERE: the desk hands the metal
    back, the keepalive returns because the desk said so, and the plane is
    ASSERTED empty (ADR 0003 / #77)."""
    verdict: dict = {"released": release_everything(reason)}
    status = desk().status()
    row = status.get("metal", {}).get(METAL, {})
    verdict["desk_says_released"] = bool(row.get("released")) and not row.get("plane")
    if call is not None:
        started = time.time()
        try:
            verdict["keepalive_returned"] = call.get(timeout=600)
            print(f"[shift] the keepalive returned "
                  f"{json.dumps(verdict['keepalive_returned'])} "
                  f"{time.time() - started:.1f}s after the release", flush=True)
        except Exception as still:
            verdict["keepalive_returned"] = f"NOT within 600s: {still}"
    verdict["plane_empty"] = plane_is_empty()
    print(json.dumps(verdict, indent=1), flush=True)
    if not verdict["plane_empty"]:
        raise SystemExit("metal left standing after the door")
    return verdict


def submit_and_wait(row: dict, timeout_s: float) -> str:
    """One spec through the desk, then its extent followed to completion."""
    import asyncio

    from rlstack.runner.remote import spec_from_json

    reply = asyncio.run(desk().submit(spec_from_json(row), subdir=SUBDIR))
    print(f"[submit] {json.dumps(reply, default=str)[:400]}", flush=True)
    if not reply.get("accepted"):
        raise SystemExit(f"not accepted: {reply}")
    run_id = reply["run_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        told = progress.remote([run_id])[run_id]
        print(f"[{run_id[:12]}] {told['completed']}/{told['planned']} "
              f"{told['extent']} — {json.dumps(told['train'])}", flush=True)
        if told["done"]:
            return run_id
        time.sleep(60)
    raise SystemExit(f"{run_id} did not finish within {timeout_s:.0f}s")


@app.local_entrypoint()
def prompts(concept: str = CONCEPT, seed: int = SPLIT_SEED) -> None:
    """THE CORPUS. No metal: the builder needs a tokenizer and a download,
    not a GPU. Prints the uris the specs below pin."""
    print(json.dumps(build_prompts.remote(concept, seed), indent=1))


@app.local_entrypoint()
def up() -> None:
    """Boot the metal (spawning the keepalive is the knock) and wait until it
    has registered itself with the desk."""
    call = metal_handle().serve.spawn()
    print(f"[up] {METAL} serving: call {call.object_id}")
    print(json.dumps(wait_for_metal(), indent=2))


@app.local_entrypoint()
def distill_set(train_tasks: str = "", timeout_s: float = 14400.0) -> None:
    """THE TEACHER'S TRAJECTORY SET, as a generation-only run. Prints the
    run_id the three arms name in their plans."""
    if not train_tasks:
        raise SystemExit("--train-tasks <cas uri from ::prompts>")
    call = metal_handle().serve.spawn()
    print(f"[distill_set] {METAL} serving: call {call.object_id}", flush=True)
    try:
        print(json.dumps(wait_for_metal(), indent=1), flush=True)
        run_id = submit_and_wait(canonical.remote("teacher", train_tasks),
                                 timeout_s)
        print(f"[set] the teacher's rollouts are run {run_id} — "
              f"pass it to ::train --teacher-run", flush=True)
    finally:
        take_down(call, "distill_set done")


@app.local_entrypoint()
def train(layer: int = 0, teacher_run: str = "",
          timeout_s: float = 14400.0) -> None:
    """ONE ARM: the steer at `resid_pre.<layer>`, SFT over the teacher's set."""
    if layer not in ANCHORS:
        raise SystemExit(f"--layer must be one of {list(ANCHORS)}")
    if not teacher_run:
        raise SystemExit("--teacher-run <run_id from ::distill_set>")
    call = metal_handle().serve.spawn()
    print(f"[train] {METAL} serving: call {call.object_id}", flush=True)
    try:
        print(json.dumps(wait_for_metal(), indent=1), flush=True)
        run_id = submit_and_wait(
            canonical.remote("student", "", teacher_run, layer), timeout_s)
        print(f"[arm] resid_pre.{layer} trained as run {run_id}", flush=True)
    finally:
        take_down(call, f"train resid_pre.{layer} done")


@app.local_entrypoint()
def measure(run_id: str = "", heldout_tasks: str = "") -> None:
    """THE DISTILLATION NUMBER, outside the run: one idempotent pass that
    backfills every 8th committed version it has not measured."""
    import asyncio

    from rlstack.runner.desk import Demand

    if not run_id or not heldout_tasks:
        raise SystemExit("--run-id <rid> --heldout-tasks <cas uri>")
    call = metal_handle().serve.spawn()
    print(f"[measure] {METAL} serving: call {call.object_id}", flush=True)
    try:
        print(json.dumps(wait_for_metal(), indent=1), flush=True)
        # a PURE CLIENT's door: demands in, addresses out — the measurer
        # joins the serving pool and is thereafter just admitted traffic
        placed = asyncio.run(desk().resolve((
            Demand(pool="main", capability="inference", base=BASE,
                   shape=WIDTH, vram_gb=MAIN_GB, group=0),)))
        print(f"[measure] placed: {json.dumps(placed, default=str)[:400]}",
              flush=True)
        if not placed.get("placed"):
            raise SystemExit(f"no serving host: {placed}")
        told = measure_once.remote(run_id, heldout_tasks,
                                   placed["pools"]["main"])
        print(json.dumps(told, indent=1), flush=True)
    finally:
        take_down(call, "measure done")


@app.local_entrypoint()
def export(run_id: str = "", version: int = 0) -> None:
    """The vector out to the volume, for the paper's ICL harness (Q10). No
    metal: blobs are store reads."""
    if not run_id:
        raise SystemExit("--run-id <rid> --version <n>")
    print(json.dumps(export_vector.remote(run_id, version), indent=1))


@app.local_entrypoint()
def status() -> None:
    told = desk().status()
    print(json.dumps({"listings": told["listings"], "metal": told["metal"],
                      "liveness": desk().liveness()}, indent=2))


@app.local_entrypoint()
def sweep() -> None:
    """Release whatever a dead door left standing, and assert the plane is
    empty."""
    release_everything("sweep")
    if not plane_is_empty():
        raise SystemExit("metal still standing after the sweep")


@app.function(image=gpu_image, timeout=1800)
def run_tests() -> str:
    """The fakes suite inside the image, where the torch-gated cases run."""
    import subprocess

    out = subprocess.run(["python", "-m", "unittest", "discover", "-s", "tests"],
                         cwd="/root", capture_output=True, text=True)
    tail = out.stderr[-4000:]
    print(tail)
    if out.returncode != 0:
        raise SystemExit("the suite is red in the image")
    return tail
