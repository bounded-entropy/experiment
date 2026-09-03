"""Steering vectors on ONE L4, through the desk: ADR 0004's check.

    modal run deploy/steer_l4.py::run_tests        # the fakes suite in the image (torch-gated cases run)
    modal run deploy/steer_l4.py::probe            # promises 1-2: parity, one container, no desk
    modal deploy deploy/steer_l4.py                # the desk and the metal, standing
    modal run deploy/steer_l4.py::check            # promises 3-5: up, two tenants, release, empty plane
    modal run deploy/steer_l4.py::status           # the desk's inventory
    modal run deploy/steer_l4.py::sweep            # release every metal the desk still holds
    modal run deploy/steer_l4.py::knock            # the door back: a placement re-acquires released metal

    RLSTACK_STEER_GPU=L4:2 RLSTACK_STEER_TP=2 modal deploy deploy/steer_l4.py   # the TP condition

THREE THINGS THIS FILE EXISTS TO SHOW. First the residual lever itself: a
steer served by the engine image's hook and replayed by the site wrapper
agree — zero is the base bit for bit on both sides, a vector's window lands
where it was recorded, the prefix cache never aliases across bundles or
windows. Second I8 with a third mechanism: two tenants — lora-only and
steer-only — submitted THROUGH THE DESK, joined onto one serving host and
one learner on one card, trained side by side. Third ADR 0003 on the venue,
for the first time: the metal is handed back by the desk's `release`, the
keepalive input returns because the desk said so, and every entrypoint that
acquires metal ends by ASSERTING the plane is empty.

Everything semantics-bearing is in the two specs — the banks, the loss, the
plans — and everything else here is venue (I5).
"""

from __future__ import annotations

import json
import os
import time

import modal

APP = "rlstack-steer-l4"
app = modal.App(APP)

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlstack-hf-cache", create_if_missing=True)

# The condition, as one flag: the card the metal is deployed on and the width
# the engine is built at. Read at deploy time on the client and inside the
# container (the image carries it), so a redeploy is the whole change.
GPU = os.environ.get("RLSTACK_STEER_GPU", "L4")
TP = int(os.environ.get("RLSTACK_STEER_TP", "1"))

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
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "RLSTACK_STEER_TP": str(TP)})
    .add_local_python_source("rlstack", "rlstack_engine")
    .add_local_dir("tests", remote_path="/root/tests")
)

BASE = "Qwen/Qwen3-0.6B"
HIDDEN = 1024                   # Qwen3-0.6B's width — a boundary has no shape, the spec states it
STORE = "modal://rlstack-store"
# the task sets built by deploy/tasks_dapo.py (#60), reused verbatim
TRAIN_TASKS = "cas://09499d32b51e5e1b2a644b1c65e01b44aa42ff1a5bfac78ead41f98f89f09c93"
EVAL_TASKS = "cas://82ae4626dbb59a2c50e2b13cbe7250c5f1ddd02dfb81edc7495efb77759d420b"

STEER_SITE = "resid_pre.8-20"   # thirteen boundaries, one vector each
LORA_SITE = "layers.0-27.self_attn.*"
RANK = 16
STEER_LR = 1e-3                 # a vector wants its own LR (the soft prompt's lesson, #46)

UPDATES = 2
GROUPS_PER_WAVE = 2
GROUP_SIZE = 4
MAX_TOKENS = 256

# The partition treaty on one L4 (24 GB), in GB (ADR 0001): serving and the
# learner compose on one device because vLLM budgets against the device total.
MAIN_GB = 7.2
LEARNER_GB = 9.6

METAL = "steer-l4"
SCHEME = "steer"
IDLE_S = 1800.0                 # the desk's clock; the venue's scaledown is no shorter (ADR 0003 Q3)
FLEET_LOG = "fleet/steer.jsonl"  # this app's OWN fleet journal on the shared volume


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

    class SteerDeskStore(ModalVolumeStore):
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

    return SteerDeskStore("/store", volume=store_volume, locator=STORE)


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
    The recipe is what a carve builds from: an engine that SERVES lora and
    steer (the steer's demands — our worker class, eager mode — are paid by
    the build, and refused at construction if they cannot be), and a plain
    learner."""
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.desk import MetalService
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    service = MetalService(
        MetalService.measure(METAL), store=a_store(),
        builds=Builds(
            engine=EngineBuild(max_model_len=2048, max_bundles=8, max_rank=RANK,
                               serves=("lora", "steer"), enforce_eager=True),
            learner=LearnerBuild()),
        address_of=lambda host_name: f"{SCHEME}://{host_name}",
        schema_for=hf_schema,
        transport_for=lambda address: MetalTransport(address))
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
    — until the desk releases this metal, at which point the duties end
    with the shift."""
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


@app.cls(image=gpu_image, gpu=GPU,
         volumes={"/store": store_volume, "/hf": hf_cache},
         timeout=86400, scaledown_window=int(IDLE_S), max_containers=1)
@modal.concurrent(max_inputs=64)
class MetalS:
    @modal.enter()
    async def bring_up(self) -> None:
        import asyncio

        self.born = time.time()
        self.metal_service = bring_up_metal()
        self.duties = asyncio.create_task(metal_duties(self.metal_service))

    @modal.method()
    async def host(self, address: str, verb: str, payload: dict) -> dict:
        return await self.metal_service.service_for(address).serve(verb, payload)

    @modal.method()
    def host_ask(self, address: str, verb: str, payload: dict) -> dict:
        return self.metal_service.service_for(address).answer(verb, payload)

    @modal.method()
    async def metal(self, verb: str, payload: dict) -> dict:
        return await self.metal_service.serve(verb, payload)

    @modal.method()
    def metal_ask(self, verb: str, payload: dict) -> dict:
        return self.metal_service.answer(verb, payload)

    @modal.method()
    async def serve(self) -> dict:
        """THE KEEPALIVE, as the shift (ADR 0003, Q3): the input in flight is
        what keeps this container from scaling down while its hosts carry
        work, and it RETURNS when the desk releases this metal — so the
        venue reclaims the container as a consequence of the desk's decision,
        never of its own timer (which is the backstop, set no shorter)."""
        await self.metal_service.until_released()
        return {"released": True, "metal": METAL,
                "shift_s": round(time.time() - self.born, 1)}

    @modal.exit()
    def bring_down(self) -> None:
        self.duties.cancel()
        for teardown in self.metal_service.shutdown():
            if not teardown.graceful:
                print(teardown.line(), flush=True)


# ---------------------------------------------------------------------------
# the science: two banks, one topology, the same plans
# ---------------------------------------------------------------------------

def rollout_plan(task_ids, updates: int):
    """Every wave: GROUPS_PER_WAVE tasks, GROUP_SIZE completions each."""
    from rlstack import GroupPlan, RunPlan, Sample, WavePlan

    def wave(u: int):
        chosen = [task_ids[(u * GROUPS_PER_WAVE + g) % len(task_ids)]
                  for g in range(GROUPS_PER_WAVE)]
        return WavePlan(tuple(
            GroupPlan(task_id, tuple(Sample(task_id, "dapo_math")
                                     for _ in range(GROUP_SIZE)))
            for task_id in chosen))
    return RunPlan(tuple(wave(u) for u in range(updates)))


def train_plan(updates: int):
    """Update u trains on rollout u, whole: plain on-policy GRPO."""
    from rlstack import RunPlan, WaveRef
    return RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                         for u in range(1, updates + 1)))


def spec_for(store, bank: dict, overrides: dict, updates: int, master: int):
    """One experiment as one value: the given bank on the shared topology."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, GenSpec, HostSpec, LearnerMember, OptimSpec,
        Plans, PolicySpec, PoolMember, SamplingSpec, Schedule, Seeds, Topology,
        encode,
    )
    from rlstack.data.tasks import load_tasks

    task_ids = [t.id for t in load_tasks(store, TRAIN_TASKS)][:16]
    plans = Plans(train=store.cas_put(encode(train_plan(updates))),
                  rollout=store.cas_put(encode(rollout_plan(task_ids, updates))))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank=bank),
        gen=GenSpec(envs=("dapo_math",), tasks=(TRAIN_TASKS, EVAL_TASKS),
                    sampling=SamplingSpec(temperature=1.0, max_tokens=MAX_TOKENS)),
        plans=plans,
        algo=AlgoSpec(loss="grpo", post=("final_answer", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-5, overrides=overrides),
                      schedule=Schedule(microbatch_tokens=512, max_policy_lag=1)),
        # the SAME demands for both specs: the second submission JOINS the
        # first's listings (coverage is capability equality), which is I8
        # through the desk — one serving host, one learner, two tenants
        topology=Topology(hosts=(
            HostSpec((PoolMember("main", tp=TP, vram_gb=MAIN_GB * TP),)),
            HostSpec((LearnerMember(fsdp=1, vram_gb=LEARNER_GB),)))),
        seeds=Seeds(master=master))


def the_two_banks() -> dict[str, tuple[dict, dict]]:
    """lora-only and steer-only: {name: (bank, optimizer overrides)}."""
    from rlstack import lora, steer
    return {
        "lora": ({"pi": lora(LORA_SITE, r=RANK)}, {}),
        "steer": ({"nudge": steer(STEER_SITE, d=HIDDEN, init_std=0.02)},
                  {"nudge": {"lr": STEER_LR}}),
    }


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def build_specs(master: int = 11) -> dict:
    """The two specs, plans in the CAS, as canonical rows for the client."""
    from rlstack import canonical_json

    store = a_store()
    rows = {name: json.loads(canonical_json(
                spec_for(store, bank, overrides, UPDATES, master)))
            for name, (bank, overrides) in the_two_banks().items()}
    store_volume.commit()
    return rows


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def ledgers(run_ids: list[str]) -> dict:
    """Each run's committed updates and its last train block, off the store."""
    store_volume.reload()
    store = a_store()
    out = {}
    for run_id in run_ids:
        entries = store.peek_ledger(run_id)
        out[run_id] = {"committed": int(entries[-1]["update"]) if entries else 0,
                       "train": [dict(e.get("train", {})) for e in entries]}
    return out


# ---------------------------------------------------------------------------
# promises 1-2: parity, one container, no desk
# ---------------------------------------------------------------------------

class Checks:
    """A named list of pass/fail lines; the run fails loudly at the end."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
              flush=True)

    def summary(self) -> dict:
        passed = sum(1 for _, ok, _ in self.rows if ok)
        return {"passed": passed, "total": len(self.rows),
                "failed": [name for name, ok, _ in self.rows if not ok]}


@app.function(image=gpu_image, gpu=GPU, volumes={"/hf": hf_cache}, timeout=2400)
def probe() -> dict:
    """The lowering's own exam: the engine's hook against the trainer's
    wrapper, on the same numbers, plus the cache and the window."""
    import asyncio

    import torch
    import transformers
    import vllm

    from rlstack import Message, Role, SteerWindow
    from rlstack.data.flatten import TokenBatch
    from rlstack.policy.adapters import lora_torch, steer_torch
    from rlstack.policy.adapters.replay import ReplayRows, row_plan
    from rlstack.policy.adapters.steer import STEER_RECORD
    from rlstack.policy.compile import compile_bundle
    from rlstack.policy.siteschema import hf_schema, resolve
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.learners.torch_learner import TorchLearner, _doc_spans
    from rlstack.spec.specs import SamplingSpec

    check = Checks()
    print(f"[pins] vllm={vllm.__version__} torch={torch.__version__} "
          f"transformers={transformers.__version__} tp={TP}")

    engine = VllmEngine(BASE, gpu_memory_utilization=0.45, max_model_len=512,
                        max_rank=RANK, tp=TP, serves=("lora", "steer"))
    learner = TorchLearner()
    learner._ensure_base(BASE)
    model = learner._model
    schema = hf_schema(BASE)
    boundaries = resolve(schema.sites, STEER_SITE)
    weighted = resolve(schema.sites, LORA_SITE)
    check("the base is as wide as the spec says",
          int(model.config.hidden_size) == HIDDEN, f"{model.config.hidden_size}")

    reach = engine.reachability(schema.sites)
    check("the build reaches the residual boundaries through the hook",
          all(str(reach[m.name]) == "residual" for m in boundaries)
          and str(reach["final_hidden"]) == "residual")
    check("the build still reports punica on weighted sites",
          str(reach[weighted[0].name]) == "punica")
    check("the logits are honestly another lever's", str(reach["logits"]) == "none")

    # ---- the states -----------------------------------------------------------
    zero = steer_torch.build(boundaries, {"d": HIDDEN, "init_std": 0.0})
    nudges = {std: steer_torch.build(boundaries, {"d": HIDDEN, "seed": 300 + i,
                                                  "init_std": std})
              for i, std in enumerate((0.01, 0.05, 0.2))}
    delta = lora_torch.build(weighted, {"r": RANK, "seed": 101})
    generator = torch.Generator().manual_seed(101)
    for path in delta.b:
        delta.b[path].data = torch.randn(*delta.b[path].shape,
                                         generator=generator) / (RANK * 40)
    steer_torch.install(model, zero)
    for state in nudges.values():
        steer_torch.install(model, state)
    lora_torch.install(model, delta)

    def steer_slot(state):
        return {meta.path: state for meta in boundaries}

    lora_slot = {meta.path: delta for meta in weighted}
    banks = {
        "base": ({}, {}),
        "zero": (steer_slot(zero), {"nudge": steer_torch.emit(zero)}),
        "lora": (lora_slot, {"pi": lora_torch.emit(delta)}),
        "both": (lora_slot | steer_slot(nudges[0.05]),
                 {"pi": lora_torch.emit(delta),
                  "nudge": steer_torch.emit(nudges[0.05])}),
    }
    for std, state in nudges.items():
        banks[f"steer@{std}"] = (steer_slot(state), {"nudge": steer_torch.emit(state)})
    adapter_types = {"pi": "lora", "nudge": "steer"}
    bundles = {}
    for name, (_, payloads) in banks.items():
        bundle = compile_bundle(payloads, {n: 0 for n in payloads},
                                servable=payloads, adapter_types=adapter_types)
        engine.add_bundle(bundle)
        bundles[name] = bundle
    check("a bank of lora + steer registers with BOTH lowerings",
          set(engine.attachments(bundles["both"].bundle_id)) == {"lora", "steer"})

    # ---- the two sides ---------------------------------------------------------
    def replayed(slot, prompt: str, answer: str, window=(0, None)) -> tuple:
        """The trainer's logprob per answer token from ONE routed forward,
        the row's turn recording `window` — forward_backward's own call."""
        context = list(engine.tokenize(prompt))
        ids = context + list(engine.tokenize(answer))
        batch = TokenBatch(token_ids=tuple(ids), loss_mask=(1,) * len(ids),
                           behavior_logprobs=(0.0,) * len(ids),
                           segment_ids=(0,) * len(ids), doc_starts=(0,))
        plan = ReplayRows(slots=(slot,),
                          index=torch.zeros(1, dtype=torch.long, device=learner.device),
                          facts=(({STEER_RECORD: list(window)},),))
        with torch.no_grad():
            with row_plan(model).route(plan):
                out = learner._batched_logprobs(batch, _doc_spans(batch))
        return tuple(float(x) for x in out[len(context):])

    async def scored(name: str, prompt: str, answer: str, directives=()) -> tuple:
        return await engine.score_tokens([Message(Role.USER, prompt)],
                                         engine.tokenize(answer),
                                         bundles[name].bundle_id, directives)

    def gap(a, b) -> float:
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    def worst(a, b) -> float:
        return max(abs(x - y) for x, y in zip(a, b))

    def shifted_gap(a, b) -> float:
        return sum(abs(x - y) for x, y in zip(a[1:], b[:-1])) / (len(a) - 1)

    PAIRS = [
        ("What is 17 + 26? Answer with the number only.", " 43"),
        ("Name the largest planet in the solar system.", " Jupiter is the largest."),
        ("List three primes greater than ten.", " 11, 13 and 17."),
    ]

    async def measure() -> dict:
        report: dict = {}
        # 1. the zero-tolerance control, both sides
        for prompt, answer in PAIRS:
            check(f"engine: zero steer is the base bit for bit ({prompt[:18]!r})",
                  worst(await scored("zero", prompt, answer),
                        await scored("base", prompt, answer)) == 0.0)
            check(f"trainer: zero steer is the base bit for bit ({prompt[:18]!r})",
                  worst(replayed(banks["zero"][0], prompt, answer),
                        replayed({}, prompt, answer)) == 0.0)
        # 2. parity at three magnitudes, and lora + steer together
        for name in [f"steer@{std}" for std in nudges] + ["both", "lora"]:
            gaps, peaks, shifted = [], [], []
            for prompt, answer in PAIRS:
                engine_side = await scored(name, prompt, answer)
                trainer_side = replayed(banks[name][0], prompt, answer)
                gaps.append(gap(engine_side, trainer_side))
                peaks.append(worst(engine_side, trainer_side))
                shifted.append(shifted_gap(engine_side, trainer_side))
            report[name] = {"gap": max(gaps), "peak": max(peaks),
                            "shifted": min(shifted)}
            print(f"  [{name}] gap {max(gaps):.4f} peak {max(peaks):.4f} "
                  f"shift-control {min(shifted):.3f}")
            check(f"{name}: the engine and the trainer agree",
                  max(gaps) < 0.15, f"gap {max(gaps):.4f}")
            check(f"{name}: an off-by-one would have shown",
                  min(shifted) > 5 * max(gaps))
        # 3. the window: completion-only served == completion-only replayed,
        #    and != every-position served (the window matters)
        prompt, answer = PAIRS[1]
        n = len(engine.tokenize(prompt))
        windowed = await scored("steer@0.2", prompt, answer,
                                directives=(SteerWindow(start=n),))
        everywhere = await scored("steer@0.2", prompt, answer)
        check("a completion-only window replays as recorded",
              gap(windowed, replayed(banks["steer@0.2"][0], prompt, answer,
                                     window=(n, None))) < 0.15)
        check("the window changes the answer",
              worst(windowed, everywhere) > 1e-3, f"{worst(windowed, everywhere):.4f}")
        # 4. the prefix cache: the same prompt under other bundles and windows
        #    in between, then again — a bit-identical repeat, or it aliased
        first = await scored("steer@0.2", prompt, answer)
        await scored("base", prompt, answer)
        await scored("lora", prompt, answer)
        await scored("steer@0.2", prompt, answer, directives=(SteerWindow(start=n),))
        again = await scored("steer@0.2", prompt, answer)
        check("the prefix cache never aliases across bundles or windows",
              worst(first, again) == 0.0, f"{worst(first, again)}")
        base_first = await scored("base", prompt, answer)
        check("...and the base is still the base afterwards",
              worst(base_first, await scored("base", prompt, answer)) == 0.0)
        # 5. the record, and steering on decode: a greedy sample under an
        #    open window and under one closed at the prompt's end
        async def greedy(directives):
            events = [e async for e in engine.sample_tokens(
                [Message(Role.USER, prompt)],
                SamplingSpec(temperature=0.0, max_tokens=4), (),
                bundles["steer@0.2"].bundle_id, seed=1, directives=directives)]
            return events
        open_events = await greedy(())
        closed_events = await greedy((SteerWindow(0, n),))
        check("the turn records the resolved window",
              open_events[-1].turn_extras[STEER_RECORD] == [0, None]
              and closed_events[-1].turn_extras[STEER_RECORD] == [0, n])
        first_open = open_events[0].logprob
        first_closed = closed_events[0].logprob
        check("decode is steered under an open window and not under a closed one",
              abs(first_open - first_closed) > 1e-4,
              f"{first_open:.4f} vs {first_closed:.4f}")
        return report

    report = asyncio.run(measure())
    engine.shutdown()
    summary = check.summary()
    print(json.dumps({"summary": summary, "gaps": report}, indent=1))
    if summary["failed"]:
        raise SystemExit(f"probe: {summary['failed']}")
    return {"summary": summary, "gaps": report}


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


# ---------------------------------------------------------------------------
# promises 3-5: the doors
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
    """ADR 0004 promise 5: nothing on the plane — every metal released."""
    held = desk().status().get("metal", {})
    standing = [name for name, row in held.items() if row.get("plane")]
    print(f"[plane] standing: {standing or 'none'}")
    return not standing


async def release_everything(reason: str) -> list[str]:
    """Every metal the desk can still command, released by the desk."""
    held = desk().status().get("metal", {})
    released = []
    for name, row in sorted(held.items()):
        if row.get("plane"):
            told = await desk().release(name, reason=reason)
            print(f"[release] {name}: {json.dumps(told)}")
            released.append(name)
    return released


@app.local_entrypoint()
def up() -> None:
    """Boot the metal (spawning the keepalive is the knock) and wait until
    it has registered itself with the desk."""
    call = metal_handle().serve.spawn()
    print(f"[up] {METAL} serving: call {call.object_id}")
    print(json.dumps(wait_for_metal(), indent=2))


@app.local_entrypoint()
async def check(master: int = 11) -> None:
    """THE CHECK: up, two tenants through the desk, their ledgers, then the
    desk's release with the keepalive observed to return — and, whatever
    happened, every metal released and the plane asserted empty."""
    from rlstack.runner.remote import spec_from_json

    call = metal_handle().serve.spawn()
    print(f"[check] {METAL} serving: call {call.object_id}")
    verdict: dict = {"keepalive": call.object_id}
    try:
        print(json.dumps(wait_for_metal(), indent=1))
        rows = await build_specs.remote.aio(master)
        runs: dict[str, str] = {}
        hosts: dict[str, dict] = {}
        for name in ("lora", "steer"):
            reply = await desk().submit(spec_from_json(rows[name]), subdir="steer")
            print(f"[submit] {name}: {json.dumps(reply, default=str)[:400]}")
            if not reply.get("accepted"):
                raise SystemExit(f"{name} was not accepted: {reply}")
            runs[name] = reply["run_id"]
            hosts[name] = reply["pools"]
        verdict["runs"] = runs
        verdict["joined"] = hosts["lora"] == hosts["steer"]
        print(f"[join] lora on {hosts['lora']} / steer on {hosts['steer']} "
              f"-> {'ONE serving host, ONE learner' if verdict['joined'] else 'NOT joined'}")

        deadline = time.time() + 3600
        while time.time() < deadline:
            progress = await ledgers.remote.aio(list(runs.values()))
            line = {name: progress[rid]["committed"] for name, rid in runs.items()}
            print(f"[ledger] {json.dumps(line)}")
            if all(n >= UPDATES for n in line.values()):
                break
            time.sleep(30)
        else:
            raise SystemExit("the two runs did not finish within the hour")
        for name, rid in runs.items():
            gaps = [round(t.get("logprob_gap", -1.0), 4) for t in progress[rid]["train"]]
            losses = [round(t.get("loss", 0.0), 4) for t in progress[rid]["train"]]
            print(f"[{name}] {rid}: logprob_gap per update {gaps}, loss {losses}")
            verdict[f"{name}_logprob_gap"] = gaps
    finally:
        # promise 4 and 5: the desk hands the metal back and the shift ends
        released = await release_everything("steer check done")
        verdict["released"] = released
        status = desk().status()
        row = status.get("metal", {}).get(METAL, {})
        verdict["desk_says_released"] = bool(row.get("released")) and not row.get("plane")
        verdict["listings_left"] = sorted(status.get("listings", {}))
        started = time.time()
        try:
            shift = call.get(timeout=600)
            verdict["keepalive_returned"] = shift
            print(f"[shift] the keepalive returned {shift} "
                  f"{time.time() - started:.1f}s after the release")
        except Exception as still:
            verdict["keepalive_returned"] = f"NOT within 600s: {still}"
        verdict["plane_empty"] = plane_is_empty()
        print(json.dumps(verdict, indent=1))
        if not (verdict["desk_says_released"] and verdict["plane_empty"]):
            raise SystemExit("metal left standing after the check")


@app.local_entrypoint()
async def knock() -> None:
    """The door back (ADR 0003 Q4, ADR 0004 Q10): a placement for the same
    demands KNOCKS the released metal awake, the container announces, the
    desk lists it carve-able again — then it is released once more."""
    from rlstack.runner.desk import Demand

    before = desk().status().get("metal", {}).get(METAL, {})
    print(f"[knock] before: released={before.get('released')} plane={before.get('plane')}")
    try:
        placed = await desk().resolve((Demand(pool="main", capability="inference",
                                              base=BASE, shape=TP,
                                              vram_gb=MAIN_GB * TP, group=0),))
        print(f"[knock] placed: {json.dumps(placed, default=str)[:400]}")
        after = desk().status().get("metal", {}).get(METAL, {})
        print(f"[knock] after: released={after.get('released')} plane={after.get('plane')}")
        if not after.get("plane"):
            raise SystemExit("the knock did not bring the metal back")
    finally:
        await release_everything("knock check done")
        if not plane_is_empty():
            raise SystemExit("metal left standing after the knock")


@app.local_entrypoint()
def status() -> None:
    told = desk().status()
    print(json.dumps({"listings": told["listings"], "metal": told["metal"],
                      "liveness": desk().liveness()}, indent=2))


@app.local_entrypoint()
async def sweep() -> None:
    """Release whatever a dead run left standing, and assert the plane is
    empty."""
    await release_everything("sweep")
    if not plane_is_empty():
        raise SystemExit("metal still standing after the sweep")


@app.local_entrypoint()
async def down(call_id: str = "") -> None:
    """Hand the metal back by hand and, given the keepalive's call id, watch
    it return."""
    await release_everything("released by hand")
    if call_id:
        print(json.dumps(modal.FunctionCall.from_id(call_id).get(timeout=600)))
    if not plane_is_empty():
        raise SystemExit("metal still standing")
