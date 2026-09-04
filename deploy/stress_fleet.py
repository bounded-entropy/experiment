"""The fleet under the conditions the specs never tried, on the smallest metal
that can show each: TP and FSDP with every adapter type, a tenant joining a
learner mid-run, and bases that are not Qwen.

    modal run deploy/stress_fleet.py::bases           # five bases on single L4s, in parallel, no desk
    modal deploy deploy/stress_fleet.py               # the desk and one L4:2 metal (tp=2, fsdp=2)
    PYTHONUNBUFFERED=1 modal run deploy/stress_fleet.py::topology   # three tenants (lora / steer /
                                                      #   soft_prompt) through the desk, released
    PYTHONUNBUFFERED=1 modal run deploy/stress_fleet.py::latejoin   # A runs; B joins A's learner mid-run
    PYTHONUNBUFFERED=1 modal run deploy/stress_fleet.py::remote_learner  # UNRUN: anchored on main,
                                                      #   the learner on the other listing (ADR 0006 A)
    modal run deploy/stress_fleet.py::status / ::sweep / ::down --call-id <id>

    RLSTACK_STRESS_GPU=L4 RLSTACK_STRESS_TP=1 RLSTACK_STRESS_FSDP=1 modal deploy ...   # the 1x1 shape
    RLSTACK_HF_SECRET=huggingface modal run deploy/stress_fleet.py::bases                # gated bases
        (after: modal secret create huggingface HF_TOKEN=hf_...)

Every door that acquires metal ends by releasing every metal through the desk
and asserting the plane is empty (ADR 0004, promise 5). Everything
semantics-bearing is in the specs; everything else here is venue (I5).
"""

from __future__ import annotations

import json
import os
import time

import modal

APP = "rlstack-stress-fleet"
app = modal.App(APP)

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlstack-hf-cache", create_if_missing=True)

# The shape, as three flags read at deploy time on the client and inside the
# container: the card(s) the metal is deployed on, the engine's width, the
# learner's width.
GPU = os.environ.get("RLSTACK_STRESS_GPU", "L4:2")
TP = int(os.environ.get("RLSTACK_STRESS_TP", "2"))
FSDP = int(os.environ.get("RLSTACK_STRESS_FSDP", "2"))
HF_SECRET = os.environ.get("RLSTACK_HF_SECRET", "")
SECRETS = [modal.Secret.from_name(HF_SECRET)] if HF_SECRET else []

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
          "RLSTACK_STRESS_TP": str(TP), "RLSTACK_STRESS_FSDP": str(FSDP)})
    .add_local_python_source("rlstack", "rlstack_engine")
    .add_local_dir("rlstack/observe/web", remote_path="/root/rlstack/observe/web")
)

BASE = "Qwen/Qwen3-0.6B"        # the fleet's base
HIDDEN = 1024
STORE = "modal://rlstack-store"
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
SCHEME = "stress"
IDLE_S = 1800.0
FLEET_LOG = "fleet/stress.jsonl"

# The base matrix (single L4 each, no desk). Gated bases need RLSTACK_HF_SECRET.
BASES = (
    ("Qwen/Qwen3-0.6B", "the baseline every other check ran on"),
    ("allenai/OLMo-2-0425-1B", "OLMo 2, 1B — llama-shaped names, q/k norms, post-norms"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "a Llama architecture that is not Qwen"),
    ("EleutherAI/pythia-1b", "GPT-NeoX: gpt_neox.layers.N.attention — NOT llama-shaped"),
    ("google/gemma-3-1b-it", "Gemma 3, 1B — gated on the Hub"),
)


# ---------------------------------------------------------------------------
# the venue's transports (I5)
# ---------------------------------------------------------------------------

def blocking_ask(fn):
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


def same_metal_transport(address: str):
    """A host reaching a pool on its OWN metal goes in-process — LocalTransport
    over the sibling host's own service (json both ways, admission at that
    host's arbiter; the plora venue's shape, proven by the steer venue). Never
    a Modal self-call: a host adopts on the container's loop and asks
    reachability through the transport's SYNC verb, so a self-call would wait
    on the loop it is blocking. A foreign scheme is another metal's."""
    from rlstack.runner.remote import LocalTransport

    if address.startswith(f"{SCHEME}://"):
        return LocalTransport(_METAL_SERVICE["service"].service_for(address))
    return MetalTransport(address)


_METAL_SERVICE: dict = {}


# ---------------------------------------------------------------------------
# the desk
# ---------------------------------------------------------------------------

def a_store():
    from rlstack import ModalVolumeStore

    class StressDeskStore(ModalVolumeStore):
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

    return StressDeskStore("/store", volume=store_volume, locator=STORE)


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
              f"/ released: {sorted(self.desk.released)}", flush=True)

    @modal.method()
    async def desk(self, verb: str, payload: dict) -> dict:
        return await self.door.serve(verb, payload)

    @modal.method()
    def desk_ask(self, verb: str, payload: dict) -> dict:
        return self.door.answer(verb, payload)


# ---------------------------------------------------------------------------
# the metal: one container, tp=TP engines and fsdp=FSDP learners, released by the desk
# ---------------------------------------------------------------------------

def bring_up_metal():
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.desk import MetalService
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    service = MetalService(
        MetalService.measure(METAL), store=a_store(),
        builds=Builds(
            engine=EngineBuild(max_model_len=2048, max_bundles=8, max_rank=RANK,
                               serves=("lora", "steer", "soft_prompt"),
                               enforce_eager=True),
            learner=LearnerBuild()),
        address_of=lambda host_name: f"{SCHEME}://{host_name}",
        schema_for=hf_schema,
        transport_for=same_metal_transport)
    _METAL_SERVICE["service"] = service
    print(f"[{METAL}] up, bare: {service.metal.gpu} x{service.metal.devices} "
          f"at {service.metal.vram_gb:g} GB; residual {service.residual()}",
          flush=True)
    return service


async def announce(service) -> None:
    from rlstack.runner.remote import RemoteDesk

    metal = service.metal
    told = await RemoteDesk(DeskTransport()).register_metal(
        metal.name, metal.gpu, metal.devices, metal.vram_gb,
        f"{SCHEME}://metal", builds=service.builds.row(), idle_s=IDLE_S)
    print(f"[{METAL}] registered on the metal plane: {json.dumps(told)}", flush=True)


async def metal_duties(service) -> None:
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
        self.stand_up()

    def stand_up(self) -> None:
        import asyncio

        self.born = time.time()
        self.metal_service = bring_up_metal()
        self.duties = asyncio.create_task(metal_duties(self.metal_service))

    def live(self):
        """Reborn first if the desk RELEASED the metal standing here (a
        released container is a zombie until the venue's scaledown)."""
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
        """THE KEEPALIVE, as the shift (ADR 0003, Q3)."""
        service = self.live()
        await service.until_released()
        try:
            from modal.experimental import stop_fetching_inputs
            stop_fetching_inputs()
        except ImportError:
            pass
        return {"released": True, "metal": METAL,
                "shift_s": round(time.time() - self.born, 1)}

    @modal.exit()
    def bring_down(self) -> None:
        self.duties.cancel()
        for teardown in self.metal_service.shutdown():
            if not teardown.graceful:
                print(teardown.line(), flush=True)


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


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def ledgers(run_ids: list[str]) -> dict:
    store_volume.reload()
    store = a_store()
    out = {}
    for run_id in run_ids:
        entries = store.peek_ledger(run_id)
        out[run_id] = {"committed": int(entries[-1]["update"]) if entries else 0,
                       "train": [dict(e.get("train", {})) for e in entries]}
    return out


@app.function(image=cpu_image, timeout=300)
def host_status(address: str) -> dict:
    """A listed host's own status — its roster, its residents — over the wire."""
    from rlstack.runner.remote import RemoteHost
    return RemoteHost(MetalTransport(address)).status()


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
    held = desk().status().get("metal", {})
    standing = [name for name, row in held.items() if row.get("plane")]
    print(f"[plane] standing: {standing or 'none'}", flush=True)
    return not standing


def release_everything(reason: str) -> list[str]:
    import asyncio

    held = desk().status().get("metal", {})
    released = []
    for name, row in sorted(held.items()):
        if row.get("plane"):
            told = asyncio.run(desk().release(name, reason=reason))
            print(f"[release] {name}: {json.dumps(told)}", flush=True)
            released.append(name)
    return released


def submit(name: str, row: dict, anchor: str | None = None) -> dict:
    import asyncio

    from rlstack.runner.remote import spec_from_json

    reply = asyncio.run(desk().submit(spec_from_json(row), subdir="stress",
                                      anchor=anchor))
    print(f"[submit] {name}: {json.dumps(reply, default=str)[:400]}", flush=True)
    if not reply.get("accepted"):
        raise SystemExit(f"{name} was not accepted: {reply}")
    return reply


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


def take_down(call, reason: str) -> dict:
    verdict: dict = {"released": release_everything(reason)}
    status = desk().status()
    row = status.get("metal", {}).get(METAL, {})
    verdict["desk_says_released"] = bool(row.get("released")) and not row.get("plane")
    verdict["listings_left"] = sorted(status.get("listings", {}))
    if call is not None:
        started = time.time()
        try:
            shift = call.get(timeout=600)
            verdict["keepalive_returned"] = shift
            print(f"[shift] the keepalive returned {json.dumps(shift)} "
                  f"{time.time() - started:.1f}s after the release", flush=True)
        except Exception as still:
            verdict["keepalive_returned"] = f"NOT within 600s: {still}"
    verdict["plane_empty"] = plane_is_empty()
    print(json.dumps(verdict, indent=1), flush=True)
    if not (verdict["desk_says_released"] and verdict["plane_empty"]):
        raise SystemExit("metal left standing")
    return verdict


@app.local_entrypoint()
def topology(master: int = 41) -> None:
    """Three tenants — lora, steer, soft_prompt — through the desk onto ONE
    tp=TP serving host and ONE fsdp=FSDP learner, two updates each, then the
    release. What this shows: every adapter type's rollout lowering on a
    sharded engine and its replay lowering on a sharded learner, three
    tenants deep."""
    call = metal_handle().serve.spawn()
    print(f"[topology] {METAL} serving: call {call.object_id} "
          f"(gpu {GPU}, tp {TP}, fsdp {FSDP})", flush=True)
    try:
        print(json.dumps(wait_for_metal(), indent=1), flush=True)
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
        take_down(call, "topology check done")


@app.local_entrypoint()
def latejoin(master: int = 51) -> None:
    """A runs first (lora, UPDATES_A updates); once A has committed at
    least one update, B (steer) is submitted and JOINS A's serving host and
    learner mid-run. What this shows: additive install on a learner that is
    training, requests of two tenants batching on one engine, A's rails
    undisturbed by B's arrival, both plans finishing."""
    call = metal_handle().serve.spawn()
    print(f"[latejoin] {METAL} serving: call {call.object_id}", flush=True)
    try:
        print(json.dumps(wait_for_metal(), indent=1), flush=True)
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
        learner_address = desk().status()["listings"][a["pools"]["learner"]]["address"]
        roster = host_status.remote(learner_address).get("tenants", {})
        print(f"[roster] the learner host after B's arrival: "
              f"{json.dumps(roster)}", flush=True)
        progress = await_runs(runs, {"A": UPDATES_A, "B": UPDATES})
        rails = report_rails(runs, progress)
        print(json.dumps({"joined": joined, "a_committed_before_b": a_before,
                          "roster_after_b": roster, "rails": rails}, indent=1),
              flush=True)
    finally:
        take_down(call, "latejoin check done")


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
    call = metal_handle().serve.spawn()
    print(f"[remote_learner] {METAL} serving: call {call.object_id}", flush=True)
    try:
        print(json.dumps(wait_for_metal(), indent=1), flush=True)
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
            desk().status()["listings"][pools["main"]]["address"]
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
        take_down(call, "remote-learner check done")


@app.local_entrypoint()
def status() -> None:
    told = desk().status()
    print(json.dumps({"listings": told["listings"], "metal": told["metal"],
                      "liveness": desk().liveness()}, indent=2))


@app.local_entrypoint()
def sweep() -> None:
    release_everything("sweep")
    if not plane_is_empty():
        raise SystemExit("metal still standing after the sweep")


@app.local_entrypoint()
def down(call_id: str = "") -> None:
    started = time.time()
    release_everything("released by hand")
    if call_id:
        shift = modal.FunctionCall.from_id(call_id).get(timeout=600)
        print(f"[shift] the keepalive returned {json.dumps(shift)} "
              f"{time.time() - started:.1f}s after the release")
    if not plane_is_empty():
        raise SystemExit("metal still standing")
