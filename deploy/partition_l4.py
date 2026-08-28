"""Mass partitioning ONE L4: many hosts carved from one device (#51).

    modal run deploy/partition_l4.py::preview         # the ladder on fakes (CPU)
    modal run deploy/partition_l4.py::mass_partition  # legs 1,3,4,5 on metal
    modal run deploy/partition_l4.py::alternating     # leg 2: REAL sleep hooks
    modal run deploy/partition_l4.py::overcommit      # leg 7: the boundary
    modal run deploy/partition_l4.py::observe         # leg 6: the views
    modal run deploy/partition_l4.py                  # all of it, in order

The sub-GPU half of the atomic purposed partition, on silicon: one L4 carved
by the fleet's ladder into hosts that must coexist in one container — two
static inference partitions plus a training one (leg 1); an ALTERNATING host
whose arbiter wake/evict hooks are real vLLM sleep()/wake_up() and a learner
offload (leg 2); fraction-free joins onto them, each tenant reaching over the
wire the pools it did not land on (leg 3); a live carve out of residual and
the acquire refusal on a full device (leg 4); kill/resume of one tenant
through the fleet (leg 5); the observer's views over the journals the campaign
wrote (leg 6); and the overcommit boundary (leg 7). Every leg prints the
device-wide HBM ledger — torch.cuda.mem_get_info is a driver call, so it sees
vLLM's engine-core children too — beside the fleet's own arithmetic, so a
claim about a partition can be checked against the metal it claims, and a
FAILING CHECK HERE IS A FINDING rather than a broken harness. Two findings
stand and are not bugs to fix: a training partition's fraction is
unenforceable in-process (a stated cost of sharing one process), and two hosts
of one capability are unaddressable, since find_join takes the first covering
host — which is why this file still gives its second inference partition a
different base. Results in CONTEXT #51/#52.

Deployment only (I5): wiring and measurement, nothing semantics-bearing.
Image pins: keep in sync with deploy/modal_app.py.
"""

from __future__ import annotations

import time

import modal

from probe import CHECKS, arith_tasks, check

app = modal.App("rlstack-partition")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlstack-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          # #45's deployment fact, kept even at tp=1: a thread team that never
          # forms cannot segfault in libgomp
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
          "OMP_NUM_THREADS": "1",
          "HF_HOME": "/hf"})
    .add_local_python_source("probe", "rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-0.6B"
# a SECOND base is what makes a second inference partition addressable at all:
# capability is (kind, base, shape), so two hosts serving one base at one tp
# are interchangeable and the ladder always joins the first (see FINDINGS)
ALT_BASE = "Qwen/Qwen3-0.6B-Base"
STORE = "modal://rlstack-store"
VOLUMES = {"/store": store_volume, "/hf": hf_cache}

METAL_NAME = "l4-solo"
MAX_LEN = 512               # small KV so real fractions have room to be small
EVERY = 2                   # eval cadence


def l4_solo():
    """The one piece of metal this campaign owns: ONE L4, 24 GB, registered
    by a human — the acquire rung already executed (#43)."""
    from rlstack import Metal

    return Metal(METAL_NAME, "L4", 1, vram_gb=24.0)


# ---------------------------------------------------------------------------
# the device ledger: what the metal says, beside what the fleet claims
# ---------------------------------------------------------------------------

LEDGER: list[tuple[str, float, float]] = []


def hbm(label: str) -> float:
    """One device-wide HBM reading, appended to the campaign's ledger.

    mem_get_info is a driver call about the DEVICE, not this process, so it
    sees vLLM's engine-core children too — which is the only way a claim about
    a sub-GPU partition can be checked at all."""
    import torch

    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2 ** 30
    LEDGER.append((label, used, free / 2 ** 30))
    print(f"  [hbm] {label:<44} used {used:6.2f} GiB  "
          f"free {free / 2 ** 30:6.2f} GiB")
    return used


def print_ledger(title: str) -> None:
    print(f"\n== {title}: device memory ledger ============================")
    previous = None
    for label, used, free in LEDGER:
        delta = "" if previous is None else f"  ({used - previous:+.2f})"
        print(f"  {label:<44} used {used:6.2f} GiB  free {free:6.2f} GiB{delta}")
        previous = used


# ---------------------------------------------------------------------------
# the factories — and the seam where a partition's fraction reaches the build
# ---------------------------------------------------------------------------

class Partitioned:
    """The engine/learner factories a carve builds through.

    Each is handed BOTH birth facts the carve decided — the Regime (what
    capability) and the Partition (how much of what metal) — so the fraction
    reaches gpu_memory_utilization directly (#52 fixing #51c; this class used
    to re-derive it from the plan through an absorb() step that no longer has
    anything to do)."""

    def __init__(self, *, sleepy: bool = False) -> None:
        self.engines: dict[str, object] = {}
        self.learners: dict[str, object] = {}
        self.sleepy = sleepy

    def engine(self, regime, partition):
        """One vLLM engine per inference regime, built with
        gpu_memory_utilization = ITS partition's memory — the fraction is a
        reservation of the whole device, so this is where the partition stops
        being a record and starts being metal. `sleepy` builds it able to hand
        that memory back (the #52 seam), which is what an alternating host's
        arbiter hooks call."""
        from rlstack.runner.engines.vllm_engine import VllmEngine

        print(f"  [build] engine {regime.name}: {regime.base} tp={regime.shape} "
              f"gpu_memory_utilization={partition.memory} "
              f"on {partition.gpu} {partition.gpuset}{list(partition.devices)}")
        engine = VllmEngine(regime.base,
                            gpu_memory_utilization=partition.memory,
                            max_model_len=MAX_LEN, max_bundles=8,
                            max_rank=16, tp=regime.shape,
                            enable_sleep_mode=self.sleepy)
        self.engines[regime.name] = engine
        return engine

    def learner(self, regime, partition):
        """One learner per training regime. NOTE (#51e, still true): nothing
        here can cap the learner's share of the device — torch's
        set_per_process_memory_fraction is per PROCESS, and every host in this
        campaign shares one process, so a training partition's fraction is
        declared, journaled, and unenforced."""
        from rlstack.runner.learners.torch_learner import TorchLearner

        print(f"  [build] learner {regime.name}: {regime.base} "
              f"fsdp={regime.shape} (declared {partition.memory})")
        lrn = TorchLearner()
        self.learners[regime.name] = lrn
        return lrn


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------

def make_spec(store, *, master: int, loss: str = "grpo",
              post: tuple[str, ...] = ("verifier", "grpo_advantage"),
              n_updates: int = 4, base: str = BASE,
              main_fraction: float = 0.30, learner_fraction: float = 0.20,
              judge: tuple[str, float] | None = None,
              sharing: str = "concurrent", tp: int = 1, group_size: int = 4,
              per_wave: int = 8, max_tokens: int = 12, lr: float = 1e-4):
    """One tenant. `main_fraction`/`learner_fraction`/`judge` are CARVE HINTS:
    they size a partition the first time the capability is demanded and are
    ignored by every later join (#43)."""
    from rlstack import (
        AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
        OptimSpec, PolicySpec, SamplingSpec, Schedule, Seeds, TrajectorySource,
        gpus, learner, lora, pool,
    )

    train = store.cas_put(arith_tasks(64, seed=0))
    heldout = store.cas_put(arith_tasks(16, seed=1))
    members = (pool("main", tp=tp, fraction=main_fraction),)
    if judge is not None:
        members += (pool("judge", base=judge[0], fraction=judge[1]),)
    members += (learner(fraction=learner_fraction),)
    return ExperimentSpec(
        policy=PolicySpec(base=base,
                          bank={"pi": lora("layers.*.self_attn.*", r=16)}),
        gen=GenSpec(env="math_single_turn", tasks=train,
                    sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                          max_tokens=max_tokens)),
        trajectories=TrajectorySource("live"),
        algo=AlgoSpec(loss=loss, post=post, optim=OptimSpec("adamw", lr=lr),
                      schedule=Schedule(group_size=group_size,
                                        trajectories_per_wave=per_wave,
                                        n_updates=n_updates,
                                        microbatch_tokens=2048,
                                        max_policy_lag=0)),
        eval=EvalSpec(tasks=heldout, every=EVERY, n_samples=2,
                      env="math_single_turn", post=("verifier",)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), members, sharing=sharing),)),
        seeds=Seeds(master=master),
    )


def report_run(store, run_id: str, label: str, n_updates: int,
               gap_alarm: float = 0.15) -> dict:
    """One finished run's story; the gap is the cross-contamination alarm —
    many tenants sharing one engine is exactly the situation a wrong adapter
    would be served in, and the trainer recomputes under its own weights."""
    run = store.open_run(run_id)
    entries = run.read_ledger()
    updates = [int(e["update"]) for e in entries]
    check(f"{label}: ledger complete", updates == list(range(1, n_updates + 1)),
          f"{len(updates)}/{n_updates} updates")
    gaps = [e["train"]["logprob_gap"] for e in entries] or [float("nan")]
    check(f"{label}: logprob_gap at the kernel floor", max(gaps) < gap_alarm,
          f"max {max(gaps):.4f} < {gap_alarm}")
    for e in entries:
        print(f"    u{e['update']:>3}: reward "
              f"{e['post'].get('reward', float('nan')):.3f} "
              f"loss {e['train']['loss']:+.4f} "
              f"gap {e['train']['logprob_gap']:.4f} "
              f"grad {e['train']['grad_norm']:.2f}")
    due = [u for u in range(1, n_updates + 1) if u % EVERY == 0]
    have = [u for u in due if run.has_eval(u)]
    check(f"{label}: evals present", have == due, f"{have} of {due}")
    return {"run_id": run_id, "updates": len(updates), "max_gap": max(gaps)}


async def warm(engine, label: str) -> None:
    """Force the LAZY vLLM build (VllmEngine builds inside the running loop,
    at the first sample) so the partition's memory shows up in the ledger at a
    moment the campaign chose. A payload-free bundle is what loop.py gives a
    non-policy pool: the bare base, served."""
    from rlstack import Message, Role, SamplingSpec
    from rlstack.policy.compile import Bundle

    bundle = Bundle("bundle:base:warm", {}, {})
    engine.add_bundle(bundle)
    text = ""
    async for event in engine.sample_tokens(
            (Message(Role.USER, "What is 21+34? The answer is"),),
            SamplingSpec(temperature=0.0, top_p=1.0, max_tokens=4),
            (), bundle.bundle_id, 0):
        text += getattr(event, "text_delta", "")
    print(f"  [warm] {label} says {text.strip()!r}")
    hbm(f"after {label} is built + warm")


async def cancel_after(task, seconds: float) -> None:
    """Cancel a run mid-flight; a crash that beat the cancel is REPORTED."""
    import asyncio

    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as failure:
        check("run failed BEFORE the deliberate cancel", False, repr(failure))


def summarize(title: str) -> dict:
    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    print(f"\n[{title}] {sum(ok for _, ok, _ in CHECKS)} passed, "
          f"{len(failed)} failed: {failed}")
    return {"passed": sum(ok for _, ok, _ in CHECKS), "failed": failed}


# ---------------------------------------------------------------------------
# leg 0: the ladder on fakes — the whole campaign rehearsed for free
# ---------------------------------------------------------------------------

def training_host(fleet, base: str):
    """The host serving the TRAINING capability for one base — the thing the
    collision used to destroy (#51a): two bases' learner regimes are named
    alike, so before #52 the second carve replaced the first host in the
    fleet's dict while its metal stayed resident."""
    return next(host for host in fleet.hosts.values()
                for regime in host.regimes
                if regime.kind == "training" and regime.base == base)


def preview_ladder(store) -> dict:
    """Every placement decision the metal legs depend on, decided on fakes:
    the static partition carves, the joins, both refusals, the live carve, and
    what the observer can see of a carved host afterwards.

    Order matters here exactly as it does on metal — the refusals are asked
    while the residual is still honest, because the live carve is what makes
    it dishonest (see the collision block)."""
    from rlstack import (
        Acquire, Carve, FakeEngine, FakeLearner, Fleet, FleetError, Join,
        fraction_for_gb,
    )

    metal = l4_solo()
    fleet = Fleet((metal,), store=store,
                  engine_factory=lambda r, p: FakeEngine(base=r.base, tp=r.shape),
                  learner_factory=lambda r, p: FakeLearner(fsdp=r.shape))

    print("\n-- leg 1 rehearsal: the static partition ---------------------")
    first = make_spec(store, master=1)
    plan = fleet.place(first)
    print("   ", [type(s).__name__ for s in plan.steps])
    fleet.apply(plan)
    judged = make_spec(store, master=2, post=("llm_judge", "grpo_advantage"),
                       judge=(ALT_BASE, 0.15))
    fleet.apply(fleet.place(judged))
    check("preview: three hosts carved from one device", len(fleet.hosts) == 3,
          str(sorted(fleet.hosts)))
    check("preview: residual is what the arithmetic says",
          abs(fleet.residual(METAL_NAME)[0] - 0.35) < 1e-9,
          f"{fleet.residual(METAL_NAME)}")

    print("\n-- leg 3 rehearsal: a later tenant joins, fraction-free ------")
    again = fleet.place(make_spec(store, master=3))
    check("preview: a second tenant joins every partition",
          all(isinstance(s, Join) for s in again.steps),
          str([type(s).__name__ for s in again.steps]))
    loads = {name: host.arbiter.declared_load()
             for name, host in fleet.hosts.items()}
    check("preview: joins move no declared load",
          all(v == 0.0 for v in loads.values()), str(loads))

    print("\n-- leg 4 rehearsal: the refusals, on a nearly full device ----")
    wide = fleet.place(make_spec(store, master=5, tp=2))
    check("preview: tp=2 on a one-device metal needs a human",
          wide.needs_human and isinstance(wide.steps[0], Acquire),
          str([type(s).__name__ for s in wide.steps]))
    fat_hint = fraction_for_gb(20.0, metal)
    fat = fleet.place(make_spec(store, master=6, base="Qwen/Qwen2.5-0.5B",
                                main_fraction=fat_hint,
                                learner_fraction=fat_hint))
    check("preview: a 20 GB demand on a 0.35 residual needs a human",
          fat.needs_human and all(isinstance(s, Acquire) for s in fat.steps),
          str([type(s).__name__ for s in fat.steps]))
    try:
        fleet.apply(fat)
        check("preview: apply refuses an acquire plan", False, "no raise")
    except FleetError as refusal:
        check("preview: apply refuses an acquire plan", True, str(refusal)[:90])
    try:
        fraction_for_gb(30.0, metal)
        check("preview: 30 GB on a 24 GB device is the acquire rung", False,
              "no raise")
    except FleetError as refusal:
        check("preview: 30 GB on a 24 GB device is the acquire rung", True,
              str(refusal)[:90])

    print("\n-- leg 4 rehearsal: the live carve, and what it collides with -")
    hint = fraction_for_gb(3.0, metal)
    check("preview: fraction_for_gb converts on the metal it will live on",
          abs(hint - 0.125) < 1e-9, f"3 GB of an L4 = {hint}")
    live = make_spec(store, master=4, base=ALT_BASE, main_fraction=0.15,
                     learner_fraction=hint)
    plan = fleet.place(live)
    carves = [s for s in plan.steps if isinstance(s, Carve)]
    check("preview: an uncovered training capability carves live",
          len(carves) == 1 and carves[0].memory == hint,
          str([(type(s).__name__, getattr(s, "memory", None))
               for s in plan.steps]))
    before = training_host(fleet, BASE)
    fleet.apply(plan)
    names = [e["host"] for e in store.read_fleet_log()
             if e.get("event") == "carve"]
    check("preview: every carve names a distinct host",
          len(names) == len(set(names)), f"carved {names}")
    check("preview: a live carve adds a host, never replaces one",
          len(fleet.hosts) == 4 and fleet.hosts.get(before.name) is before,
          f"{len(fleet.hosts)} hosts; {before.name} still serves "
          f"{[r.base for r in before.regimes]}, beside "
          f"{[r.base for r in training_host(fleet, ALT_BASE).regimes]}")
    check("preview: a live carve leaves the residual honest",
          abs(fleet.residual(METAL_NAME)[0] - 0.225) < 1e-9,
          f"residual {fleet.residual(METAL_NAME)[0]:.3f}, expected 0.225 "
          f"(0.35 - {hint})")

    print("\n-- leg 6 rehearsal: what the observer can see ----------------")
    print("    carved host names:", sorted(fleet.hosts))
    print("    store.list_hosts():", store.list_hosts())
    check("preview: the observer can name every carved host",
          sorted(fleet.hosts) == store.list_hosts(),
          f"carved {sorted(fleet.hosts)} vs journaled {store.list_hosts()}")
    for event in store.read_fleet_log():
        print(f"    fleet: {event}")
    return {"hosts": sorted(fleet.hosts), "listed": store.list_hosts()}


@app.function(image=image, volumes={"/hf": hf_cache}, timeout=1800)
def preview() -> dict:
    """The rehearsal in the deploy image, on fakes: no GPU, no GPU minutes.

    It also pulls both bases into the HF cache volume, because a weight
    download is the same number of bytes whether or not a GPU is idle
    underneath it."""
    import tempfile

    from huggingface_hub import snapshot_download

    from rlstack.data.stores.local import LocalStore

    for name in (BASE, ALT_BASE):
        path = snapshot_download(name, ignore_patterns=["*.pt", "*.bin",
                                                        "*.pth", "*.h5"])
        check(f"preview: {name} is in the cache", bool(path), path)
    hf_cache.commit()
    out = preview_ladder(LocalStore(tempfile.mkdtemp()))
    out["checks"] = summarize("preview")
    return out


# ---------------------------------------------------------------------------
# legs 1, 3, 4, 5: the static partition, its tenants, a live carve, resume
# ---------------------------------------------------------------------------

@app.function(image=image, gpu="L4", volumes=VOLUMES, timeout=7200,
              cpu=8.0, memory=32768)
def mass_partition(n_updates: int = 4, kill_after: float = 75.0,
                   stagger: float = 5.0, seed_base: int = 500) -> dict:
    """One L4, carved into three hosts, then five tenants, then a fourth host
    carved live out of the residual.

    `seed_base` moves every tenant's master seed and therefore every run's
    identity: a re-run against the same store with the same seeds would
    RESUME finished runs (correct, and instant), which is not what a
    measurement wants."""
    import asyncio

    import torch
    import transformers
    import vllm

    from rlstack import (
        Acquire, Carve, Fleet, FleetError, Join, ModalVolumeStore,
        fraction_for_gb,
    )
    from rlstack.policy.siteschema import hf_schema

    started = time.monotonic()
    print(f"[pins] vllm={vllm.__version__} torch={torch.__version__} "
          f"transformers={transformers.__version__}")
    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    metal = l4_solo()
    factories = Partitioned()
    fleet = Fleet((metal,), store=store, engine_factory=factories.engine,
                  learner_factory=factories.learner)
    out: dict = {}
    # the volume's fleet log is CUMULATIVE — every campaign that ever ran
    # against this store is in it, pre-#52 names included. This campaign's own
    # events start where the log stood when the container opened.
    log_from = len(store.read_fleet_log())

    async def main() -> dict:
        hbm("container start (bare CUDA context)")

        # ---- leg 1: the static partition --------------------------------
        print("\n== leg 1: carve a static partition out of one L4 ==========")
        first = make_spec(store, master=seed_base + 1, n_updates=n_updates)
        plan = fleet.place(first)
        print("    plan:", [(type(s).__name__, getattr(s, "memory", None))
                            for s in plan.steps])
        fleet.apply(plan)
        judged = make_spec(store, master=seed_base + 5, n_updates=n_updates,
                           post=("llm_judge", "grpo_advantage"),
                           judge=(ALT_BASE, 0.15))
        plan = fleet.place(judged)
        print("    plan:", [(type(s).__name__, getattr(s, "memory", None))
                            for s in plan.steps])
        fleet.apply(plan)
        check("leg1: three hosts carved from ONE device", len(fleet.hosts) == 3,
              str(sorted(fleet.hosts)))
        residual = fleet.residual(METAL_NAME)[0]
        check("leg1: residual = 1 - 0.30 - 0.15 - 0.20",
              abs(residual - 0.35) < 1e-9, f"residual {residual:.3f}")
        for name, host in sorted(fleet.hosts.items()):
            print(f"    host {name}: {host.partition.row()} "
                  f"regimes={[r.name for r in host.regimes]}")

        # the engines build lazily, so the partition's memory is a claim until
        # something samples through it: two engines, one device, in order
        main_engine = factories.engines["main-tp1"]
        judge_engine = factories.engines["judge-tp1"]
        hbm("before any engine is built")
        await warm(main_engine, "main-tp1 @ 0.30")
        after_main = LEDGER[-1][1]
        await warm(judge_engine, "judge-tp1 @ 0.15")
        after_judge = LEDGER[-1][1]
        check("leg1: TWO vLLM engines coexist on ONE device",
              after_judge > after_main,
              f"main +{after_main - LEDGER[-3][1]:.2f} GiB, "
              f"judge +{after_judge - after_main:.2f} GiB")
        out["hbm_main_engine"] = after_main
        out["hbm_two_engines"] = after_judge

        stats = asyncio.get_running_loop().create_task(
            fleet.hosts[sorted(fleet.hosts)[0]].run_stats(20.0))

        # ---- leg 3: many tenants, joining fraction-free ------------------
        print("\n== leg 3: concurrent tenants join the partitions ==========")
        schema = hf_schema(BASE)
        tenants = {
            "grpo": first,
            "grpo-b": make_spec(store, master=seed_base + 2, n_updates=n_updates),
            "gspo": make_spec(store, master=seed_base + 3, n_updates=n_updates,
                              loss="gspo"),
            "sdft": make_spec(store, master=seed_base + 4, n_updates=n_updates,
                              loss="sdft", post=("verifier",)),
            "judge": judged,
        }
        for name, spec in tenants.items():
            steps = fleet.place(spec).steps
            check(f"leg3: {name} joins, never carves",
                  all(isinstance(s, Join) for s in steps),
                  str([(type(s).__name__, getattr(s, "host", "")) for s in steps]))

        async def launch(name: str, delay: float):
            await asyncio.sleep(delay)
            print(f"  [join] {name} starts (t+{delay:.0f}s)")
            return name, await fleet.submit(tenants[name], schema)

        results = await asyncio.gather(*(launch(name, i * stagger)
                                         for i, name in enumerate(tenants)))
        hbm("after 5 tenants ran on 3 partitions")
        loads = {name: round(host.arbiter.declared_load(), 3)
                 for name, host in sorted(fleet.hosts.items())}
        check("leg3: joins moved no declared load",
              all(v == 0.0 for v in loads.values()), str(loads))
        for name, report in results:
            print(f"\n  -- {name} ({report.run_id})")
            out[name] = report_run(store, report.run_id, name, n_updates,
                                   gap_alarm=0.5 if name == "sdft" else 0.15)
        placed = {event["run_id"] for event in store.read_fleet_log()
                  if event.get("event") == "place"}
        check("leg3: every tenant's placement is journaled under its run_id",
              all(report.run_id in placed for _, report in results),
              f"{len(placed)} place events")

        # ---- leg 4a: the acquire refusals, asked on a full device --------
        # asked BEFORE the live carve, because the live carve is what makes
        # the residual dishonest (leg 4b) — a refusal measured after it would
        # be measuring the bug, not the ladder
        print("\n== leg 4a: the acquire refusal on a full device ===========")
        wide = fleet.place(make_spec(store, master=seed_base + 7, tp=2))
        check("leg4: tp=2 on a one-device metal is a human's call",
              wide.needs_human and isinstance(wide.steps[0], Acquire),
              str([type(s).__name__ for s in wide.steps]))
        fat_hint = fraction_for_gb(20.0, metal)
        fat = fleet.place(make_spec(store, master=seed_base + 8,
                                    base="Qwen/Qwen2.5-0.5B",
                                    main_fraction=fat_hint,
                                    learner_fraction=fat_hint))
        check("leg4: a 20 GB demand on a 0.35 residual is a human's call",
              fat.needs_human and all(isinstance(s, Acquire) for s in fat.steps),
              str([type(s).__name__ for s in fat.steps]))
        try:
            fleet.apply(fat)
            check("leg4: apply refuses to run an acquire plan", False, "no raise")
        except FleetError as refusal:
            print(f"    refusal: {refusal}")
            check("leg4: apply refuses to run an acquire plan", True,
                  str(refusal)[:80])

        # ---- leg 5: kill one tenant, resume it through the fleet ---------
        print("\n== leg 5: cancel mid-flight, resubmit, resume =============")
        victim_updates = n_updates + 4
        victim = make_spec(store, master=seed_base + 9, n_updates=victim_updates)
        task = asyncio.ensure_future(fleet.submit(victim, schema))
        await cancel_after(task, kill_after)
        del task
        print("  cancelled; resubmitting the same spec")
        report = await fleet.submit(victim, schema)
        check("leg5: the resubmission attached to committed state",
              report.resumed_from is not None
              and 0 < report.resumed_from <= victim_updates,
              f"resumed_from={report.resumed_from} of {victim_updates}")
        out["resume"] = report_run(store, report.run_id, "resume",
                                   victim_updates)

        # ---- leg 4b: a live carve out of the residual --------------------
        print("\n== leg 4b: live carve from the residual ===================")
        hint = fraction_for_gb(3.0, metal)
        check("leg4: fraction_for_gb sizes the hint on real metal",
              abs(hint - 0.125) < 1e-9, f"3 GB of a 24 GB L4 = {hint}")
        live = make_spec(store, master=seed_base + 6, n_updates=2, base=ALT_BASE,
                         main_fraction=0.15, learner_fraction=hint)
        plan = fleet.place(live)
        carves = [s for s in plan.steps if isinstance(s, Carve)]
        check("leg4: the uncovered capability carves LIVE",
              len(carves) == 1 and abs(carves[0].memory - hint) < 1e-9,
              str([(type(s).__name__, getattr(s, "memory", None))
                   for s in plan.steps]))
        before = training_host(fleet, BASE)
        fleet.apply(plan)
        carve_events = [e for e in store.read_fleet_log()[log_from:]
                        if e.get("event") == "carve"]
        check("leg4: the carve is journaled with its metal stamped",
              all(e.get("gpu") == "L4" for e in carve_events),
              str([(e["host"], e["gpu"], e["memory"]) for e in carve_events]))
        names = [e["host"] for e in carve_events]
        check("leg4: every carve names a distinct host",
              len(names) == len(set(names)), f"carved {names}")
        check("leg4: a live carve adds a host, never replaces one",
              len(fleet.hosts) == 4
              and fleet.hosts.get(before.name) is before,
              f"{len(fleet.hosts)} hosts; {before.name} still serves "
              f"{[r.base for r in before.regimes]}, beside "
              f"{[r.base for r in training_host(fleet, ALT_BASE).regimes]}")
        check("leg4: a live carve leaves the residual honest",
              abs(fleet.residual(METAL_NAME)[0] - 0.225) < 1e-9,
              f"residual {fleet.residual(METAL_NAME)[0]:.3f}, expected 0.225")
        report = await fleet.submit(live, hf_schema(ALT_BASE))
        out["live-carve"] = report_run(store, report.run_id, "live-carve", 2)
        hbm("end of campaign (4 partitions' worth of metal resident)")

        stats.cancel()
        for name, host in sorted(fleet.hosts.items()):
            print(f"\n[status] {host.status()}")
        return out

    result = asyncio.run(main())
    store_volume.commit()
    print_ledger("mass_partition")
    result["ledger"] = [(label, round(used, 2)) for label, used, _ in LEDGER]
    result["gpu_minutes"] = round((time.monotonic() - started) / 60.0, 2)
    print(f"\n[gpu-minutes] mass_partition {result['gpu_minutes']}")
    result["checks"] = summarize("mass_partition")
    return result


# ---------------------------------------------------------------------------
# leg 2: the alternating host, with hooks that really evict
# ---------------------------------------------------------------------------

def vllm_sleep_hooks(engine, facts: dict):
    """The engine's OWN sleep seam, as the arbiter's evict/wake hooks (#52).

    `engine.sleep()` is vLLM's sleep(level=1) — weights to host RAM, KV cache
    discarded — and `engine.wake()` its inverse; both are idempotent and quiet
    before the lazy build, so this deploy attaches alternation without
    touching a single private (the #51d repro reached `engine._llm` and poked
    `_engine_args` to get here). All that is left for the harness is the
    measurement: the HBM ledger on either side of each verb."""

    async def evict() -> None:
        before = hbm("evict: engine before sleep")
        await engine.sleep()
        after = hbm("evict: engine after sleep")
        facts["freed"] = max(facts.get("freed", 0.0), before - after)

    async def wake() -> None:
        before = hbm("wake: engine before wake_up")
        await engine.wake()
        hbm("wake: engine after wake_up")
        facts["woke"] = facts.get("woke", 0) + 1
        facts["reclaimed"] = before

    return wake, evict


def offload_hooks(learner, facts: dict):
    """The learner's half of the alternation: the base (and the deltas wired
    into its tree) move to host RAM on evict and back on wake. The arbiter
    only switches when in-flight work is zero, so no step ever runs against a
    half-offloaded module."""
    import torch

    state = {"offloaded": False}

    async def evict() -> None:
        if learner._model is None or state["offloaded"]:
            return
        before = hbm("evict: learner before offload")
        learner._model.to("cpu")
        torch.cuda.empty_cache()
        after = hbm("evict: learner after offload")
        state["offloaded"] = True
        facts["freed"] = max(facts.get("freed", 0.0), before - after)

    async def wake() -> None:
        if learner._model is None or not state["offloaded"]:
            return
        hbm("wake: learner before reload")
        learner._model.to(learner.device)
        hbm("wake: learner after reload")
        state["offloaded"] = False
        facts["woke"] = facts.get("woke", 0) + 1

    return wake, evict


@app.function(image=image, gpu="L4", volumes=VOLUMES, timeout=5400,
              cpu=8.0, memory=32768)
def alternating(n_updates: int = 4, memory: float = 0.45) -> dict:
    """ONE multi-regime host: main pool + learner on one partition, worn in
    turns. The regimes attach at birth into the host's own exclusive group
    ("host:<name>"); this leg then attaches REAL wake/evict hooks onto those
    same residents — attach is idempotent and fills the missing hooks — and
    proves the alternation both happens (arbiter.switches) and costs the
    device something (the HBM ledger)."""
    import asyncio

    import torch
    import vllm

    from rlstack import Fleet, ModalVolumeStore
    from rlstack.policy.siteschema import hf_schema

    started = time.monotonic()
    print(f"[pins] vllm={vllm.__version__} torch={torch.__version__}")
    import dataclasses

    from vllm import AsyncEngineArgs

    known = ({f.name for f in dataclasses.fields(AsyncEngineArgs)}
             if dataclasses.is_dataclass(AsyncEngineArgs)
             else set(vars(AsyncEngineArgs)))
    supported = "enable_sleep_mode" in known
    check("leg2: this vLLM build has a sleep mode to ask for", supported,
          f"AsyncEngineArgs.enable_sleep_mode present={supported}")

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    factories = Partitioned(sleepy=supported)
    fleet = Fleet((l4_solo(),), store=store, engine_factory=factories.engine,
                  learner_factory=factories.learner)
    engine_facts: dict = {}
    learner_facts: dict = {}

    async def main() -> dict:
        hbm("container start (bare CUDA context)")
        spec = make_spec(store, master=601, n_updates=n_updates,
                         main_fraction=memory, learner_fraction=memory,
                         sharing="sleep")
        plan = fleet.place(spec)
        check("leg2: a sleep unit carves ONE host with two regimes",
              len(plan.steps) == 1 and len(plan.steps[0].regimes) == 2,
              str([(type(s).__name__, [r.name for r in getattr(s, 'regimes', ())])
                   for s in plan.steps]))
        placement = fleet.apply(plan)
        host = placement["main"]
        check("leg2: one host wears both regimes", placement[None] is host,
              f"{host.name} {[r.name for r in host.regimes]}")
        check("leg2: the host owns one exclusive group",
              list(host.arbiter.residency()) == [f"host:{host.name}"],
              str(host.arbiter.residency()))

        engine, learner = host.engines[0], host.learner
        check("leg2: the engine was BUILT able to hand the device back",
              engine.sleeps, f"VllmEngine.sleeps={engine.sleeps} "
              f"(enable_sleep_mode, a build fact — #52)")
        wake_e, evict_e = vllm_sleep_hooks(engine, engine_facts)
        wake_l, evict_l = offload_hooks(learner, learner_facts)
        host.arbiter.attach(engine, label=f"{host.name}:main-tp1",
                            wake=wake_e, evict=evict_e)
        host.arbiter.attach(learner, label=f"{host.name}:learner-fsdp1",
                            wake=wake_l, evict=evict_l)
        check("leg2: attach filled the hooks without re-legislating the group",
              host.arbiter.attached_group(engine) == f"host:{host.name}",
              str(host.arbiter.attached_group(engine)))

        stats = asyncio.get_running_loop().create_task(host.run_stats(20.0))
        report = await fleet.submit(spec, hf_schema(BASE))
        stats.cancel()
        hbm("after the alternating run")

        switches = host.arbiter.switches
        print(f"\n  switches ({len(switches)}): {switches}")
        check("leg2: the host actually alternated", len(set(switches)) == 2,
              f"{len(switches)} switches over {sorted(set(switches))}")
        check("leg2: vLLM's sleep returned HBM to the device",
              engine_facts.get("freed", 0.0) > 1.0,
              f"freed {engine_facts.get('freed', 0.0):.2f} GiB, "
              f"woke {engine_facts.get('woke', 0)}x "
              f"{engine_facts.get('missing', '')}")
        check("leg2: the learner offload returned HBM to the device",
              learner_facts.get("freed", 0.0) > 0.5,
              f"freed {learner_facts.get('freed', 0.0):.2f} GiB, "
              f"woke {learner_facts.get('woke', 0)}x")
        out = {"switches": switches, "engine": engine_facts,
               "learner": learner_facts}
        out["run"] = report_run(store, report.run_id, "alternating", n_updates)
        print(f"\n[status] {host.status()}")
        return out

    result = asyncio.run(main())
    store_volume.commit()
    print_ledger("alternating")
    result["ledger"] = [(label, round(used, 2)) for label, used, _ in LEDGER]
    result["gpu_minutes"] = round((time.monotonic() - started) / 60.0, 2)
    print(f"\n[gpu-minutes] alternating {result['gpu_minutes']}")
    result["checks"] = summarize("alternating")
    return result


# ---------------------------------------------------------------------------
# leg 7: what the metal does when the partition arithmetic lies
# ---------------------------------------------------------------------------

@app.function(image=image, gpu="L4", volumes=VOLUMES, timeout=3600,
              cpu=8.0, memory=32768)
def overcommit(first: float = 0.30, second: float = 0.75) -> dict:
    """The overcommit boundary, measured — because the #51a collision was a
    way to reach it BY ACCIDENT (a replaced host returned a live partition's
    memory to the residual, so the next carve was sized against memory that
    was already gone). #52 closed that path; the boundary itself is still
    worth knowing.

    Two questions, and only the metal can answer them: what does a partition
    sized past the device DO, and does its failure take down the partitions
    that were already serving? A fleet that mass-partitions one device needs
    the second answer to be 'no'."""
    import asyncio

    from rlstack import ModalVolumeStore

    started = time.monotonic()
    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    factories = Partitioned()

    async def main() -> dict:
        from rlstack import Partition, Regime

        def partition(memory: float) -> Partition:
            return Partition(METAL_NAME, (0,), memory, "L4")

        hbm("container start (bare CUDA context)")
        engine_a = factories.engine(Regime("a-tp1", "inference", BASE, 1),
                                    partition(first))
        await warm(engine_a, f"a-tp1 @ {first}")
        print(f"\n  now asking for {second} of a device that has "
              f"{1.0 - first:.2f} left")
        engine_b = factories.engine(Regime("b-tp1", "inference", ALT_BASE, 1),
                                    partition(second))
        failure = None
        try:
            await warm(engine_b, f"b-tp1 @ {second}")
        except BaseException as raised:          # noqa: BLE001 - it is the datum
            failure = raised
            print(f"  [refused] {type(raised).__name__}: {str(raised)[:400]}")
        check(f"leg7: {first} + {second} of one device is refused, not served",
              failure is not None,
              "the build came back happy — the device was overcommitted"
              if failure is None else f"{type(failure).__name__}")
        surviving = None
        try:
            await warm(engine_a, "a-tp1 after the refused build")
            surviving = True
        except BaseException as raised:          # noqa: BLE001
            surviving = False
            print(f"  [poisoned] {type(raised).__name__}: {str(raised)[:400]}")
        check("leg7: the partition that was already serving survives it",
              bool(surviving), "the first engine still samples"
              if surviving else "a failed build took a healthy partition down")
        return {"failure": type(failure).__name__ if failure else None,
                "detail": str(failure)[:400] if failure else "",
                "survivor_ok": surviving}

    result = asyncio.run(main())
    print_ledger("overcommit")
    result["ledger"] = [(label, round(used, 2)) for label, used, _ in LEDGER]
    result["gpu_minutes"] = round((time.monotonic() - started) / 60.0, 2)
    print(f"\n[gpu-minutes] overcommit {result['gpu_minutes']}")
    result["checks"] = summarize("overcommit")
    return result


# ---------------------------------------------------------------------------
# leg 6: what the observer can see of a mass-partitioned device
# ---------------------------------------------------------------------------

@app.function(image=image, volumes=VOLUMES, timeout=900)
def observe() -> dict:
    """The three views over the journals this campaign wrote — and the check
    that a CARVED host is visible in them at all."""
    from rlstack import ModalVolumeStore
    from rlstack.observe import render_gpu, render_hosts, render_runs

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    store_volume.reload()
    for view in (render_hosts, render_runs, render_gpu):
        print(view([store]), end="")
        print("-" * 72)

    log = store.read_fleet_log()
    every = [e["host"] for e in log if e.get("event") == "carve"]
    # the pre-#52 carve names are still in this volume's history and always
    # will be: they contain "/", so they journaled one directory deeper than
    # list_hosts() looks and NOTHING can recover them (#51b). They are the
    # finding, kept; the check is about the names a fixed fleet writes.
    legacy = sorted({name for name in every if "/" in name})
    carved = sorted({name for name in every if "/" not in name})
    listed = store.list_hosts()
    print(f"\ncarved hosts (fleet log): {carved}")
    print(f"observer's list_hosts()  : {listed}")
    print(f"pre-#52 names, invisible for good: {legacy}")
    check("leg6: the observer can name every carved host",
          all(name in listed for name in carved),
          f"{len(carved)} carved, {len(listed)} listed, "
          f"{len(legacy)} pre-#52 names unreachable")
    journaled = {name: [e for e in store.read_host_log(name)
                        if e.get("event") == "host-up"] for name in carved}
    check("leg6: every carved host journaled host-up with partition+regimes",
          all(ups and ups[-1].get("partition") and ups[-1].get("regimes")
              for ups in journaled.values()),
          str({k: len(v) for k, v in journaled.items()}))
    check("leg6: the hosts view prints the metal a partition is made of",
          "metal   : L4 " in render_hosts([store]),
          "looked for 'metal   : L4 '")
    print("\n-- the fleet log, whole ------------------------------------")
    for event in log:
        print(f"  {event}")
    return {"carved": carved, "listed": listed, "checks": summarize("observe")}


@app.local_entrypoint()
def main(n_updates: int = 4, seed_base: int = 500) -> None:
    """The campaign, in order: rehearse on fakes, then spend the L4. Give a
    fresh `seed_base` for a fresh measurement — the same seeds against the
    same store resume finished runs instead of running them."""
    print("\n[preview]", preview.remote()["checks"])
    print("\n[mass_partition]", mass_partition.remote(
        n_updates, 20.0, 5.0, seed_base)["checks"])
    print("\n[alternating]", alternating.remote(n_updates)["checks"])
    print("\n[overcommit]", overcommit.remote()["checks"])
    print("\n[observe]", observe.remote()["checks"])
