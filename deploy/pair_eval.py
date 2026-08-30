"""The SIDE EVALUATOR: measurement without a run, riding the pair's own engine.

    MODAL_PROFILE=yu-masala-workspace modal deploy deploy/pair_eval.py
    ... run deploy/pair_eval.py::run_screen    # pick 10 held-out problems
    ... run deploy/pair_eval.py::run_tick      # backfill + catch up, once
    ... run deploy/pair_eval.py::show          # the comparison, aligned

The pair's IN-RUN eval measures a floor-scoring task set for its whole plan
— identity pins it there (I3). This process is the sanctioned other half:
eval is FIREWALLED measurement and retention keeps every adapter version
forever precisely so "a version the run has long moved past" can be rebuilt
(KeepRestorable), so a separate client can measure ANY committed version on
a CORRECTED task set — backfilling the past, which no in-run daemon could —
and keep pace with new seals on a schedule.

It asks the DESK where inference is served and JOINS that pool as one more
admitted client (no carve — the join is the placement); restored bundles ride
the engine's own LRU residency, so nothing here books or frees metal. It
peeks, never attaches (an attach would race the live trainer's staged
blobs), and writes only its own measurements/ area — a run dir's eval/ is
the run's own record, and this is not the run.
"""

import json
import time

import modal

APP = "rlstack-pair-eval"
FLEET_APP = "rlstack-fleet-a100"
app = modal.App(APP)

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy")
         .add_local_python_source("rlstack", "rlstack_engine"))

BASE = "Qwen/Qwen3-0.6B"
STORE = "modal://rlstack-store"
EVAL_TASKS = "cas://82ae4626dbb59a2c50e2b13cbe7250c5f1ddd02dfb81edc7495efb77759d420b"
RUNS = {"lora": "725889b9812b", "gated": "3bcaa5a0086c"}

SCREEN_N = 128              # eval-split problems screened for the pick
SCREEN_SAMPLES = 8          # completions per problem while screening
BAND = (0.25, 0.75)         # a useful problem has headroom BOTH ways
PICK = 10                   # problems measured, fixed once picked
EVAL_SAMPLES = 4            # completions per problem per measured version
EVERY = 5                   # measure each version the in-run cadence sealed
MASTER = 7                  # the measurement's own seed, never the runs'
MAX_TOKENS = 512

AREA = "measurements/pair-eval"


class DeskTransport:
    def __init__(self) -> None:
        self._handle = None

    def handle(self):
        if self._handle is None:
            self._handle = modal.Cls.from_name(FLEET_APP, "Desk")()
        return self._handle

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.handle().desk.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return self.handle().desk_ask.remote(verb, payload)


class MetalTransport:
    def __init__(self, address: str) -> None:
        self.address = address
        self._handle = None

    def handle(self):
        if self._handle is None:
            self._handle = modal.Cls.from_name(FLEET_APP, "Metal")()
        return self._handle

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.handle().host.remote.aio(self.address, verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return self.handle().host_ask.remote(self.address, verb, payload)


def serving_pool():
    """Ask the desk, join the pool: the live listing wearing an inference
    regime is where this client's traffic goes — placement without a carve."""
    from rlstack.runner.remote import RemoteDesk, RemotePool

    fleet = RemoteDesk(DeskTransport())
    told = fleet.status()["listings"]
    alive = fleet.liveness()
    for name, row in sorted(told.items()):
        if not alive.get(name):
            continue
        if any(not regime.startswith("learner") for regime in row["regimes"]):
            return RemotePool(MetalTransport(row["address"]),
                              base=BASE, tp=1)
    raise RuntimeError(f"no live inference listing at the desk: {told}")


def opened(store):
    """Peeks only: blob reader and adapter-type map for one run, no attach."""
    def reader_for(rid: str):
        prefix = store.run_prefix(rid)
        return lambda section, name, version: store._read(
            f"{prefix}/{section}/{name}@{version}.bin")

    def bank_of(rid: str) -> dict[str, str]:
        manifest = store.peek_manifest(rid)
        row = manifest["spec"]        # the canonical form is stored as a string
        spec = json.loads(row) if isinstance(row, str) else row
        return {name: entry["adapter_type"]
                for name, entry in spec["policy"]["bank"].items()}

    return reader_for, bank_of


async def measure(pool, bundle, tasks, task_ids, samples, update):
    """One wave over the pinned problems under one restored version: the
    exact machinery the in-run evaluator uses, addressed from outside."""
    from rlstack.data.plan import GroupPlan, Sample, WavePlan
    from rlstack.runner.assemble import sample_wave
    from rlstack.runner.post import run_pipeline
    from rlstack.spec.specs import SamplingSpec

    sampling = SamplingSpec(temperature=1.0, max_tokens=MAX_TOKENS)
    plan = WavePlan(tuple(
        GroupPlan(task, tuple(Sample(task, "dapo_math")
                              for _ in range(samples)))
        for task in task_ids))
    routes = {"main": (pool, bundle)}
    wave = await sample_wave(plan, index=update, tasks=tasks,
                             sampling=sampling, routes=routes,
                             master=MASTER, phase="eval")
    scored = await run_pipeline(("final_answer",), wave, routes, sampling,
                                MASTER, update)
    rewards = list(scored["reward"])
    per_task = {task: sum(rewards[i * samples:(i + 1) * samples]) / samples
                for i, task in enumerate(task_ids)}
    return sum(rewards) / len(rewards), per_task


@app.function(image=image, volumes={"/store": store_volume}, timeout=7200)
async def screen() -> dict:
    """Pick the held-out set: SCREEN_N eval-split problems sampled under the
    lora arm's v1 (one update of lr 1e-4 — base plus epsilon), keep the PICK
    with pass rates nearest 0.5 inside BAND. Writes the pick where every
    tick reads it."""
    from rlstack import ModalVolumeStore
    from rlstack.data.tasks import load_tasks
    from rlstack.policy.compile import restore_bundle

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    reader_for, bank_of = opened(store)
    rid = RUNS["lora"]
    first = store.peek_ledger(rid)[0]
    bundle = restore_bundle({n: int(v) for n, v in first["versions"].items()},
                            first["bundle_id"], reader_for(rid),
                            list(bank_of(rid)), bank_of(rid))
    pool = serving_pool()
    if not pool.knows_bundle(bundle.bundle_id):
        pool.add_bundle(bundle)

    tasks = {t.id: t for t in load_tasks(store, EVAL_TASKS)[:SCREEN_N]}
    ids = list(tasks)
    aggregate, rates = await measure(pool, bundle, tasks, ids,
                                     SCREEN_SAMPLES, update=0)
    fit = sorted((task for task, rate in rates.items()
                  if BAND[0] <= rate <= BAND[1]),
                 key=lambda task: (abs(rates[task] - 0.5), task))
    picked = sorted(fit[:PICK])
    chosen = {"picked": picked, "rates": rates,
              "aggregate": sum(rates[t] for t in picked) / max(len(picked), 1),
              "screened": SCREEN_N, "policy": f"{rid}@v1", "t": time.time()}
    store._write(f"{AREA}/tasks.json", json.dumps(chosen).encode())
    store_volume.commit()
    return {"picked": picked, "in_band": len(fit),
            "aggregate": chosen["aggregate"]}


@app.function(image=image, volumes={"/store": store_volume},
              schedule=modal.Period(minutes=10), timeout=3600)
async def tick() -> dict:
    """Backfill and follow: every EVERY-th committed version of each arm not
    yet measured, restored from blobs and measured on the picked problems.
    The first tick reaches back to update EVERY; later ticks keep pace."""
    from rlstack import ModalVolumeStore
    from rlstack.data.tasks import load_tasks
    from rlstack.policy.compile import restore_bundle

    store_volume.reload()
    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    try:
        chosen = json.loads(store._read(f"{AREA}/tasks.json"))
    except FileNotFoundError:
        return {"waiting": "no tasks.json — run screen first"}
    task_ids = chosen["picked"]
    tasks = {t.id: t for t in load_tasks(store, EVAL_TASKS)
             if t.id in set(task_ids)}
    reader_for, bank_of = opened(store)
    pool = serving_pool()

    report = {}
    for arm, rid in RUNS.items():
        try:
            done = json.loads(store._read(f"{AREA}/{arm}.json"))
        except FileNotFoundError:
            done = {}
        for entry in store.peek_ledger(rid):
            update = int(entry["update"])
            if update % EVERY or str(update) in done:
                continue
            bundle = restore_bundle(
                {n: int(v) for n, v in entry["versions"].items()},
                entry["bundle_id"], reader_for(rid),
                list(bank_of(rid)), bank_of(rid))
            if not pool.knows_bundle(bundle.bundle_id):
                pool.add_bundle(bundle)
            mean, per_task = await measure(pool, bundle, tasks, task_ids,
                                           EVAL_SAMPLES, update)
            done[str(update)] = {"mean": mean, "per_task": per_task}
            print(f"[{arm}] u{update}: {mean:.3f}")
        store._write(f"{AREA}/{arm}.json", json.dumps(done).encode())
        report[arm] = {str(u): round(done[u]["mean"], 3)
                       for u in sorted(done, key=int)}
    store_volume.commit()
    return report


@app.function(image=image, volumes={"/store": store_volume}, timeout=600)
def read_out() -> dict:
    store_volume.reload()
    from rlstack import ModalVolumeStore

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    out = {}
    for name in ("tasks", *RUNS):
        try:
            out[name] = json.loads(store._read(f"{AREA}/{name}.json"))
        except FileNotFoundError:
            out[name] = None
    return out


@app.local_entrypoint()
def run_screen() -> None:
    print(json.dumps(screen.remote(), indent=2))


@app.local_entrypoint()
def run_tick() -> None:
    print(json.dumps(tick.remote(), indent=2))


@app.local_entrypoint()
def show() -> None:
    told = read_out.remote()
    tasks = told.pop("tasks") or {}
    print(f"picked {len(tasks.get('picked', []))} problems, screening "
          f"aggregate {tasks.get('aggregate')}")
    rows = {arm: {int(u): v["mean"] for u, v in (told[arm] or {}).items()}
            for arm in RUNS}
    updates = sorted(set().union(*[set(r) for r in rows.values()]))
    print("update  " + "  ".join(f"{arm:>8}" for arm in RUNS))
    for u in updates:
        cells = "  ".join(f"{rows[arm].get(u, float('nan')):8.3f}"
                          for arm in RUNS)
        print(f"{u:6d}  {cells}")
