"""THE ANGER CAMPAIGN'S MEASUREMENT, as its own app (2026-09-05).

    MODAL_PROFILE=yu-masala-workspace modal deploy deploy/anger_measure.py
    MODAL_PROFILE=yu-masala-workspace modal run deploy/anger_measure.py::measure \\
        --run-id <rid> [--address modal://rlstack-concept-anger/MetalS#<host>]
    MODAL_PROFILE=yu-masala-workspace modal run deploy/anger_measure.py::loop \\
        --run-ids a,b,c --address ... --every-s 600 --for-s 7200

WHY A THIRD APP. The measurement is `concept_steer.py`'s `measure` door,
restated: one idempotent pass (`measure_run`) over a run's ledger that
samples the held-out prompts under every 8th committed version, scores those
tokens through the conditioned teacher, and records `reverse_kl` per point.
It lives in its own app because redeploying `rlstack-concept-anger` while its
metal carries the teacher set would strand that metal's address (found on the
first venue), and a CPU function needs no metal of its own: it is a PURE
CLIENT of a serving host.

WHICH HOST. Given `--address`, that one; otherwise the desk's `place` door
resolves the first listing serving `nsteer` on the base. Prefer the TEACHER'S
metal while it stands: its engine is awake and alone. The arms' host
alternates its pool with a learner that already holds its shard, and the
pool's FIRST build on that pair is the boot loop of 2026-09-05 (46 GiB free
of 79, 64 wanted) — a hazard this door does not walk into by default.
"""
from __future__ import annotations

import json
import time

import modal

from concept_campaign import Campaign
from modal_venue import a_store, cpu_image_for, desk, store_volume

APP = "rlstack-concept-anger-measure"
app = modal.App(APP)
cpu_image = cpu_image_for().add_local_python_source("concept_campaign")

# concept_anger.py's constants, restated (a venue file is the only Python a
# container ships besides the chassis; importing the sibling would build its
# metal class inside this app)
BASE, HIDDEN, WIDTH = "Qwen/Qwen3-32B", 5120, 2
MAIN_GB = 44.0 * WIDTH
SUBDIR = "anger"
HELDOUT = "cas://51d2778d99c9256ca34d750b593824e6b6fa7bd0a19af7981ec5a9ffefce9fe1"
ANGER = Campaign(base=BASE, hidden=HIDDEN, width=WIDTH, anchors=(10, 32, 54),
                 concept="anger", subdir=SUBDIR, main_gb=MAIN_GB,
                 learner_gb=64.0 * WIDTH, split=False)


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=3600)
def measure_once(run_id: str, heldout_tasks: str, address: str) -> dict:
    """One idempotent measuring pass against the serving host at `address`.
    "teacher" and "main" route to the SAME engine under different bundles:
    the student's restored version for `main`, a payload-free base bundle
    for `teacher` — the bare 32B told about anger by the hint."""
    import asyncio

    from rlstack import load_tasks, measure_run
    from rlstack.runner.remote import RemotePool, transport_for

    store_volume.reload()
    store = a_store()
    tasks = {t.id: t for t in load_tasks(store, heldout_tasks)}
    pool = RemotePool(transport_for(address), base=BASE, tp=WIDTH)
    fresh = asyncio.run(measure_run(
        store, run_id, ANGER.the_measurement(sorted(tasks)), pool, tasks,
        pools={"teacher": pool}))
    store_volume.commit()
    told = store.read_measurements(run_id).get("distill", {})
    return {"run_id": run_id, "measured": fresh,
            "points": [{"update": p["update"], **p["means"]}
                       for p in told.get("points", [])]}


def serving_address(address: str) -> str:
    """The host to measure through: the one given, else the desk's resolution
    of an nsteer-serving pool on the base (a PURE CLIENT's placement)."""
    import asyncio

    from rlstack.runner.desk import Demand

    if address:
        return address
    placed = asyncio.run(desk().resolve((
        Demand(pool="main", capability="inference", base=BASE, shape=WIDTH,
               vram_gb=MAIN_GB, group=0, adapter_types=("nsteer",)),)))
    if not placed.get("placed"):
        raise SystemExit(f"no serving host: {placed}")
    return placed["pools"]["main"]


@app.local_entrypoint()
def measure(run_id: str = "", address: str = "",
            heldout_tasks: str = HELDOUT) -> None:
    """One pass for one run: every 8th committed version not yet measured."""
    if not run_id:
        raise SystemExit("--run-id <rid> [--address modal://...]")
    where = serving_address(address)
    print(f"[measure] {run_id} through {where}", flush=True)
    print(json.dumps(measure_once.remote(run_id, heldout_tasks, where), indent=1))


@app.local_entrypoint()
def loop(run_ids: str = "", address: str = "", every_s: float = 600.0,
         for_s: float = 7200.0, heldout_tasks: str = HELDOUT) -> None:
    """The passes on a cadence, for several runs, until `for_s` elapses:
    each pass backfills whatever committed since the last. The traffic it
    sends is admitted traffic, which the idle rule counts, so the host it
    measures through is not released under its own sampling."""
    ids = [r for r in run_ids.split(",") if r]
    if not ids:
        raise SystemExit("--run-ids a,b,c [--address ...]")
    where = serving_address(address)
    start = time.time()
    while time.time() - start < for_s:
        for rid in ids:
            try:
                told = measure_once.remote(rid, heldout_tasks, where)
                last = told["points"][-1] if told["points"] else None
                print(f"[loop {time.strftime('%H:%M:%S')}] {rid}: measured "
                      f"{told['measured']} — {len(told['points'])} point(s), "
                      f"last {last}", flush=True)
            except Exception as refused:
                print(f"[loop {time.strftime('%H:%M:%S')}] {rid}: {refused}",
                      flush=True)
        time.sleep(every_s)
