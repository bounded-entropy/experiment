"""ANGER ON THE 32B: the concept campaign at scale (ADR 0005), Qwen3-32B told
to imbue as many words as it can with anger — a per-token signal where the
burgers block gave a sparse one (Samarth, 2026-09-05: "at least with the
burger prompts, mentions of burgers were so sparse compared to the huge
corpus of other text ... a better emotion is anger ... imbue many words with
anger in the system prompt, so the signal is clearer").

    MODAL_PROFILE=yu-masala-workspace modal run deploy/concept_anger.py::prompts   # the corpus -> cas uris
    MODAL_PROFILE=yu-masala-workspace modal deploy deploy/concept_anger.py        # this venue's metal
    MODAL_PROFILE=yu-masala-workspace modal run deploy/concept_anger.py::up
    MODAL_PROFILE=yu-masala-workspace modal run deploy/concept_anger.py::distill_set --train-tasks cas://<sha>
    MODAL_PROFILE=yu-masala-workspace modal run deploy/concept_anger.py::submit \\
        --layer 14 --adapter nsteer --algo sft --teacher-run <rid>
    MODAL_PROFILE=yu-masala-workspace modal run deploy/concept_anger.py::submit \\
        --layer 14 --adapter nsteer --algo opd --train-tasks cas://<sha>
    MODAL_PROFILE=yu-masala-workspace modal run deploy/concept_anger.py::follow_run --run-id <rid>

The same science as `concept_steer.py` — `concept_campaign.Campaign` is where
it is written out — with three things this record changes (Samarth,
2026-09-05: "let's just use a 0.6B model as a proof of concept that this
works first, still across 3 evenly spaced layers roughly at 15%, 50%, 85%,
before we scale up"):

- THE BASE is Qwen3-32B (64 layers, width 5120), the anchors 10 / 32 / 54 and the thirds 0-20 / 21-42 / 43-63.
- THE PLACEMENT ALTERNATES (split=False): the pool and the learner share
  one 80 GB pair and take turns, as concept_steer.py's did; an SFT arm never
  wakes the pool, so the arms train on the learner's turn alone. The teacher
  set needs its own pair: `anger_metal_b.py` is the second metal.
- THE CONCEPT is anger, its system block in `concept_prompts.SYSTEM_PROMPTS`.

The arms: nsteer under sft at each anchor and each third — six.
"""

from __future__ import annotations

import json
import os

import modal

from concept_campaign import ADAPTERS, ALGOS, Campaign
from modal_venue import (
    a_store, cpu_image_for, follow, gpu_image_for, hf_cache, metal_class,
    metal_handle, progress_function, run_suite, store_volume, submit_and_follow,
    submit_spec, wait_for_metal,
)

APP = "rlstack-concept-anger"
app = modal.App(APP)

# A 32B at tp=2/fsdp=2 needs ~80 GB per device pair (concept_steer.py's
# treaty): the pool and the learner ALTERNATE on one pair (split=False).
GPUS = [g for g in os.environ.get("RLSTACK_ANGER_GPU", "A100-80GB:2,H100:2"
                                  ).split(",") if g]
WIDTH = 2

cpu_image = cpu_image_for().add_local_python_source("concept_campaign")
gpu_image = gpu_image_for(with_tests=True).add_local_python_source("concept_campaign")
tasks_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==5.16.1", "huggingface_hub", "safetensors",
                 "numpy")
    .pip_install("pyarrow", "jinja2")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("rlstack", "modal_venue", "concept_campaign")
)

BASE = "Qwen/Qwen3-32B"
HIDDEN = 5120
SUBDIR = "anger"
ANCHORS = (10, 32, 54)          # of 64: the paper's own anchors, ~15/50/85 %
THIRDS = ("0-20", "21-42", "43-63")
"""The depth in three ranges, one direction per boundary and alpha shared
across a range: the second campaign on this base (2026-09-05)."""
CONCEPT = "anger"
SPLIT_SEED = 5
TRAIN_ASK, HELDOUT_ASK = 2304, 160

# The partition treaty on one 80 GB pair, in GB TOTAL across a member's
# shards (concept_steer.py, MEASURED there): serving ~35 GB/device of weights
# plus KV; the learner 64 GB/device with the chunked logprob pass. The two
# alternate, so the unit's carve is the larger.
MAIN_GB = 44.0 * WIDTH
LEARNER_GB = 64.0 * WIDTH

METAL = "anger-a100"
IDLE_S = 90.0

BURGERS = Campaign(
    base=BASE, hidden=HIDDEN, width=WIDTH, anchors=ANCHORS, concept=CONCEPT,
    subdir=SUBDIR, main_gb=MAIN_GB, learner_gb=LEARNER_GB, split=False)
"""This venue's campaign, whole (see the module docstring)."""


def proposed_recipe():
    """What this metal serves: the three families, on a pool that SLEEPS —
    it shares its pair with the learner and hands the device back between
    waves (ADR 0002 Q8; the 32B cannot co-reside with its own learner)."""
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    return Builds(
        engine=EngineBuild(max_model_len=4096, max_bundles=32, max_rank=16,
                           serves=("steer", "nsteer", "lora"), enforce_eager=True,
                           enable_sleep_mode=True),
        learner=LearnerBuild(checkpoint_activations=True))


MetalS = metal_class(app, APP, METAL, GPUS, gpu_image, module=__name__,
                     idle_s=IDLE_S, recipe=proposed_recipe())


# ---------------------------------------------------------------------------
# the science, by name: the campaign's, for the doors and the tests
# ---------------------------------------------------------------------------

def teacher_spec(store, train_tasks: str):
    return BURGERS.teacher_spec(store, train_tasks)


def student_spec(store, teacher_run: str, layer: int | str, adapter: str = "nsteer"):
    return BURGERS.student_spec(store, teacher_run, layer, adapter)


def opd_spec(store, train_tasks: str, layer: int | str, adapter: str = "nsteer"):
    return BURGERS.opd_spec(store, train_tasks, layer, adapter)


def bank_entry(adapter: str, layer: int | str):
    return BURGERS.bank_entry(adapter, layer)


def probe_spec(store, heldout_tasks: str, parent_run: str | None, version: int,
               layer: int | str, adapter: str = "nsteer"):
    return BURGERS.probe_spec(store, heldout_tasks, parent_run, version, layer, adapter)


def the_measurement(task_ids):
    return BURGERS.the_measurement(task_ids)


# ---------------------------------------------------------------------------
# the volume-side functions
# ---------------------------------------------------------------------------

@app.function(image=tasks_image, volumes={"/store": store_volume,
                                          "/hf": hf_cache}, timeout=3600)
def build_prompts(concept: str = CONCEPT, seed: int = SPLIT_SEED) -> dict:
    """The corpus into the cas, under THIS concept's system block, rendered
    by this base's own chat template (the hint must concatenate byte-for-byte
    with the prompt under it)."""
    from rlstack.__main__ import build_task_sets
    from rlstack.data.tasks.concept_prompts import prompt_splits

    uris = build_task_sets(
        a_store(), "concept_prompts",
        prompt_splits(train=TRAIN_ASK, heldout=HELDOUT_ASK), seed,
        concept=concept, base=BASE)
    store_volume.commit()
    return uris


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def canonical(kind: str, train_tasks: str = "", teacher_run: str = "",
              layer: int | str = 0, adapter: str = "nsteer") -> dict:
    """One spec as its canonical row, for the client to submit. The plan
    bytes go into the cas HERE, on the volume, which is where the host reads
    them (a client-minted plan never reaches it — found 2026-09-05)."""
    from rlstack import canonical_json

    store = a_store()
    spec = (teacher_spec(store, train_tasks) if kind == "teacher"
            else opd_spec(store, train_tasks, layer, adapter) if kind == "opd"
            else student_spec(store, teacher_run, layer, adapter))
    store_volume.commit()
    return json.loads(canonical_json(spec))


progress = progress_function(app, cpu_image, module=__name__)


# ---------------------------------------------------------------------------
# the doors — submit, never release (ADR 0007, Q6)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def prompts(concept: str = CONCEPT, seed: int = SPLIT_SEED) -> None:
    """THE CORPUS. No metal. Prints the uris the specs pin."""
    print(json.dumps(build_prompts.remote(concept, seed), indent=1))


@app.local_entrypoint()
def up() -> None:
    """Boot the metal and wait until it has registered with the desk."""
    call = metal_handle(APP).serve.spawn()
    print(f"[up] {METAL} serving: call {call.object_id}")
    print(json.dumps(wait_for_metal(METAL), indent=2))


@app.local_entrypoint()
def distill_set(train_tasks: str = "", timeout_s: float = 7200.0) -> None:
    """THE TEACHER'S TRAJECTORY SET, as a generation-only run, followed to
    its extent. Prints the run_id the SFT arms name in their plans."""
    if not train_tasks:
        raise SystemExit("--train-tasks <cas uri from ::prompts>")
    metal_handle(APP).serve.spawn()
    print(json.dumps(wait_for_metal(METAL), indent=1), flush=True)
    run_id = submit_and_follow(progress, canonical.remote("teacher", train_tasks),
                               SUBDIR, timeout_s)
    print(f"[set] the teacher's rollouts are run {run_id} — "
          f"pass it to ::submit --teacher-run", flush=True)


@app.local_entrypoint()
def submit(layer: int = 0, adapter: str = "nsteer", algo: str = "sft",
           teacher_run: str = "", train_tasks: str = "", site: str = "",
           solo: bool = False) -> None:
    """ONE ARM, submitted and left running — the desk's reply printed, the
    run followed by `::follow_run` or watched at the observer. An SFT arm
    names the teacher's set; an on-policy arm names the train set."""
    if site and site not in THIRDS:
        raise SystemExit(f"--site must be one of {list(THIRDS)}")
    if not site and layer not in ANCHORS:
        raise SystemExit(f"--layer must be one of {list(ANCHORS)} (or --site)")
    where = site or layer
    if adapter not in ADAPTERS or algo not in ALGOS:
        raise SystemExit(f"--adapter in {ADAPTERS}, --algo in {ALGOS}")
    if algo == "sft" and not teacher_run:
        raise SystemExit("--teacher-run <run_id from ::distill_set>")
    if algo == "opd" and not train_tasks:
        raise SystemExit("--train-tasks <cas uri from ::prompts>")
    row = (canonical.remote("opd", train_tasks, "", where, adapter) if algo == "opd"
           else canonical.remote("student", "", teacher_run, where, adapter))
    # --solo: joins nothing that stands, carves its own listings (a card of
    # its own where one is registered bare) — "i want this to run fast"
    print(json.dumps(submit_spec(row, SUBDIR, solo=solo), default=str)[:600], flush=True)


@app.local_entrypoint()
def follow_run(run_id: str = "", timeout_s: float = 14400.0) -> None:
    """RE-ATTACH to a run on the metal and follow it to its extent."""
    if not run_id:
        raise SystemExit("--run-id <rid>")
    print(f"[follow] {follow(progress, run_id, timeout_s)} reached its extent",
          flush=True)


@app.function(image=gpu_image, timeout=1800)
def run_tests() -> str:
    """The fakes suite inside the image, where the torch-gated cases run."""
    return run_suite()
