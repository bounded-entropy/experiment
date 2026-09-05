"""A SECOND METAL for the burgers venue: a card of its own for a solo campaign.

    MODAL_PROFILE=yu-masala-workspace modal deploy deploy/burgers_metal_b.py
    MODAL_PROFILE=yu-masala-workspace modal run deploy/burgers_metal_b.py::up
    MODAL_PROFILE=yu-masala-workspace modal run deploy/concept_burgers.py::submit \\
        --site 0-8 --adapter nsteer --algo opd --train-tasks cas://<sha> --solo

WHY A SECOND APP AND NOT A SECOND CLASS: redeploying `rlstack-concept-burgers`
while its metal carries runs would strand them (an app is an address, ADR
0007 Q3); a second metal in a second app is reachable without touching the
first, and the ONE desk places across both. A `--solo` submit prefers a metal
with nothing standing on it, which is what this one is for (Samarth,
2026-09-05: "please request solo container cuz i want this to run fast").

The card, the idle clock and the recipe are `concept_burgers.py`'s, restated
rather than imported: a venue file is the only Python Modal ships with a metal
container besides the chassis, so importing the sibling would fail at boot.
"""

from __future__ import annotations

import json
import os

import modal

from modal_venue import gpu_image_for, metal_class, metal_handle, wait_for_metal

APP = "rlstack-concept-burgers-b"
app = modal.App(APP)
GPUS = [g for g in os.environ.get("RLSTACK_BURGERS_GPU", "A100-80GB:1"
                                  ).split(",") if g]
gpu_image = gpu_image_for()

METAL = "burgers-a100-b"
IDLE_S = 90.0


def proposed_recipe():
    """`concept_burgers.proposed_recipe`, restated (see the module docstring)."""
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    return Builds(
        engine=EngineBuild(max_model_len=4096, max_bundles=32, max_rank=16,
                           serves=("steer", "nsteer", "lora"), enforce_eager=True,
                           enable_sleep_mode=False),
        learner=LearnerBuild(checkpoint_activations=True))


MetalS = metal_class(app, APP, METAL, GPUS, gpu_image, module=__name__,
                     idle_s=IDLE_S, recipe=proposed_recipe())


@app.local_entrypoint()
def up() -> None:
    """Boot this metal (the keepalive's spawn is the knock) and wait until it
    has registered itself with the one desk."""
    call = metal_handle(APP).serve.spawn()
    print(f"[up] {METAL} serving: call {call.object_id}")
    print(json.dumps(wait_for_metal(METAL), indent=2))
