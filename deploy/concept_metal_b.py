"""A SECOND METAL for the concept venue: the student arms beside the teacher.

    modal deploy deploy/concept_metal_b.py          # this metal's app
    modal run deploy/concept_metal_b.py::up         # boot it, wait for the desk
    modal run deploy/concept_steer.py::train --layer 10 --teacher-run <rid>

WHY A SECOND APP AND NOT A SECOND CLASS. Redeploying `rlstack-concept-steer`
while its metal carries the teacher run would route the desk's calls at that
address to a fresh container and strand the tenancy (found on the venue). An
app is an address (ADR 0007, Q3), so a second metal in a second app is reachable
without touching the first, and the ONE desk places across both (Q2): a student
arm submitted through `concept_steer.py::train` lands here when the teacher's
metal has no room for its learner, and paces on the teacher's rollouts as they
seal (refs.py: a store ref says "not yet" while its source is still running).

The card, the idle clock and the recipe are `concept_steer.py`'s, restated
rather than imported: a venue file is the only Python Modal ships with a metal
container besides the chassis, so importing the sibling would fail at boot.
"""

from __future__ import annotations

import json
import os

import modal

from modal_venue import gpu_image_for, metal_class, metal_handle, wait_for_metal

APP = "rlstack-concept-steer-b"
app = modal.App(APP)

GPUS = [g for g in os.environ.get("RLSTACK_CONCEPT_GPU", "A100-80GB:2,H100:2"
                                  ).split(",") if g]
gpu_image = gpu_image_for()

METAL = "concept-a100-b"
IDLE_S = 1800.0


def proposed_recipe():
    """`concept_steer.proposed_recipe`, restated (see the module docstring)."""
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    return Builds(
        engine=EngineBuild(max_model_len=4096, max_bundles=8, max_rank=16,
                           serves=("steer", "nsteer"), enforce_eager=True,
                           enable_sleep_mode=True),
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
