"""Land the DAPO-Math-17k task sets on the rlstack-store volume.

    MODAL_PROFILE=yu-masala-workspace modal run deploy/tasks_dapo.py::build

Deployment only (I5): an image, a volume, and one call. The prompt, the
integer filter and the split rule are semantics and live in
rlstack/data/tasks/, where they hash into a run; this file chooses only WHERE
the bytes land. The build sequence is the CLI's own `build_task_sets`, so
`python -m rlstack tasks dapo_math --store <root>` against a local store and
this against the volume run the same code and print the same table.
"""

from __future__ import annotations

import modal

app = modal.App("rlstack-tasks")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)

# I7: the pins every deploy keeps in sync, in ONE layer so the layer stays
# cache-identical with the campaign images; parquet is this job's own need and
# rides in a layer of its own after it.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .pip_install("pyarrow")
    .add_local_python_source("rlstack")
)


@app.function(image=image, volumes={"/store": store_volume}, timeout=3600)
def build(eval_fraction: float = 0.02, seed: int = 17) -> dict[str, str]:
    """Build, split and write the task sets; print the uris a spec pins.

    One knob, because the fractions must cover the set: eval takes its share
    and train takes the rest.
    """
    from rlstack import ModalVolumeStore
    from rlstack.__main__ import build_task_sets

    store = ModalVolumeStore("/store", volume=store_volume,
                             locator="modal://rlstack-store")
    uris = build_task_sets(
        store, "dapo_math",
        {"train": 1.0 - eval_fraction, "eval": eval_fraction}, seed)
    # a cas write is not a commit point (only a ledger append is), so the
    # volume is persisted here, once the sets are whole
    store_volume.commit()
    return uris
