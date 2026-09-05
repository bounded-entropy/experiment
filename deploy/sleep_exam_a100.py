"""THE SLEEP EXAM (one A100 pair, ~10m): does a sharded learner's sleep hand
the device back?

    MODAL_PROFILE=yu-masala-workspace modal run deploy/sleep_exam_a100.py::run

ADR 0002's alternation — engine and learner taking turns on one partition —
rests on the learner's `sleep` returning its shard's memory to the driver,
and CONTEXT #82 left that CONDITIONAL on metal. This exam measures it in
isolation: the base loaded sharded, sleep / wake, then a real tenant off the
concept venue's own spec builder stepped twice through the chorus, sleep /
wake / sleep again — per-device used memory (`cudaMemGetInfo`, which sees
every rank's process) and the leader's allocator at each phase, and an
inventory of what reachable from the model is still on a device at the end.

MEASURED (2026-09-05, Qwen3-32B, fsdp=2, two A100-80GB): the base shards to
30.5 GiB per device; asleep, 1.1 GiB stays (context and NCCL). After the
tenant's two steps a device holds 45.6 GiB; asleep again, 1.15 GiB. The
sharded offload works, trained or not, and both ranks put their shard down.
What the concept venue's failing engine build had run into (33 GiB per
device, 46 GiB free where it wanted 64) was a learner that had never been
EVICTED, not one that could not sleep: the arbiter's switch used to evict
only the member last admitted, and a learner holds the partition from
install, before any admit (fixed in `Arbiter._switch`, 82103e2).
"""

from __future__ import annotations

import json
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_venue import gpu_image_for, hf_cache  # noqa: E402

DEPLOY = os.path.dirname(os.path.abspath(__file__))

APP = "rlstack-sleep-exam"
app = modal.App(APP)
gpu_image = gpu_image_for().add_local_dir(DEPLOY, remote_path="/root/deploy")
BASE = "Qwen/Qwen3-32B"
WIDTH = 2
MEMORY = 0.807570977917981     # the concept carve's fraction, verbatim
ADAPTER = "mlp_lora"           # the venue's rank-4 arm: no recorded extras to replay
DOCS, DOC_TOKENS, STEPS = 4, 1024, 2


@app.function(image=gpu_image, gpu=f"A100-80GB:{WIDTH}", volumes={"/hf": hf_cache},
              timeout=2400)
def exam() -> dict:
    import asyncio
    import gc

    import torch

    sys.path.insert(0, "/root/deploy")

    from rlstack.runner.learners.fsdp_torch import lead_fsdp_learner

    rows: list[dict] = []

    def read(phase: str) -> None:
        torch.cuda.synchronize()
        row: dict = {"phase": phase}
        for device in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(device)
            row[f"cuda{device}_used_gib"] = round((total - free) / 2 ** 30, 2)
        row["leader_allocated_gib"] = round(torch.cuda.memory_allocated(0) / 2 ** 30, 2)
        row["leader_reserved_gib"] = round(torch.cuda.memory_reserved(0) / 2 ** 30, 2)
        print(json.dumps(row), flush=True)
        rows.append(row)

    read("bare")
    learner = lead_fsdp_learner(WIDTH, memory_fraction=MEMORY)
    read("chorus up")
    learner._ensure_base(BASE)
    read("base sharded")
    asyncio.run(learner.sleep())
    read("asleep, never trained")
    asyncio.run(learner.wake())
    read("awake")

    # now what the wedged host had done before it slept: a real tenant
    # installed off the venue's own spec builder, stepped through the chorus
    import random
    import tempfile

    from concept_steer import student_spec
    from rlstack import LocalStore
    from rlstack.data.flatten import Flat, pack
    from rlstack.policy.siteschema import hf_schema, resolve
    from rlstack.runner.loop import parameterization_of
    from rlstack.spec.validate import site_space

    spec = student_spec(LocalStore(tempfile.mkdtemp()), "e2ae3a58c5bf", 10, ADAPTER)
    space = site_space(spec, hf_schema(BASE))
    resolved = {name: resolve(space, a.site) for name, a in spec.policy.bank.items()}
    learner.install("exam", parameterization_of(spec, resolved))
    read("tenant installed")
    rng = random.Random(13)
    docs = []
    for _ in range(DOCS):
        ids = tuple(rng.randrange(1000, 100000) for _ in range(DOC_TOKENS))
        docs.append((Flat(ids, (1,) * DOC_TOKENS, (0,) * DOC_TOKENS,
                          (0.0,) * DOC_TOKENS, DOC_TOKENS), {}))
    for step in range(STEPS):
        for batch in pack(docs, spec.algo.schedule.microbatch_tokens):
            learner.forward_backward("exam", batch)
        learner.optim_step("exam")
        learner.emit("exam")
        read(f"stepped {step + 1}")
    asyncio.run(learner.sleep())
    read("asleep after training")
    gc.collect()
    torch.cuda.empty_cache()
    read("asleep after training + gc + empty_cache")
    asyncio.run(learner.wake())
    read("awake again")
    asyncio.run(learner.sleep())
    read("asleep again")

    # what stays: every tensor reachable from the model that is still on a device
    params_on_cuda = local_on_cuda = 0
    for param in learner._model.parameters():
        if param.device.type == "cuda":
            params_on_cuda += param.numel() * param.element_size()
        local = getattr(param, "_local_tensor", None)
        if local is not None and local.device.type == "cuda":
            local_on_cuda += local.numel() * local.element_size()
    buffers_on_cuda = sum(b.numel() * b.element_size()
                          for b in learner._model.buffers() if b.device.type == "cuda")
    inventory = {"phase": "inventory (leader rank, asleep)",
                 "params_on_cuda_gib": round(params_on_cuda / 2 ** 30, 3),
                 "local_tensors_on_cuda_gib": round(local_on_cuda / 2 ** 30, 3),
                 "buffers_on_cuda_gib": round(buffers_on_cuda / 2 ** 30, 3),
                 "sleeps": bool(learner.sleeps), "refusal": learner.sleep_refusal}
    print(json.dumps(inventory), flush=True)
    rows.append(inventory)
    learner.shutdown()
    return {"base": BASE, "width": WIDTH, "rows": rows}


@app.local_entrypoint()
def run() -> None:
    print(json.dumps(exam.remote(), indent=1))
