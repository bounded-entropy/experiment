"""FSDP training on real metal: the frozen base across two L4s, fsdp=2.

    modal run deploy/fsdp_l4.py               # attest + shard + train + resume
    modal run deploy/fsdp_l4.py::fsdp_8b      # the same at Qwen3-8B

Four claims, each the next one's precondition. ATTEST: a host born with a
training Regime at width 2 accepts a learner BUILT fsdp=2 and refuses one
built fsdp=1, because capability is a birth fact (#43). SHARD: after the wrap
every base parameter is a DTensor at about half the per-device footprint —
the point of the exercise, measured rather than assumed. TRAIN: a small GRPO
run with the sampler engine and the sharded learner CO-RESIDENT (an FSDP host
never alternates; sleep x FSDP is out of scope, so the partition is dedicated
to training and the engine simply co-resides). RESUME: kill mid-run, re-attach
with a FRESH rank chorus, finish — then load the sealed blobs into an
UNSHARDED TorchLearner and re-emit to the same bytes and the same compiled
bundle id, which is width-independence end to end.

Deployment only (I5): wiring and measurement, nothing semantics-bearing.
Image pins: keep in sync with deploy/modal_app.py.
"""

from __future__ import annotations

import modal

from probe import CHECKS, arith_tasks, check

app = modal.App("rlstack-fsdp")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("probe", "rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-0.6B"
BIG = "Qwen/Qwen3-8B"





def fsdp_spec(store, base: str, *, width: int, n_updates: int, master: int):
    """The spec declares the learner's width and nothing about where it runs:
    `learner(fsdp=width)` is a demand on the metal, attested at submit."""
    from rlstack import (
        AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
        OptimSpec, PolicySpec, SamplingSpec, Schedule, Seeds, TrajectorySource,
        gpus, learner, lora, pool,
    )

    return ExperimentSpec(
        policy=PolicySpec(base=base,
                          bank={"pi": lora("layers.*.self_attn.*", r=16)}),
        gen=GenSpec(env="math_single_turn", tasks=store.cas_put(arith_tasks(64, 0)),
                    sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                          max_tokens=12)),
        trajectories=TrajectorySource("live"),
        algo=AlgoSpec(loss="grpo", post=("verifier", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-4),
                      schedule=Schedule(group_size=4, trajectories_per_wave=8,
                                        n_updates=n_updates,
                                        microbatch_tokens=2048)),
        eval=EvalSpec(tasks=store.cas_put(arith_tasks(16, 1)), every=3,
                      n_samples=2, post=("verifier",)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=width), (pool("main", fraction=0.30),
                                     learner(fsdp=width, fraction=0.40))),)),
        seeds=Seeds(master=master),
    )


# ---------------------------------------------------------------------------
# claim 1: the width is a birth fact, attested at both ends
# ---------------------------------------------------------------------------

def width_is_attested(store, learner, base: str, width: int) -> None:
    """A training regime is a promise about the metal: the host attests it at
    construction, and would rather die there than mid-run."""
    from rlstack import Host
    from rlstack.runner.host import HostError, Regime
    from rlstack.runner.learners.torch_learner import TorchLearner

    regime = Regime(f"learner-fsdp{width}", "training", base, width)
    Host("l4-fsdp-attest", engines=(), learner=learner, store=store,
         regimes=(regime,))
    check(f"a fsdp={width} learner attests against its regime", True,
          f"learner.fsdp={learner.fsdp}")
    try:
        Host("l4-fsdp-attest-wrong", engines=(), learner=TorchLearner(),
             store=store, regimes=(regime,))
    except HostError as refused:
        check("an unsharded learner is REFUSED by that regime", True,
              str(refused)[:80] + "...")
    else:
        check("an unsharded learner is REFUSED by that regime", False,
              "the host accepted metal of the wrong width")


# ---------------------------------------------------------------------------
# claim 2: the base is actually shared out
# ---------------------------------------------------------------------------

def base_is_sharded(learner, width: int) -> None:
    """DTensors and a memory reading: the wrap must show up in both — and the
    tenant's deltas must NOT, because they are meant to stay whole."""
    import torch

    report = learner.shard_report()
    check(f"every base parameter is sharded at fsdp={width}",
          report["sharded"] == report["parameters"],
          f"{report['sharded']}/{report['parameters']}")
    whole, local = report["whole"], report["local"]
    check("this rank holds exactly its share of the base",
          abs(local * width - whole) / whole < 0.02,
          f"rank 0 holds {local/1e6:.1f}M of {whole/1e6:.1f}M base params "
          f"({local/whole:.1%})")
    check("the tenant's deltas stayed whole", report["deltas"] > 0,
          f"{report['deltas']/1e6:.2f}M delta params, replicated per rank")
    print(f"    cuda:0 allocated {torch.cuda.memory_allocated(0)/2**30:.2f} GiB, "
          f"cuda:1 {torch.cuda.memory_allocated(1)/2**30:.2f} GiB")


# ---------------------------------------------------------------------------
# claim 4: the store's bytes do not know the width
# ---------------------------------------------------------------------------

def sealed_bytes_are_width_free(store, run_id: str, spec, schema) -> None:
    """Load the fsdp=2 run's sealed state into an UNSHARDED learner and
    re-emit it. Same bytes, same compiled bundle id — which is precisely what
    a resume on other metal, a warm start, and every add_bundle depend on."""
    from rlstack.policy.compile import compile_bundle
    from rlstack.policy.siteschema import resolve
    from rlstack.registry import ADAPTER_TYPES
    from rlstack.runner.learners.torch_learner import TorchLearner
    from rlstack.spec.validate import site_space

    run = store.open_run(run_id)
    tail = run.ledger_tail()
    versions = {name: int(v) for name, v in tail["versions"].items()}
    names = sorted(spec.policy.bank)
    sealed = {name: run.read_blob("adapters", name, versions[name])
              for name in names}
    moments = {name: run.read_blob("optim", name, versions[name])
               for name in names if spec.policy.bank[name].trainable}

    space = site_space(spec, schema)
    resolved = {name: resolve(space, a.site)
                for name, a in spec.policy.bank.items()}
    plain = TorchLearner()
    plain.install(run_id, spec, resolved)
    plain.load(run_id, sealed, moments)
    again = plain.emit(run_id)

    check("sealed adapter bytes reload into an UNSHARDED learner unchanged",
          all(again.adapters[name] == sealed[name] for name in names),
          f"{[len(sealed[n]) for n in names]} bytes")
    adapter_types = {name: spec.policy.bank[name].adapter_type for name in names}
    servable = [name for name in names
                if ADAPTER_TYPES.get(adapter_types[name]).instance.serving is not None]
    check("and compile to the same bundle id",
          compile_bundle(again.adapters, versions, servable, adapter_types).bundle_id
          == compile_bundle(sealed, versions, servable, adapter_types).bundle_id,
          compile_bundle(sealed, versions, servable, adapter_types).bundle_id)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

async def cancel_after(task, seconds: float) -> None:
    """Kill the run mid-flight; a crash that beat the cancel is REPORTED."""
    import asyncio

    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as failure:
        check("the run failed BEFORE the deliberate cancel", False,
              repr(failure))


def free() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def run_fsdp(base: str, width: int, n_updates: int, kill_after: float,
             gpu_memory_utilization: float) -> dict:
    import asyncio

    import torch
    import transformers
    import vllm

    from rlstack import Host, ModalVolumeStore
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.host import Partition, Regime
    from rlstack.runner.learners.fsdp_torch import lead_fsdp_learner

    print(f"[pins] vllm={vllm.__version__} torch={torch.__version__} "
          f"transformers={transformers.__version__} "
          f"cuda_devices={torch.cuda.device_count()}")
    store = ModalVolumeStore("/store", volume=store_volume,
                             locator="modal://rlstack-store")
    schema = hf_schema(base)
    spec = fsdp_spec(store, base, width=width, n_updates=n_updates, master=57)
    engine = VllmEngine(base, tp=1, gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=512, max_bundles=8, max_rank=16)

    print(f"\n== claim 1: the width is a birth fact ==========================")
    learner = lead_fsdp_learner(width)
    print(f"[chorus] rank 0 of {width}, learner.fsdp={learner.fsdp}")
    try:
        width_is_attested(store, learner, base, width)

        # the FSDP host: born a TRAINING partition over `width` devices; the
        # sampler engine co-resides (concurrent, never alternating)
        host = Host(f"l4-fsdp{width}", engines=(engine,), learner=learner,
                    store=store,
                    partition=Partition("modal-l4", tuple(range(width)),
                                        0.40, "L4"),
                    regimes=(Regime(f"learner-fsdp{width}", "training", base,
                                    width),))

        async def train_and_resume() -> dict:
            nonlocal learner
            print("\n== claims 2+3: shard, then train ==========================")
            task = asyncio.ensure_future(host.submit(spec, schema))
            await asyncio.sleep(kill_after)
            base_is_sharded(learner, width)     # mid-run: the model is built
            print("\n== claim 4a: kill mid-run ================================")
            await cancel_after(task, 0.0)
            # the old chorus goes away COMPLETELY before the new one starts:
            # its children hold a shard of the base on every device, and a
            # resume that overlapped two builds would need twice the metal
            killed, learner = learner, None
            killed.stop()
            del killed, task
            free()

            print("  cancelled; re-attaching behind a FRESH chorus")
            learner = lead_fsdp_learner(width)
            resumed_host = Host(
                f"l4-fsdp{width}", engines=(engine,), learner=learner,
                store=store,
                partition=Partition("modal-l4", tuple(range(width)), 0.40, "L4"),
                regimes=(Regime(f"learner-fsdp{width}", "training", base,
                                width),))
            report = await resumed_host.submit(spec, schema)
            check("re-attached to committed state",
                  report.resumed_from is not None,
                  f"resumed_from={report.resumed_from}")
            return {"run_id": report.run_id,
                    "resumed_from": report.resumed_from}

        out = asyncio.run(train_and_resume())
        store_volume.commit()

        run = store.open_run(out["run_id"])
        entries = run.read_ledger()
        print(f"\nrun_id={out['run_id']}  resumed_from={out['resumed_from']}")
        for entry in entries:
            print(f"  update {entry['update']}: "
                  f"reward {entry['post']['reward']:.3f}  "
                  f"loss {entry['train']['loss']:+.4f}  "
                  f"gap {entry['train']['logprob_gap']:.4f}  "
                  f"grad {entry['train']['grad_norm']:.3f}")
        gaps = [e["train"]["logprob_gap"] for e in entries]
        check("ledger complete",
              [int(e["update"]) for e in entries] == list(range(1, n_updates + 1)),
              f"{len(entries)}/{n_updates}")
        check("logprob_gap bounded under FSDP", bool(gaps) and max(gaps) < 0.15,
              f"max {max(gaps):.4f}" if gaps else "no updates")

        print("\n== claim 4b: the sealed bytes do not know the width =========")
        # the chorus goes down FIRST: what follows must be able to read the
        # store's bytes with no sharded learner anywhere in the process
        learner.stop()
        learner = None
        free()
        sealed_bytes_are_width_free(store, out["run_id"], spec, schema)
        store_volume.commit()
    finally:
        # a run that crashed and a run that finished both leave children
        # holding a shard of the base on every device
        if learner is not None:
            learner.stop()

    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    print(f"\n[fsdp={width} checks] {sum(ok for _, ok, _ in CHECKS)} passed, "
          f"{len(failed)} failed: {failed}")
    return {"run_id": out["run_id"], "resumed_from": out["resumed_from"],
            "passed": sum(ok for _, ok, _ in CHECKS), "failed": failed}


@app.function(image=image, gpu="L4:2", volumes={"/store": store_volume},
              timeout=5400, cpu=8.0, memory=32768)
def fsdp_run(width: int = 2, n_updates: int = 10,
             kill_after: float = 100.0) -> dict:
    return run_fsdp(BASE, width, n_updates, kill_after,
                    gpu_memory_utilization=0.30)


@app.function(image=image, gpu="L4:2", volumes={"/store": store_volume},
              timeout=7200, cpu=8.0, memory=65536)
def fsdp_8b(width: int = 2, n_updates: int = 3,
            kill_after: float = 420.0) -> dict:
    """8B at fsdp=2. WHAT THIS PROBE ESTABLISHED, and where it stops.

    The sharding works: 399/399 base parameters became DTensors, rank 0 held
    exactly 50.0% of 8,190.7M params, 7.63 GiB resident on cuda:0, with the
    tenant's 15.34M delta params whole on every rank.

    The RUN does not fit on 2xL4 in this shape, and the reason is the
    sampler, not the learner: an 8B engine at tp=1 must hold the whole model
    (~15.3 GiB of weights — `gpu_memory_utilization` bounds its budget, not
    its weights), and it wants that on cuda:0 where the learner's 7.63 GiB
    shard already lives. 7.6 + 15.3 > 22.03, so vLLM's engine core dies part
    way through loading (observed: OOM with 13.75 GiB of weights in).

    The fix is not a smaller fraction — it is the fleet's own answer: put the
    sampler on its OWN partition and reach it over the wire
    (deploy/modal_host.py), which is exactly the per-capability host shape
    #43 designed and the OPD 8B<-32B milestone needs. Sharing one 2xL4 host
    between an 8B learner and an 8B sampler would need both sharded AND room
    for the trainer's vocabulary-sized logits; that is a capacity question
    for bigger metal, not a design question.

    Also note the load is WHOLE-THEN-SHARD: TorchLearner puts the entire base
    on the rank's device and fully_shard divides it afterwards, so the peak
    is the whole 16 GiB. shard_the_frozen_base returns the difference to the
    driver; removing the peak itself needs a CPU-side load, in the learner."""
    import os

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["OMP_NUM_THREADS"] = "1"
    return run_fsdp(BIG, width, n_updates, kill_after,
                    gpu_memory_utilization=0.15)


@app.local_entrypoint()
def main() -> None:
    result = fsdp_run.remote()
    print("\n[fsdp run]", result)
    if result["failed"]:
        raise SystemExit(f"FSDP checks FAILED: {result['failed']}")
    print("\nALL FSDP CHECKS PASSED")
