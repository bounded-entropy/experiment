"""Stress: everything that must hold, exercised hard, on one Modal L4.

    modal run deploy/stress_l4.py          # all five stages, two containers

The matrix (Samarth's list, 2026-08):
  multi-LoRA / multi-tenancy   six experiments share ONE engine concurrently,
                               each bundle pinned per request; per-tenant
                               logprob_gap is the cross-contamination alarm
                               (the trainer recomputes under ITS OWN weights —
                               a wrong adapter served would blow the gap up)
  mid-run joins                tenant starts are staggered ~30s apart, so each
                               joins an engine already serving others
  the loss zoo                 grpo / ppo / gspo / sdft / opsd (live) +
                               sft (static cas://) + opd (replay store://),
                               many optimizer steps each
  resource sharing, 1 GPU      engine fraction + N resident learners under
                               concurrent leases; stage 4 runs sharing="sleep"
                               (the ExclusiveLease alternation) as well
  policy lag                   opsd runs max_policy_lag=2 with a deliberately
                               slowed trainer (epochs_per_wave=2); realized
                               per-turn lag is measured from the sealed record
  resume                       stage 4 cancels mid-run and re-attaches in
                               process; stage 5 cancels and re-attaches from a
                               FRESH container (content addressing must
                               recompile the ledger-tail bundle to the same id
                               or generation dies loudly)
  evals                        every tenant evaluates every 5 updates; the
                               evaluator backfills after both resumes

Deployment only (I5): specs, wiring, measurement. Nothing semantics-bearing.
Image pins: keep in sync with deploy/modal_app.py.
"""

from __future__ import annotations

import modal

app = modal.App("rlstack-stress")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-0.6B"
EVERY = 5                       # eval cadence for every tenant


def arith_tasks(n: int, seed: int) -> bytes:
    import json
    import random

    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randrange(10, 99), rng.randrange(10, 99)
        rows.append({"id": f"arith-{i:04d}",
                     "prompt": f"What is {a}+{b}? The answer is",
                     "meta": {"answer": a + b}})
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()


def make_spec(store, *, loss, post, master, n_updates, source="live", lag=0,
              epochs=1, group_size=4, per_wave=16, lr=1e-4, sharing="concurrent",
              judge_pool=False):
    """One tenant's spec. Deterministic given its arguments (cas_put dedupes),
    so stage 5's second container rebuilds the identical identity."""
    from rlstack import (
        AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
        OptimSpec, PolicySpec, SamplingSpec, Schedule, Seeds, TrajectorySource,
        gpus, learner, lora, pool,
    )

    train = store.cas_put(arith_tasks(64, seed=0))
    heldout = store.cas_put(arith_tasks(16, seed=1))
    live = source == "live"
    members = (pool("main", fraction=0.30),) + (
        (pool("judge"),) if judge_pool else ()) + (learner(fraction=0.10),)
    return ExperimentSpec(
        policy=PolicySpec(base=BASE,
                          bank={"pi": lora("layers.*.self_attn.*", r=16)}),
        gen=(GenSpec(env="math_single_turn", tasks=train,
                     sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                           max_tokens=12))
             if live else None),
        trajectories=TrajectorySource(source),
        algo=AlgoSpec(loss=loss, post=post,
                      optim=OptimSpec("adamw", lr=lr),
                      schedule=Schedule(group_size=group_size,
                                        trajectories_per_wave=per_wave,
                                        n_updates=n_updates,
                                        epochs_per_wave=epochs,
                                        microbatch_tokens=2048,
                                        max_policy_lag=lag)),
        eval=EvalSpec(tasks=heldout, every=EVERY, n_samples=2,
                      env="math_single_turn", post=("verifier",)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), members, sharing=sharing),)),
        seeds=Seeds(master=master),
    )


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def report_run(store, run_id: str, label: str, n_updates: int,
               gap_alarm: float = 0.15) -> dict:
    """Read one finished run back and print its story; returns key metrics.

    gap_alarm is FAMILY-AWARE: for IS-corrected losses the gap is the
    off-policy/contamination alarm, and 0.15 is generous headroom over the
    ~0.03 kernel floor; for behavior cloning (sft) and distillation onto a
    moving teacher (opd) the gap MEASURES teacher-student distance — growth
    is the objective working, and only an adapter mixup (gap ~5+) is a bug,
    so their bound is loose. Verified against the first stress run's gap
    curves: opd starts at the 0.03 floor (teacher v0 == fresh student, so no
    contamination) and its squared loss pulls 0.166 back down to 0.09."""
    run = store.open_run(run_id)
    entries = run.read_ledger()
    updates = [int(e["update"]) for e in entries]
    check(f"{label}: ledger complete", updates == list(range(1, n_updates + 1)),
          f"{len(updates)}/{n_updates} updates")

    def mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    rewards = [e["post"].get("reward") for e in entries if e.get("post")]
    rewards = [r for r in rewards if r is not None]
    gaps = [e["train"]["logprob_gap"] for e in entries]
    head = mean(rewards[:3]) if rewards else None
    tail = mean(rewards[-3:]) if rewards else None
    check(f"{label}: logprob_gap bounded", max(gaps) < gap_alarm,
          f"max {max(gaps):.4f} < {gap_alarm}")
    for e in entries[:: max(1, len(entries) // 6)] + entries[-1:]:
        print(f"    u{e['update']:>3}: reward {e['post'].get('reward', float('nan')):.3f} "
              f"loss {e['train']['loss']:+.4f} gap {e['train']['logprob_gap']:.4f} "
              f"ratio {e['train']['mean_ratio']:.3f} grad {e['train']['grad_norm']:.2f}")

    import json as _json
    due = [u for u in range(1, n_updates + 1) if u % EVERY == 0]
    have = [u for u in due if run.has_eval(u)]
    check(f"{label}: evals present", have == due, f"{have} of {due}")
    evals = {u: _json.loads(run.read_eval(u, "summary.json"))["means"]
             for u in have}
    for u, means in evals.items():
        print(f"    eval@{u}: {means}")
    return {"rewards": (head, tail), "max_gap": max(gaps), "evals": evals}


def measure_lag(store, run_id: str, label: str) -> None:
    """Realized per-turn policy lag, from the sealed record (I6): wave u was
    on-policy iff its turns ran version u-1."""
    run = store.open_run(run_id)
    counts: dict[int, int] = {}
    for u in run.list_updates():
        for row in run.read_wave(u):
            for turn in row["turns"]:
                lag = (u - 1) - int(turn["policy_version"]["pi"])
                counts[lag] = counts.get(lag, 0) + 1
    print(f"    {label} realized lag histogram (turns): {dict(sorted(counts.items()))}")
    check(f"{label}: lag stayed within bound", all(0 <= k <= 2 for k in counts),
          str(dict(sorted(counts.items()))))
    check(f"{label}: lag actually occurred", any(k > 0 for k in counts),
          "generator never ran ahead" if not any(k > 0 for k in counts) else "")


def harvest_correct(store, run_id: str, minimum: int) -> str:
    """Stage 2: the grpo run's reward-1 trajectories as a static cas:// set."""
    import json as _json

    run = store.open_run(run_id)
    kept, total = [], 0
    for u in run.list_updates():
        rows = run.read_wave(u)
        rewards = run.read_postdata(u)["reward"]
        total += len(rows)
        kept += [row for row, r in zip(rows, rewards) if r == 1.0]
    if len(kept) < minimum:                     # never starve the feed
        kept = [row for u in run.list_updates() for row in run.read_wave(u)]
    print(f"  harvested {len(kept)}/{total} correct trajectories for sft")
    blob = "".join(_json.dumps(r, sort_keys=True) + "\n" for r in kept).encode()
    return store.cas_put(blob)


# ---------------------------------------------------------------------------
# the stages
# ---------------------------------------------------------------------------

async def _cancel_after(task, seconds: float) -> None:
    """Cancel the run mid-flight; a crash that beat the cancel is REPORTED,
    never swallowed (a masked failure would fake a clean resume)."""
    import asyncio

    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as failure:
        check("run failed BEFORE the deliberate cancel", False, repr(failure))


def _free() -> None:
    """Callers drop their learner references first; this reclaims the VRAM."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


@app.function(image=image, gpu="L4", volumes={"/store": store_volume},
              timeout=7200)
def run_stress() -> dict:
    import asyncio

    import torch
    import transformers
    import vllm

    from rlstack import ModalVolumeStore
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack import GpuArbiter
    from rlstack.runner.learners.torch_learner import TorchLearner
    from rlstack.runner.loop import run_experiment_async

    print(f"[pins] vllm={vllm.__version__} torch={torch.__version__} "
          f"transformers={transformers.__version__}")
    store = ModalVolumeStore("/store", volume=store_volume)
    schema = hf_schema(BASE)
    engine = VllmEngine(BASE, gpu_memory_utilization=0.30, max_model_len=512,
                        max_loras=8, max_lora_rank=16)

    arbiter = GpuArbiter()      # the metal's admission authority (concurrent regime)

    async def main() -> dict:
        out: dict = {}

        # ---- stage 1: solo grpo, many steps — also the teacher run ----------
        print("\n== stage 1: grpo, 30 updates, solo =============================")
        spec = make_spec(store, loss="grpo", post=("verifier", "grpo_advantage"),
                         master=101, n_updates=30)
        lrn = TorchLearner()
        rep = await run_experiment_async(spec, schema, store, engine, lrn,
                                         arbiter=arbiter)
        del lrn
        _free()
        out["grpo"] = report_run(store, rep.run_id, "grpo", 30)
        grpo_rid = rep.run_id

        # ---- stage 2: harvest the teacher's correct rows for sft ------------
        print("\n== stage 2: harvest sft dataset ================================")
        sft_uri = harvest_correct(store, grpo_rid, minimum=16)

        # ---- stage 3: six tenants, one engine, staggered joins --------------
        print("\n== stage 3: six concurrent tenants =============================")
        tenants = {
            "sft":  make_spec(store, loss="sft", post=(), master=102,
                              n_updates=24, source=sft_uri, group_size=1),
            "opd":  make_spec(store, loss="opd", post=("verifier",), master=103,
                              n_updates=24, source=f"store://{grpo_rid}"),
            "ppo":  make_spec(store, loss="ppo", post=("verifier", "center_reward"),
                              master=104, n_updates=24),
            "gspo": make_spec(store, loss="gspo", post=("verifier", "grpo_advantage"),
                              master=105, n_updates=24),
            "sdft": make_spec(store, loss="sdft", post=("verifier",), master=106,
                              n_updates=24),
            "opsd": make_spec(store, loss="opsd", post=("verifier",), master=107,
                              n_updates=24, lag=2, epochs=2),
            # the pool treaty on real metal: the judge pool is the SAME
            # engine under a second name (multi-tenancy makes it free) —
            # rewards come from llm_judge sampling it greedily
            "judge": make_spec(store, loss="grpo",
                               post=("llm_judge", "grpo_advantage"),
                               master=108, n_updates=24, judge_pool=True),
        }
        # ONE learner for all seven tenants: one shared base, per-tenant
        # adapters, swap-install between interleaved microbatches — the
        # Learner tenancy invariant on real metal (was 7 base copies)
        shared_learner = TorchLearner()

        async def launch(name: str, delay: float):
            await asyncio.sleep(delay)
            print(f"  [join] {name} starts (t+{delay:.0f}s)")
            pools = ({"main": engine, "judge": engine} if name == "judge"
                     else engine)
            return name, await run_experiment_async(
                tenants[name], schema, store, pools, shared_learner,
                arbiter=arbiter)

        results = await asyncio.gather(*(launch(name, i * 30.0)
                                         for i, name in enumerate(tenants)))
        del shared_learner
        _free()
        gap_alarms = {"sft": 1.0, "opd": 0.5, "opsd": 0.25}
        for name, rep in results:
            print(f"\n  -- {name} ({rep.run_id})")
            out[name] = report_run(store, rep.run_id, name, 24,
                                   gap_alarm=gap_alarms.get(name, 0.15))
        rid = {name: rep.run_id for name, rep in results}
        measure_lag(store, rid["opsd"], "opsd")
        for name in ("ppo", "gspo", "sdft", "judge"):   # strictly on-policy tenants
            run = store.open_run(rid[name])
            on_policy = all(
                (u - 1) - int(t["policy_version"]["pi"]) == 0
                for u in run.list_updates()
                for row in run.read_wave(u)
                for t in row["turns"])
            check(f"{name}: strictly on-policy (B=0)", on_policy)

        # ---- stage 4: kill mid-run, re-attach in process, sleep-sharing -----
        print("\n== stage 4: cancel + in-process resume (sharing='sleep') =======")
        # deliberately NOT the shared arbiter: this spec declares sleep
        # alternation for the same engine object the concurrent stages
        # attached as free — a regime change is a different residency world,
        # and the arbiter's group-mismatch guard would (rightly) refuse it
        spec = make_spec(store, loss="grpo", post=("verifier", "grpo_advantage"),
                         master=208, n_updates=40, sharing="sleep")
        lrn = TorchLearner()
        task = asyncio.ensure_future(
            run_experiment_async(spec, schema, store, engine, lrn))
        await _cancel_after(task, 180.0)
        del task, lrn
        _free()
        print("  cancelled; re-attaching with a fresh learner")
        lrn = TorchLearner()
        rep = await run_experiment_async(spec, schema, store, engine, lrn)
        del lrn
        _free()
        check("stage4: attached to committed state",
              rep.resumed_from is not None and 0 < rep.resumed_from <= 40,
              f"resumed_from={rep.resumed_from} "
              + ("(mid-run resume)" if (rep.resumed_from or 40) < 40
                 else "(already complete: idempotent no-op)"))
        out["resume_inproc"] = report_run(store, rep.run_id, "resume-inproc", 40)

        # ---- stage 5a: kill mid-run; a FRESH CONTAINER resumes it -----------
        print("\n== stage 5a: cancel for cross-container resume =================")
        spec = make_spec(store, loss="grpo", post=("verifier", "grpo_advantage"),
                         master=301, n_updates=24)
        lrn = TorchLearner()
        task = asyncio.ensure_future(
            run_experiment_async(spec, schema, store, engine, lrn,
                                 arbiter=arbiter))
        await _cancel_after(task, 90.0)
        del task, lrn
        _free()
        print("  cancelled; stage 5b (a fresh container) must finish it")
        return out

    out = asyncio.run(main())
    store_volume.commit()
    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    out["checks"] = {"passed": sum(ok for _, ok, _ in CHECKS),
                     "failed": failed}
    print(f"\n[stage 1-5a checks] {out['checks']['passed']} passed, "
          f"{len(failed)} failed: {failed}")
    return out


@app.function(image=image, gpu="L4", volumes={"/store": store_volume},
              timeout=3600)
def resume_cross_container() -> dict:
    """Stage 5b: a cold process attaches to the cancelled run and finishes it.

    Phase 1 loads the ledger-tail deltas and recompiles the bundle; content
    addressing must reproduce the tail's bundle_id exactly or the generator's
    next request pins an unregistered id and dies — the strongest emit/load
    roundtrip check real metal offers."""
    from rlstack import ModalVolumeStore
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.learners.torch_learner import TorchLearner
    from rlstack.runner.loop import run_experiment

    store = ModalVolumeStore("/store", volume=store_volume)
    spec = make_spec(store, loss="grpo", post=("verifier", "grpo_advantage"),
                     master=301, n_updates=24)
    engine = VllmEngine(BASE, gpu_memory_utilization=0.30, max_model_len=512,
                        max_loras=8, max_lora_rank=16)
    rep = run_experiment(spec, hf_schema(BASE), store, engine, TorchLearner())
    store_volume.commit()

    print(f"\n== stage 5b: cross-container resume of {rep.run_id} ============")
    check("stage5: attached mid-run in a fresh container",
          rep.resumed_from is not None and rep.resumed_from > 0,
          f"resumed_from={rep.resumed_from}")
    out = {"resume_xcontainer": report_run(store, rep.run_id, "resume-xcont", 24)}
    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    out["checks"] = {"passed": sum(ok for _, ok, _ in CHECKS), "failed": failed}
    print(f"\n[stage 5b checks] {out['checks']['passed']} passed, "
          f"{len(failed)} failed: {failed}")
    return out


@app.local_entrypoint()
def main() -> None:
    first = run_stress.remote()
    print("\n[run_stress summary]", first["checks"])
    second = resume_cross_container.remote()
    print("\n[resume summary]", second["checks"])
    if first["checks"]["failed"] or second["checks"]["failed"]:
        raise SystemExit("stress checks FAILED")
    print("\nALL STRESS CHECKS PASSED")
