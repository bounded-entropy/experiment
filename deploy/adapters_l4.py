"""Adapter KINDS on one engine: soft prompts served beside LoRA (#46).

    modal run deploy/adapters_l4.py::parity      # ~5 min, one L4
    modal run deploy/adapters_l4.py::adapters    # ~25 min, one L4

Two questions, one venue (Qwen3-0.6B on a single L4).

PARITY asks the mandatory kind question (#3): do the rows the ENGINE serves
and the rows the TRAINER replays produce the same numbers? The instrument is
score_tokens — the engine's own logprob for each token of a document under a
pinned bundle — against the trainer's batched replay of the same document under
the same params. As in #45 the proof is a SHIFT TEST, not a tolerance: the
aligned gap against the gap a one-position shift gives. An off-by-one in the
virtual rows (rows counted twice, or not at all, or trimmed at the wrong end)
moves every position by one and cannot survive that comparison at any row
magnitude. The same measurement runs for lora, for soft_prompt, and for a bank
carrying BOTH, so the soft prompt's numbers are read against the punica floor
this stack has been calibrated at since #28 rather than against zero.

ADAPTERS asks Samarth's question: can GRPO experiments with DIFFERENT adapter
kinds run concurrently on ONE VllmEngine? Three tenants — lora, soft_prompt,
and a bank with both — submitted to one Host with staggered joins, sharing one
engine and one multi-tenant learner. Success is every tenant completing with
its logprob_gap at the kernel floor: the gap is the cross-contamination alarm,
and a request served the wrong prefix (or the wrong adapter) blows it up.

Deployment only (I5): wiring and measurement, nothing semantics-bearing.
Image pins: keep in sync with deploy/modal_app.py.
"""

from __future__ import annotations

import modal

app = modal.App("rlstack-adapters")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-0.6B"
WIDTH = 1024                    # Qwen3-0.6B hidden size: a virtual row's width
N_ROWS = 8
PATTERN = "layers.*.self_attn.*"
RANK = 16

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


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


# ---------------------------------------------------------------------------
# parity: the engine's rows against the trainer's rows
# ---------------------------------------------------------------------------

DOCS = [
    ("What is 12+34? The answer is", " 46. Let me check that again"),
    ("The capital of France is", " Paris, and the capital of Italy is"),
    ("Count with me: 1 2 3", " 4 5 6 7 8 9"),
]


@app.function(image=image, gpu="L4", timeout=2400)
def parity() -> dict:
    import asyncio

    import torch
    import transformers
    import vllm

    from rlstack.data.flatten import TokenBatch
    from rlstack.data.trajectory import Message
    from rlstack.policy.adapters import lora_torch, soft_prompt_torch
    from rlstack.policy.adapters.replay import ReplayRows, row_plan
    from rlstack.policy.compile import compile_bundle
    from rlstack.policy.siteschema import hf_schema, resolve
    from rlstack.registry import ADAPTERS
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.learners.torch_learner import TorchLearner, _doc_spans
    from rlstack.spec.specs import soft_prompt as soft_prompt_spec

    print(f"[pins] vllm={vllm.__version__} torch={torch.__version__} "
          f"transformers={transformers.__version__}")

    engine = VllmEngine(BASE, gpu_memory_utilization=0.45, max_model_len=512,
                        max_lora_rank=RANK, prompt_embeds=True)
    learner = TorchLearner()
    learner._ensure_base(BASE)
    model = learner._model

    schema = hf_schema(BASE)
    weighted = resolve(schema.sites, PATTERN)
    boundary = ADAPTERS.get("soft_prompt").instance.exports(
        soft_prompt_spec(f"prompt[:{N_ROWS}]", n=N_ROWS, d=WIDTH))[:1]
    print(f"[sites] {len(weighted)} weighted by {PATTERN!r}; boundary "
          f"{boundary[0].name!r} at {boundary[0].path!r}")

    # reachability is a build fact: this engine was ASKED for prompt_embeds
    reach = engine.reachability(schema.sites + boundary)
    check("the build reaches the embedding boundary via prompt_embeds",
          str(reach[boundary[0].name]) == "prompt_embeds",
          f"{reach[boundary[0].name]}")
    check("the build still reports punica on weighted sites",
          str(reach[weighted[0].name]) == "punica")
    check("side_attention is honestly unreachable",
          str(reach["final_hidden"]) == "none")

    # ---- the states, with deltas big enough to bite -------------------------
    delta = lora_torch.build(weighted, {"r": RANK, "seed": 101})
    generator = torch.Generator().manual_seed(101)
    for path in delta.b:
        delta.b[path].data = torch.randn(*delta.b[path].shape,
                                         generator=generator) / (RANK * 8)
    rows = soft_prompt_torch.build(boundary, {"n": N_ROWS, "d": WIDTH,
                                              "seed": 202, "init_std": 0.05})
    lora_torch.install(model, delta)
    soft_prompt_torch.install(model, rows)

    lora_slot = {meta.path: delta for meta in weighted}
    rows_slot = {boundary[0].path: rows}
    banks = {
        "base": ({}, {}),
        "lora": (lora_slot, {"pi": lora_torch.emit(delta)}),
        "soft_prompt": (rows_slot, {"sp": soft_prompt_torch.emit(rows)}),
        "both": (lora_slot | rows_slot,
                 {"pi": lora_torch.emit(delta), "sp": soft_prompt_torch.emit(rows)}),
    }
    kinds = {"pi": "lora", "sp": "soft_prompt"}

    bundles = {}
    for name, (_, payloads) in banks.items():
        bundle = compile_bundle(payloads, {n: 0 for n in payloads},
                                servable=payloads, kinds=kinds)
        engine.add_bundle(bundle)
        bundles[name] = bundle
        print(f"[bundle] {name}: {bundle.bundle_id} "
              f"payloads={sorted(bundle.payloads)}")
    check("a bank of two kinds registers with BOTH consumers",
          bundles["both"].bundle_id in engine._lora
          and bundles["both"].bundle_id in engine._rows)

    # ---- the trainer's side --------------------------------------------------

    def replayed(slot, prompt: str, answer: str) -> tuple:
        """The trainer's logprob for each answer token, from ONE padded
        forward over prompt+answer — the same call forward_backward makes."""
        prompt_ids = list(engine.tokenize(prompt))
        answer_ids = list(engine.tokenize(answer))
        ids = prompt_ids + answer_ids
        batch = TokenBatch(token_ids=tuple(ids), loss_mask=(1,) * len(ids),
                           behavior_logprobs=(0.0,) * len(ids),
                           segment_ids=(0,) * len(ids), doc_starts=(0,))
        plan = ReplayRows(slots=(slot,),
                          index=torch.zeros(1, dtype=torch.long,
                                            device=learner.device))
        with torch.no_grad():
            with row_plan(model).route(plan):
                out = learner._batched_logprobs(batch, _doc_spans(batch))
        return tuple(float(x) for x in out[len(prompt_ids):])

    async def scored(bundle_id: str, prompt: str, answer: str) -> tuple:
        return await engine.score_tokens([Message(role="user", content=prompt)],
                                         engine.tokenize(answer), bundle_id)

    def gap(a, b) -> float:
        return max(abs(x - y) for x, y in zip(a, b))

    def shifted_gap(a, b) -> float:
        """The #45 control: the same comparison, one position out. An indexing
        error lands HERE, not in the aligned number."""
        return max(abs(x - y) for x, y in zip(a[1:], b[:-1]))

    async def measure() -> dict:
        out = {}
        for name, (slot, _) in banks.items():
            aligned, shifted = [], []
            for prompt, answer in DOCS:
                engine_side = await scored(bundles[name].bundle_id, prompt, answer)
                trainer_side = replayed(slot, prompt, answer)
                aligned.append(gap(engine_side, trainer_side))
                shifted.append(shifted_gap(engine_side, trainer_side))
            worst, control = max(aligned), min(shifted)
            out[name] = {"aligned": worst, "shifted": control}
            print(f"\n[{name}] aligned max {worst:.4f}   shifted min {control:.4f}")
            check(f"{name}: the engine and the trainer agree",
                  worst < 0.10, f"max|d| {worst:.4f} < 0.10")
            check(f"{name}: an off-by-one would have shown",
                  control > 5 * max(worst, 1e-6),
                  f"shifted {control:.4f} vs aligned {worst:.4f}")
        return out

    out = asyncio.run(measure())

    # the rows must MATTER: a parity that compares two copies of the base
    # would pass everything above
    moved = max(gap(replayed(banks["soft_prompt"][0], p, a),
                    replayed({}, p, a)) for p, a in DOCS)
    check("the virtual rows move the trainer's logprobs", moved > 1.0,
          f"soft_prompt vs base max|d| {moved:.3f} nats")
    out["rows_move_logprobs"] = moved

    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    out["checks"] = {"passed": sum(ok for _, ok, _ in CHECKS), "failed": failed}
    print(f"\n[parity] {out['checks']['passed']} passed, {len(failed)} failed: "
          f"{failed}")
    return out


# ---------------------------------------------------------------------------
# the acceptance test: three kinds, one engine
# ---------------------------------------------------------------------------

def make_spec(store, *, bank_kinds: tuple[str, ...], master: int,
              n_updates: int, lr: float = 1e-4):
    """One tenant's spec: GRPO on arithmetic, with the named kinds in its bank.

    Everything except the BANK is identical across tenants, so a difference in
    the ledger is a difference in the adapter kind and nothing else.
    """
    from rlstack import (
        AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
        OptimSpec, PolicySpec, SamplingSpec, Schedule, Seeds, TrajectorySource,
        gpus, learner, lora, pool, soft_prompt,
    )

    bank = {}
    if "lora" in bank_kinds:
        bank["pi"] = lora(PATTERN, r=RANK)
    if "soft_prompt" in bank_kinds:
        bank["sp"] = soft_prompt(f"prompt[:{N_ROWS}]", n=N_ROWS, d=WIDTH)

    train = store.cas_put(arith_tasks(64, seed=0))
    heldout = store.cas_put(arith_tasks(16, seed=1))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank=bank),
        gen=GenSpec(env="math_single_turn", tasks=train,
                    sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                          max_tokens=12)),
        trajectories=TrajectorySource("live"),
        algo=AlgoSpec(loss="grpo", post=("verifier", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=lr),
                      schedule=Schedule(group_size=4, trajectories_per_wave=16,
                                        n_updates=n_updates,
                                        microbatch_tokens=2048)),
        eval=EvalSpec(tasks=heldout, every=4, n_samples=2,
                      env="math_single_turn", post=("verifier",)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main", fraction=0.40),
                                 learner(fraction=0.15))),)),
        seeds=Seeds(master=master),
    )


def report_run(store, run_id: str, label: str, n_updates: int,
               gap_alarm: float = 0.15) -> dict:
    """One tenant's story, read back off the store."""
    run = store.open_run(run_id)
    entries = run.read_ledger()
    updates = [int(e["update"]) for e in entries]
    check(f"{label}: ledger complete", updates == list(range(1, n_updates + 1)),
          f"{len(updates)}/{n_updates} updates")
    gaps = [e["train"]["logprob_gap"] for e in entries]
    check(f"{label}: logprob_gap at the kernel floor", max(gaps) < gap_alarm,
          f"max {max(gaps):.4f} < {gap_alarm}")
    rewards = [e["post"]["reward"] for e in entries if e.get("post")]
    for e in entries:
        print(f"    u{e['update']:>3}: reward {e['post'].get('reward', float('nan')):.3f} "
              f"loss {e['train']['loss']:+.4f} gap {e['train']['logprob_gap']:.4f} "
              f"grad {e['train']['grad_norm']:.2f}")
    due = [u for u in range(1, n_updates + 1) if u % 4 == 0]
    have = [u for u in due if run.has_eval(u)]
    check(f"{label}: evals present", have == due, f"{have} of {due}")
    return {"max_gap": max(gaps), "gaps": gaps,
            "rewards": (rewards[:2], rewards[-2:])}


@app.function(image=image, gpu="L4", volumes={"/store": store_volume},
              timeout=5400)
def adapters(n_updates: int = 8) -> dict:
    import asyncio

    import torch
    import transformers
    import vllm

    from rlstack import Host, ModalVolumeStore
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.learners.torch_learner import TorchLearner

    print(f"[pins] vllm={vllm.__version__} torch={torch.__version__} "
          f"transformers={transformers.__version__}")
    store = ModalVolumeStore("/store", volume=store_volume,
                             locator="modal://rlstack-store")
    schema = hf_schema(BASE)

    # ONE engine, built with both levers; ONE multi-tenant learner
    engine = VllmEngine(BASE, gpu_memory_utilization=0.40, max_model_len=512,
                        max_loras=8, max_lora_rank=RANK, prompt_embeds=True)
    host = Host("l4-adapters", engines=(engine,), learner=TorchLearner(),
                store=store)

    tenants = {
        "lora": make_spec(store, bank_kinds=("lora",), master=401,
                          n_updates=n_updates),
        "soft_prompt": make_spec(store, bank_kinds=("soft_prompt",), master=402,
                                 n_updates=n_updates, lr=1e-2),
        "both": make_spec(store, bank_kinds=("lora", "soft_prompt"), master=403,
                          n_updates=n_updates),
    }

    async def main() -> dict:
        stats = asyncio.get_running_loop().create_task(host.run_stats(30.0))

        async def launch(name: str, delay: float):
            await asyncio.sleep(delay)
            print(f"  [join] {name} starts (t+{delay:.0f}s)")
            return name, await host.submit(tenants[name], schema)

        results = await asyncio.gather(*(launch(name, i * 25.0)
                                         for i, name in enumerate(tenants)))
        stats.cancel()
        out = {}
        for name, rep in results:
            print(f"\n  -- {name} ({rep.run_id})")
            out[name] = report_run(store, rep.run_id, name, n_updates)
        print("\n[host status]", host.status())
        check("one engine served every kind",
              len(engine._rows) > 0 and len(engine._lora) > 0,
              f"{len(engine._lora)} punica bundles, {len(engine._rows)} "
              f"prompt-row bundles on one engine")
        return out

    out = asyncio.run(main())
    store_volume.commit()
    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    out["checks"] = {"passed": sum(ok for _, ok, _ in CHECKS), "failed": failed}
    print(f"\n[adapters] {out['checks']['passed']} passed, {len(failed)} "
          f"failed: {failed}")
    return out


@app.local_entrypoint()
def main() -> None:
    first = parity.remote()
    print("\n[parity summary]", first["checks"])
    second = adapters.remote()
    print("\n[adapters summary]", second["checks"])
    if first["checks"]["failed"] or second["checks"]["failed"]:
        raise SystemExit("adapter checks FAILED")
    print("\nALL ADAPTER CHECKS PASSED")
