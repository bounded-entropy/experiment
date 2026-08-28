"""Adapter KINDS on one engine: soft prompts served beside LoRA (#46).

    modal run deploy/adapters_l4.py::parity      # ~5 min, one L4
    modal run deploy/adapters_l4.py::adapters    # ~25 min, one L4

Two questions, one venue (Qwen3-0.6B on a single L4). PARITY asks each kind's
rollout-lowering-against-replay-lowering question with score_tokens: the
engine's own logprob per token under a pinned bundle, against the trainer's
batched replay of the same document under the same params — for lora, for
soft_prompt, and for a bank carrying both, so the soft prompt's numbers are
read against the punica floor rather than against zero. The proof is a SHIFT
TEST rather than a tolerance, the aligned gap against the gap one position of
shift gives, because a miscount of a kind's virtual rows (counted twice, not
at all, or trimmed at the wrong end) moves every position by one and cannot
survive that comparison at any row magnitude. ADAPTERS then submits three GRPO
tenants of different kinds to one Host with staggered joins, sharing one
engine and one multi-tenant learner; success is every tenant finishing with
its logprob_gap at the kernel floor, which is the cross-contamination rail a
wrong prefix or a wrong adapter would blow up.

Deployment only (I5): wiring and measurement, nothing semantics-bearing.
Image pins: keep in sync with deploy/modal_app.py.
"""

from __future__ import annotations

import modal

from probe import CHECKS, arith_tasks, check

app = modal.App("rlstack-adapters")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("probe", "rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-0.6B"
WIDTH = 1024                    # Qwen3-0.6B hidden size: a virtual row's width
N_ROWS = 8
PATTERN = "layers.*.self_attn.*"
RANK = 16





# ---------------------------------------------------------------------------
# parity: the engine's rows against the trainer's rows
# ---------------------------------------------------------------------------

# Two document sets, because the two questions want different material.
#
# PLAIN reads like a rollout: ordinary continuations at ordinary probabilities.
# The GAP is measured here, so its numbers are comparable to the ledger's
# logprob_gap and to #28's 0.022-0.033 kernel floor.
#
# JAGGED puts a predictable token beside a wildly surprising one, so the
# logprob profile swings by nats between neighbours. The SHIFT CONTROL is only
# a proof on documents like these — on a smooth sequence (" 4 5 6 7 8") every
# position looks like its neighbour and an off-by-one hides in the noise. Their
# tail tokens also make absolute gaps large, which is why they are not where
# the floor is read.
PLAIN = [
    ("What is 12+34? The answer is", " 46. That is the sum of the two numbers"),
    ("The capital of France is", " Paris, which is also its largest city"),
    ("Once upon a time there was a", " young girl who lived in a small village"),
]
JAGGED = [
    ("What is 12+34? The answer is", " 46. Bananas orbit the number 46"),
    ("The capital of France is", " Paris. Zebra, quantum, 7, and Paris"),
    ("Once upon a time there was a", " dragon. Spreadsheet! The dragon ate"),
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
                        max_lora_rank=RANK, serves=("lora", "soft_prompt"))
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

    # what a real embedding row weighs — the scale a virtual row is read
    # against, and the reason the sweep below runs at three magnitudes
    table = model.get_input_embeddings().weight
    row_norm = float(table.float().norm(dim=1).mean())
    print(f"[scale] a real {BASE} embedding row has mean L2 norm "
          f"{row_norm:.3f} (std {float(table.float().std()):.4f})")

    # ---- the states ---------------------------------------------------------
    # The magnitudes are the experiment: #45 found the score_tokens gap tracks
    # adapter MAGNITUDE (prefill applies a delta with different kernels than
    # decode), so a single loud delta cannot tell "the lowering is wrong" from
    # "the perturbation is large". Three row scales bracket the default.
    def a_lora(seed: int, scale: float):
        state = lora_torch.build(weighted, {"r": RANK, "seed": seed})
        generator = torch.Generator().manual_seed(seed)
        for path in state.b:
            state.b[path].data = torch.randn(*state.b[path].shape,
                                             generator=generator) * scale
        return state

    def a_prompt(seed: int, std: float):
        return soft_prompt_torch.build(boundary, {"n": N_ROWS, "d": WIDTH,
                                                  "seed": seed,
                                                  "init_std": std})

    delta = a_lora(101, 1.0 / (RANK * 40))          # the regime a run trains in
    prompts = {std: a_prompt(200 + i, std)
               for i, std in enumerate((0.005, 0.02, 0.05))}

    # THE EXACTNESS CONTROL: a soft prompt whose rows ARE the embeddings of n
    # real tokens. Serving it must equal serving those tokens as tokens — on
    # both sides — because that is the whole claim of the lowering: a virtual
    # row is an embedding at a position, nothing more. Any error in the
    # position ids, the padding mask, the mixed-mode is_token_ids mask or the
    # scored-suffix offset breaks THIS test exactly, with no numerics to hide
    # behind.
    PREFIX = " Please answer carefully:"
    prefix_ids = list(engine.tokenize(PREFIX))
    as_tokens = soft_prompt_torch.SoftPromptState(
        n=len(prefix_ids), d=WIDTH, path=boundary[0].path,
        rows=torch.nn.Parameter(
            model.get_input_embeddings().weight[
                torch.tensor(prefix_ids, device=learner.device)]
            .detach().float().clone()))
    print(f"[control] the prefix {PREFIX!r} is {len(prefix_ids)} tokens; its "
          f"embeddings become {len(prefix_ids)} virtual rows")

    lora_torch.install(model, delta)
    for state in prompts.values():
        soft_prompt_torch.install(model, state)
    soft_prompt_torch.install(model, as_tokens)

    lora_slot = {meta.path: delta for meta in weighted}
    default = prompts[0.02]
    banks = {
        "base": ({}, {}),
        "lora": (lora_slot, {"pi": lora_torch.emit(delta)}),
        "both": (lora_slot | {boundary[0].path: default},
                 {"pi": lora_torch.emit(delta),
                  "sp": soft_prompt_torch.emit(default)}),
        "as_tokens": ({boundary[0].path: as_tokens},
                      {"sp": soft_prompt_torch.emit(as_tokens)}),
    }
    for std, state in prompts.items():
        banks[f"soft_prompt@{std}"] = ({boundary[0].path: state},
                                       {"sp": soft_prompt_torch.emit(state)})
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
          set(engine.attachments(bundles["both"].bundle_id))
          == {"lora", "soft_prompt"},
          f"{sorted(engine.attachments(bundles['both'].bundle_id))}")

    # ---- the trainer's side --------------------------------------------------

    def replayed_ids(slot, context_ids: list[int], answer_ids: list[int]) -> tuple:
        """The trainer's logprob for each answer token, from ONE padded forward
        over context+answer — the same call forward_backward makes."""
        ids = list(context_ids) + list(answer_ids)
        batch = TokenBatch(token_ids=tuple(ids), loss_mask=(1,) * len(ids),
                           behavior_logprobs=(0.0,) * len(ids),
                           segment_ids=(0,) * len(ids), doc_starts=(0,))
        plan = ReplayRows(slots=(slot,),
                          index=torch.zeros(1, dtype=torch.long,
                                            device=learner.device))
        with torch.no_grad():
            with row_plan(model).route(plan):
                out = learner._batched_logprobs(batch, _doc_spans(batch))
        return tuple(float(x) for x in out[len(context_ids):])

    def replayed(slot, prompt: str, answer: str) -> tuple:
        return replayed_ids(slot, list(engine.tokenize(prompt)),
                            list(engine.tokenize(answer)))

    async def scored(bundle_id: str, prompt: str, answer: str) -> tuple:
        """One message: score_tokens concatenates each message's own ids, so
        two messages are a way to place an exact token prefix in front."""
        return await engine.score_tokens([Message(role="user", content=prompt)],
                                         engine.tokenize(answer), bundle_id)

    async def scored_with_prefix(prompt: str, answer: str) -> tuple:
        return await engine.score_tokens(
            [Message(role="user", content=PREFIX),
             Message(role="user", content=prompt)],
            engine.tokenize(answer), bundles["base"].bundle_id)

    def gap(a, b) -> float:
        """The ledger's own statistic: the MEAN |trainer - engine| over the
        scored tokens, so these numbers are the same quantity the loop's
        logprob_gap alarm reports and #28's 0.022-0.033 floor calibrated."""
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    def worst(a, b) -> float:
        return max(abs(x - y) for x, y in zip(a, b))

    def shifted_gap(a, b) -> float:
        """The #45 control: the same comparison, one position out. An indexing
        error — rows counted twice, trimmed at the wrong end, a suffix offset
        that forgot them — lands HERE, not in the aligned number."""
        return sum(abs(x - y) for x, y in zip(a[1:], b[:-1])) / (len(a) - 1)

    async def measure() -> dict:
        out = {}
        for name, (slot, _) in banks.items():
            plain, peak = [], []
            for prompt, answer in PLAIN:
                engine_side = await scored(bundles[name].bundle_id, prompt, answer)
                trainer_side = replayed(slot, prompt, answer)
                plain.append(gap(engine_side, trainer_side))
                peak.append(worst(engine_side, trainer_side))
            aligned, shifted = [], []
            for prompt, answer in JAGGED:
                engine_side = await scored(bundles[name].bundle_id, prompt, answer)
                trainer_side = replayed(slot, prompt, answer)
                aligned.append(gap(engine_side, trainer_side))
                shifted.append(shifted_gap(engine_side, trainer_side))
            mean_gap, control = max(plain), min(shifted)
            out[name] = {"gap": mean_gap, "peak": max(peak),
                         "jagged": max(aligned), "shifted": control}
            print(f"\n[{name}] gap {mean_gap:.4f} (worst token {max(peak):.4f})"
                  f"   per doc: " + "  ".join(f"{a:.4f}" for a in plain)
                  + f"\n{' ' * (len(name) + 3)}shift control {control:.3f} vs "
                  f"{max(aligned):.4f} aligned on the jagged set")
            # the alarm the loop already runs on: #28's 0.022-0.033 kernel
            # floor, with the headroom the stress matrix gives an IS loss
            check(f"{name}: the engine and the trainer agree",
                  mean_gap < 0.15, f"gap {mean_gap:.4f} < 0.15")
            check(f"{name}: an off-by-one would have shown",
                  control > 10 * max(aligned), f"shifted {control:.3f} vs "
                  f"aligned {max(aligned):.4f}")
        return out

    async def exactness() -> dict:
        """Each lowering against the tokens its rows ARE.

        `as_tokens` carries the embeddings of a real prefix, so on each side
        the answer is already known: serving those rows must be the same thing
        as serving those tokens. This is the lowering's own proof — no kernel
        difference between the two sides is in play, because both numbers come
        from the SAME engine (or the SAME trainer forward).
        """
        rollout, replay = [], []
        for prompt, answer in PLAIN + JAGGED:
            as_rows = await scored(bundles["as_tokens"].bundle_id, prompt, answer)
            as_text = await scored_with_prefix(prompt, answer)
            rollout.append(worst(as_rows, as_text))

            context = prefix_ids + list(engine.tokenize(prompt))
            answer_ids = list(engine.tokenize(answer))
            with_rows = replayed_ids(banks["as_tokens"][0],
                                     list(engine.tokenize(prompt)), answer_ids)
            with_tokens = replayed_ids({}, context, answer_ids)
            replay.append(worst(with_rows, with_tokens))
        print(f"\n[exact] rollout rows-as-tokens max|d| {max(rollout):.2e}   "
              f"replay rows-as-tokens max|d| {max(replay):.2e}")
        check("ROLLOUT: a row that is a token's embedding serves as that token",
              max(rollout) < 1e-3, f"max|d| {max(rollout):.2e}")
        check("REPLAY: a row that is a token's embedding replays as that token",
              max(replay) < 1e-3, f"max|d| {max(replay):.2e}")
        return {"rollout": max(rollout), "replay": max(replay)}

    async def everything() -> dict:
        """ONE event loop for the whole measurement: the AsyncLLM's output
        handler is a task of the loop that built it, so a second asyncio.run()
        would leave the engine talking to a closed loop (observed: the next
        request dies with EngineDeadError)."""
        measured = await measure()
        measured["exact"] = await exactness()
        return measured

    out = asyncio.run(everything())

    # the cross-side gap is the KERNEL's floor, read against the bare base on
    # the SAME documents — prefill applies a delta with different kernels than
    # decode (#45), so this number is never zero and is not supposed to be
    sweep = {std: out[f"soft_prompt@{std}"]["gap"] for std in prompts}
    print(f"\n[sweep] gap by row magnitude: "
          + "  ".join(f"std {s}: {g:.4f}" for s, g in sorted(sweep.items()))
          + f"   (base floor {out['base']['gap']:.4f}, lora "
          f"{out['lora']['gap']:.4f})")
    out["sweep"] = sweep

    # the rows must MATTER: every agreement above would also hold between two
    # copies of the bare base, so the prompt has to move the numbers by far
    # more than the agreement it is being held to
    moved = max(gap(replayed(banks["soft_prompt@0.02"][0], p, a),
                    replayed({}, p, a)) for p, a in PLAIN)
    check("the virtual rows move the trainer's logprobs",
          moved > 10 * out["base"]["gap"],
          f"soft_prompt vs base {moved:.3f} nats, against a {out['base']['gap']:.4f} "
          f"agreement floor")
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
    # a collapsed tenant still fills a ledger: once every rollout scores 0 the
    # advantages vanish, the gradient is exactly 0 and the policy is frozen at
    # whatever broke it. That is a real failure of the run and must be a check,
    # not a shape the reader has to notice in the numbers.
    grads = [e["train"]["grad_norm"] for e in entries[1:]]
    check(f"{label}: kept training after the first update", any(g > 0 for g in grads),
          f"grad_norm 0 on every update after the first")
    due = [u for u in range(1, n_updates + 1) if u % 4 == 0]
    have = [u for u in due if run.has_eval(u)]
    check(f"{label}: evals present", have == due, f"{have} of {due}")
    return {"max_gap": max(gaps), "gaps": gaps,
            "rewards": (rewards[:2], rewards[-2:])}


@app.function(image=image, gpu="L4", volumes={"/store": store_volume},
              timeout=5400)
def adapters(n_updates: int = 8, seeds: int = 410) -> dict:
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
                        max_loras=8, max_lora_rank=RANK,
                        serves=("lora", "soft_prompt"))
    host = Host("l4-adapters", engines=(engine,), learner=TorchLearner(),
                store=store)

    # `seeds` moves all three identities together: a spec's master seed is
    # inside its run_id, so re-running with the same one ATTACHES to the
    # finished runs (correct, and the resume path proving itself) instead of
    # training. Bump it when the point is to watch three tenants train at once.
    tenants = {
        "lora": make_spec(store, bank_kinds=("lora",), master=seeds + 1,
                          n_updates=n_updates),
        # A SOFT PROMPT WANTS ITS OWN LEARNING RATE, and it is not the LoRA's.
        # AdamW's step is ~lr per coordinate, so one update moves the rows by
        # lr*sqrt(n*d) = lr*90 — against a row block whose whole norm is 1.8 at
        # the default init. At 1e-2 that is half the prompt per step: the first
        # metal run of this file collapsed to reward 0 by update 3 and froze
        # there (zero advantage, zero gradient, and a logprob_gap of 0.25
        # because rows that far outside the embedding distribution are exactly
        # where prefill and decode kernels diverge). 5e-4 is ~4% per step.
        "soft_prompt": make_spec(store, bank_kinds=("soft_prompt",),
                                 master=seeds + 2, n_updates=n_updates, lr=5e-4),
        "both": make_spec(store, bank_kinds=("lora", "soft_prompt"),
                          master=seeds + 3, n_updates=n_updates),
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
        served = engine.residency()
        check("one engine served every kind",
              served["lora"] > 0 and served["soft_prompt"] > 0,
              f"bundles attached per kind on one engine: {served}")
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
