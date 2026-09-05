"""Steering vectors on ONE L4, through the desk: ADR 0004's check.

    modal run deploy/steer_l4.py::run_tests        # the fakes suite in the image (torch-gated cases run)
    modal run deploy/steer_l4.py::probe            # promises 1-2: parity, one container, no desk
    modal deploy deploy/desk.py                    # THE desk, once, for every venue
    modal deploy deploy/steer_l4.py                # this venue's metal
    modal run deploy/steer_l4.py::check            # promises 3-5: up, two tenants, release, MY metal freed
    modal run deploy/steer_l4.py::knock            # the door back: a placement re-acquires released metal
    modal run deploy/desk.py::status / ::sweep     # the fleet, at the desk

    RLSTACK_STEER_GPU=L4:2 RLSTACK_STEER_TP=2 modal deploy deploy/steer_l4.py   # the TP condition

THREE THINGS THIS FILE EXISTS TO SHOW. First the residual lever itself: a
steer served by the engine image's hook and replayed by the site wrapper
agree — zero is the base bit for bit on both sides, a vector's window lands
where it was recorded, the prefix cache never aliases across bundles or
windows. Second I8 with a third mechanism: two tenants — lora-only and
steer-only — submitted THROUGH THE DESK, joined onto one serving host and
one learner on one card, trained side by side. Third ADR 0003 on the venue,
for the first time: the metal is handed back by the desk's `release`, the
keepalive input returns because the desk said so, and every entrypoint that
acquires metal ends by ASSERTING ITS OWN METAL is released — its own, not the
whole plane, because since ADR 0007 one desk serves every venue and the plane
may legitimately hold someone else's card. Those releases are GUARDED: this
door hands back what it acquired and never tears down a run that is not
this check's (ADR 0007, Q6).

Everything semantics-bearing is in the two specs — the banks, the loss, the
plans — and everything else here is the chassis' (I5, ADR 0007): the
transports, the desk container and the metal container's bring-up all left
this file for `deploy/modal_venue.py` and `deploy/desk.py`, and every spec
value below is byte-identical to what it was before they did
(tests/test_venues.py).
"""

from __future__ import annotations

import json
import os
import time

import modal

from modal_venue import (
    a_store, cpu_image_for, desk, fleet, gpu_image_for, hf_cache, metal_class,
    metal_handle, progress_function, run_suite, store_volume, submit_spec,
    take_down, wait_for_metal,
)

APP = "rlstack-steer-l4"
app = modal.App(APP)

# The condition, as one flag: the card the metal is deployed on and the width
# the engine is built at. Read at deploy time on the client and inside the
# container (the image carries it), so a redeploy is the whole change.
GPU = os.environ.get("RLSTACK_STEER_GPU", "L4")
TP = int(os.environ.get("RLSTACK_STEER_TP", "1"))

cpu_image = cpu_image_for()
gpu_image = gpu_image_for({"RLSTACK_STEER_TP": str(TP)}, with_tests=True)

BASE = "Qwen/Qwen3-0.6B"
HIDDEN = 1024                   # Qwen3-0.6B's width — a boundary has no shape, the spec states it
# the task sets built by deploy/tasks_dapo.py (#60), reused verbatim
TRAIN_TASKS = "cas://09499d32b51e5e1b2a644b1c65e01b44aa42ff1a5bfac78ead41f98f89f09c93"
EVAL_TASKS = "cas://82ae4626dbb59a2c50e2b13cbe7250c5f1ddd02dfb81edc7495efb77759d420b"

STEER_SITE = "resid_pre.8-20"   # thirteen boundaries, one vector each
LORA_SITE = "layers.0-27.self_attn.*"
RANK = 16
STEER_LR = 1e-3                 # a vector wants its own LR (the soft prompt's lesson, #46)

# THE one problem both tenants train on — screened by deploy/plora_l4.py: the
# base passes it 4/8 at temperature 1.0, so its groups split and the
# advantages are non-zero. An unscreened draw of the task file gave two
# updates of all-or-nothing groups and a loss of exactly 0.0 on the first
# clean check (rails fine, gradient never exercised).
SCREENED_TASK = "dapo-math-17k/a6d38312-86c7-4022-b8d2-adcf19fa0c3a"
UPDATES = 2
GROUPS_PER_WAVE = 2
GROUP_SIZE = 8
MAX_TOKENS = 1024

# The partition treaty on one L4 (24 GB), in GB (ADR 0001): serving and the
# learner compose on one device because vLLM budgets against the device total.
MAIN_GB = 7.2
LEARNER_GB = 9.6

METAL = "steer-l4"
SUBDIR = "steer"
IDLE_S = 90.0                 # the desk's clock; the venue's scaledown is no shorter (ADR 0003 Q3)
MINE = (METAL,)                 # the metals THIS venue registers, and the only ones it releases


def proposed_recipe():
    """WHAT THIS METAL IS FOR, PROPOSED (ADR 0007, Q4): an engine that SERVES
    lora and steer — the steer's demands (our worker class, eager mode) are
    paid by the build and refused at construction if they cannot be — and a
    plain learner. The desk journals this as its own declaration and its
    `recipe` door can overwrite it; the carve carries whichever came last."""
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    return Builds(
        engine=EngineBuild(max_model_len=2048, max_bundles=8, max_rank=RANK,
                           serves=("lora", "steer"), enforce_eager=True),
        learner=LearnerBuild())


MetalS = metal_class(app, APP, METAL, GPU, gpu_image, module=__name__,
                     idle_s=IDLE_S, recipe=proposed_recipe())


# ---------------------------------------------------------------------------
# the science: two banks, one topology, the same plans
# ---------------------------------------------------------------------------

def rollout_plan(task_ids, updates: int):
    """Every wave: GROUPS_PER_WAVE tasks, GROUP_SIZE completions each."""
    from rlstack import GroupPlan, RunPlan, Sample, WavePlan

    def wave(u: int):
        chosen = [task_ids[(u * GROUPS_PER_WAVE + g) % len(task_ids)]
                  for g in range(GROUPS_PER_WAVE)]
        # a group is the advantage's baseline scope, not a task: two groups
        # of one problem are two baselines, keyed apart (a wave refuses
        # duplicate group keys — found on the venue, the plora key is the rule)
        return WavePlan(tuple(
            GroupPlan(f"{task_id}#{g}", tuple(Sample(task_id, "dapo_math")
                                              for _ in range(GROUP_SIZE)))
            for g, task_id in enumerate(chosen)))
    return RunPlan(tuple(wave(u) for u in range(updates)))


def train_plan(updates: int):
    """Update u trains on rollout u, whole: plain on-policy GRPO."""
    from rlstack import RunPlan, WaveRef
    return RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                         for u in range(1, updates + 1)))


def spec_for(store, bank: dict, overrides: dict, updates: int, master: int):
    """One experiment as one value: the given bank on the shared topology."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, GenSpec, HostSpec, LearnerMember, OptimSpec,
        Plans, PolicySpec, PoolMember, SamplingSpec, Schedule, Seeds, Topology,
        encode,
    )
    from rlstack.data.tasks import load_tasks

    if not any(t.id == SCREENED_TASK for t in load_tasks(store, TRAIN_TASKS)):
        raise ValueError(f"{SCREENED_TASK!r} is not in {TRAIN_TASKS}")
    plans = Plans(train=store.cas_put(encode(train_plan(updates))),
                  rollout=store.cas_put(encode(rollout_plan([SCREENED_TASK],
                                                            updates))))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank=bank),
        gen=GenSpec(envs=("dapo_math",), tasks=(TRAIN_TASKS, EVAL_TASKS),
                    sampling=SamplingSpec(temperature=1.0, max_tokens=MAX_TOKENS)),
        plans=plans,
        algo=AlgoSpec(loss="grpo", post=("final_answer", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-5, overrides=overrides),
                      schedule=Schedule(microbatch_tokens=512, max_policy_lag=1)),
        # the SAME demands for both specs: the second submission JOINS the
        # first's listings (coverage is capability equality), which is I8
        # through the desk — one serving host, one learner, two tenants
        topology=Topology(hosts=(
            HostSpec((PoolMember("main", tp=TP, vram_gb=MAIN_GB * TP),)),
            HostSpec((LearnerMember(fsdp=1, vram_gb=LEARNER_GB),)))),
        seeds=Seeds(master=master))


def the_two_banks() -> dict[str, tuple[dict, dict]]:
    """lora-only and steer-only: {name: (bank, optimizer overrides)}."""
    from rlstack import lora, steer
    return {
        "lora": ({"pi": lora(LORA_SITE, r=RANK)}, {}),
        "steer": ({"nudge": steer(STEER_SITE, d=HIDDEN, init_std=0.02)},
                  {"nudge": {"lr": STEER_LR}}),
    }


@app.function(image=cpu_image, volumes={"/store": store_volume}, timeout=600)
def build_specs(master: int = 11) -> dict:
    """The two specs, plans in the CAS, as canonical rows for the client."""
    from rlstack import canonical_json

    store = a_store()
    rows = {name: json.loads(canonical_json(
                spec_for(store, bank, overrides, UPDATES, master)))
            for name, (bank, overrides) in the_two_banks().items()}
    store_volume.commit()
    return rows


ledgers = progress_function(app, cpu_image, module=__name__, name="ledgers",
                            tail=8)
"""Each run's committed updates and its train blocks — the chassis' one
extent reader, which is also what a campaign door follows."""


# ---------------------------------------------------------------------------
# promises 1-2: parity, one container, no desk
# ---------------------------------------------------------------------------

class Checks:
    """A named list of pass/fail lines; the run fails loudly at the end."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
              flush=True)

    def summary(self) -> dict:
        passed = sum(1 for _, ok, _ in self.rows if ok)
        return {"passed": passed, "total": len(self.rows),
                "failed": [name for name, ok, _ in self.rows if not ok]}


@app.function(image=gpu_image, gpu=GPU, volumes={"/hf": hf_cache}, timeout=2400)
def probe() -> dict:
    """The lowering's own exam: the engine's hook against the trainer's
    wrapper, on the same numbers, plus the cache and the window."""
    import asyncio

    import torch
    import transformers
    import vllm

    from rlstack import Message, Role, SteerWindow
    from rlstack.data.flatten import TokenBatch
    from rlstack.policy.adapters import lora_torch, steer_torch
    from rlstack.policy.adapters.replay import ReplayRows, row_plan
    from rlstack.policy.adapters.steer import STEER_RECORD
    from rlstack.policy.compile import compile_bundle
    from rlstack.policy.siteschema import hf_schema, resolve
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.learners.torch_learner import TorchLearner, _doc_spans
    from rlstack.spec.specs import SamplingSpec

    check = Checks()
    print(f"[pins] vllm={vllm.__version__} torch={torch.__version__} "
          f"transformers={transformers.__version__} tp={TP}")

    engine = VllmEngine(BASE, gpu_memory_utilization=0.45, max_model_len=512,
                        max_rank=RANK, tp=TP, serves=("lora", "steer"))
    learner = TorchLearner()
    learner._ensure_base(BASE)
    model = learner._model
    schema = hf_schema(BASE)
    boundaries = resolve(schema.sites, STEER_SITE)
    weighted = resolve(schema.sites, LORA_SITE)
    check("the base is as wide as the spec says",
          int(model.config.hidden_size) == HIDDEN, f"{model.config.hidden_size}")

    reach = engine.reachability(schema.sites)
    check("the build reaches the residual boundaries through the hook",
          all(str(reach[m.name]) == "residual" for m in boundaries)
          and str(reach["final_hidden"]) == "residual")
    check("the build still reports punica on weighted sites",
          str(reach[weighted[0].name]) == "punica")
    check("the logits are honestly another lever's", str(reach["logits"]) == "none")

    # ---- the states -----------------------------------------------------------
    zero = steer_torch.build(boundaries, {"d": HIDDEN, "init_std": 0.0})
    nudges = {std: steer_torch.build(boundaries, {"d": HIDDEN, "seed": 300 + i,
                                                  "init_std": std})
              for i, std in enumerate((0.01, 0.05, 0.2))}
    delta = lora_torch.build(weighted, {"r": RANK, "seed": 101})
    generator = torch.Generator().manual_seed(101)
    for path in delta.b:
        delta.b[path].data = torch.randn(*delta.b[path].shape,
                                         generator=generator) / (RANK * 40)
    steer_torch.install(model, zero)
    for state in nudges.values():
        steer_torch.install(model, state)
    lora_torch.install(model, delta)

    def steer_slot(state):
        return {meta.path: state for meta in boundaries}

    lora_slot = {meta.path: delta for meta in weighted}
    banks = {
        "base": ({}, {}),
        "zero": (steer_slot(zero), {"nudge": steer_torch.emit(zero)}),
        "lora": (lora_slot, {"pi": lora_torch.emit(delta)}),
        "both": (lora_slot | steer_slot(nudges[0.05]),
                 {"pi": lora_torch.emit(delta),
                  "nudge": steer_torch.emit(nudges[0.05])}),
    }
    for std, state in nudges.items():
        banks[f"steer@{std}"] = (steer_slot(state), {"nudge": steer_torch.emit(state)})
    adapter_types = {"pi": "lora", "nudge": "steer"}
    bundles = {}
    for name, (_, payloads) in banks.items():
        bundle = compile_bundle(payloads, {n: 0 for n in payloads},
                                servable=payloads, adapter_types=adapter_types)
        engine.add_bundle(bundle)
        bundles[name] = bundle
    check("a bank of lora + steer registers with BOTH lowerings",
          set(engine.attachments(bundles["both"].bundle_id)) == {"lora", "steer"})

    # ---- the two sides ---------------------------------------------------------
    def replayed(slot, prompt: str, answer: str, window=(0, None)) -> tuple:
        """The trainer's logprob per answer token from ONE routed forward,
        the row's turn recording `window` — forward_backward's own call."""
        context = list(engine.tokenize(prompt))
        ids = context + list(engine.tokenize(answer))
        batch = TokenBatch(token_ids=tuple(ids), loss_mask=(1,) * len(ids),
                           behavior_logprobs=(0.0,) * len(ids),
                           segment_ids=(0,) * len(ids), doc_starts=(0,))
        plan = ReplayRows(slots=(slot,),
                          index=torch.zeros(1, dtype=torch.long, device=learner.device),
                          facts=(({STEER_RECORD: list(window)},),))
        with torch.no_grad():
            with row_plan(model).route(plan):
                out = learner._batched_logprobs(batch, _doc_spans(batch))
        return tuple(float(x) for x in out[len(context):])

    async def scored(name: str, prompt: str, answer: str, directives=()) -> tuple:
        return await engine.score_tokens([Message(Role.USER, prompt)],
                                         engine.tokenize(answer),
                                         bundles[name].bundle_id, directives)

    def gap(a, b) -> float:
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    def worst(a, b) -> float:
        return max(abs(x - y) for x, y in zip(a, b))

    def shifted_gap(a, b) -> float:
        return sum(abs(x - y) for x, y in zip(a[1:], b[:-1])) / (len(a) - 1)

    PAIRS = [
        ("What is 17 + 26? Answer with the number only.", " 43"),
        ("Name the largest planet in the solar system.", " Jupiter is the largest."),
        ("List three primes greater than ten.", " 11, 13 and 17."),
    ]

    async def measure() -> dict:
        report: dict = {}
        # 1. the zero-tolerance control, both sides
        for prompt, answer in PAIRS:
            check(f"engine: zero steer is the base bit for bit ({prompt[:18]!r})",
                  worst(await scored("zero", prompt, answer),
                        await scored("base", prompt, answer)) == 0.0)
            check(f"trainer: zero steer is the base bit for bit ({prompt[:18]!r})",
                  worst(replayed(banks["zero"][0], prompt, answer),
                        replayed({}, prompt, answer)) == 0.0)
        # 2. parity at three magnitudes, and lora + steer together
        for name in [f"steer@{std}" for std in nudges] + ["both", "lora"]:
            gaps, peaks, shifted = [], [], []
            for prompt, answer in PAIRS:
                engine_side = await scored(name, prompt, answer)
                trainer_side = replayed(banks[name][0], prompt, answer)
                gaps.append(gap(engine_side, trainer_side))
                peaks.append(worst(engine_side, trainer_side))
                shifted.append(shifted_gap(engine_side, trainer_side))
            report[name] = {"gap": max(gaps), "peak": max(peaks),
                            "shifted": min(shifted)}
            print(f"  [{name}] gap {max(gaps):.4f} peak {max(peaks):.4f} "
                  f"shift-control {min(shifted):.3f}")
            check(f"{name}: the engine and the trainer agree",
                  max(gaps) < 0.15, f"gap {max(gaps):.4f}")
            check(f"{name}: an off-by-one would have shown",
                  min(shifted) > 5 * max(gaps))
        # 3. the window: completion-only served == completion-only replayed,
        #    and != every-position served (the window matters)
        prompt, answer = PAIRS[1]
        n = len(engine.tokenize(prompt))
        windowed = await scored("steer@0.2", prompt, answer,
                                directives=(SteerWindow(start=n),))
        everywhere = await scored("steer@0.2", prompt, answer)
        check("a completion-only window replays as recorded",
              gap(windowed, replayed(banks["steer@0.2"][0], prompt, answer,
                                     window=(n, None))) < 0.15)
        check("the window changes the answer",
              worst(windowed, everywhere) > 1e-3, f"{worst(windowed, everywhere):.4f}")
        # 4. the prefix cache: the same prompt under other bundles and windows
        #    in between, then again — a bit-identical repeat, or it aliased
        first = await scored("steer@0.2", prompt, answer)
        await scored("base", prompt, answer)
        await scored("lora", prompt, answer)
        await scored("steer@0.2", prompt, answer, directives=(SteerWindow(start=n),))
        again = await scored("steer@0.2", prompt, answer)
        check("the prefix cache never aliases across bundles or windows",
              worst(first, again) == 0.0, f"{worst(first, again)}")
        base_first = await scored("base", prompt, answer)
        check("...and the base is still the base afterwards",
              worst(base_first, await scored("base", prompt, answer)) == 0.0)
        # 5. the record, and steering on decode: a greedy sample under an
        #    open window and under one closed at the prompt's end. The FIRST
        #    generated token comes off the last prompt position, inside both
        #    windows; the SECOND is the first decode step, at position n —
        #    steered only under the open window.
        async def greedy(directives):
            events = [e async for e in engine.sample_tokens(
                [Message(Role.USER, prompt)],
                SamplingSpec(temperature=0.0, max_tokens=4), (),
                bundles["steer@0.2"].bundle_id, seed=1, directives=directives)]
            return events
        open_events = await greedy(())
        closed_events = await greedy((SteerWindow(0, n),))
        check("the turn records the resolved window",
              open_events[-1].turn_extras[STEER_RECORD] == [0, None]
              and closed_events[-1].turn_extras[STEER_RECORD] == [0, n])
        check("the first token, off the last prompt position, is the same "
              "under both windows",
              open_events[0].token_id == closed_events[0].token_id
              and abs(open_events[0].logprob - closed_events[0].logprob) < 1e-6)
        second_open = open_events[1].logprob
        second_closed = closed_events[1].logprob
        check("the first DECODE step is steered under an open window and not "
              "under a closed one",
              abs(second_open - second_closed) > 1e-4,
              f"{second_open:.4f} vs {second_closed:.4f}")
        return report

    report = asyncio.run(measure())
    engine.shutdown()
    summary = check.summary()
    print(json.dumps({"summary": summary, "gaps": report}, indent=1))
    if summary["failed"]:
        raise SystemExit(f"probe: {summary['failed']}")
    return {"summary": summary, "gaps": report}


@app.function(image=gpu_image, timeout=1800)
def run_tests() -> str:
    """The fakes suite inside the image, where the torch-gated cases run."""
    return run_suite()


# ---------------------------------------------------------------------------
# promises 3-5: the doors
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def up() -> None:
    """Boot the metal (spawning the keepalive is the knock) and wait until
    it has registered itself with the desk."""
    call = metal_handle(APP).serve.spawn()
    print(f"[up] {METAL} serving: call {call.object_id}")
    print(json.dumps(wait_for_metal(METAL), indent=2))


def submit_two_tenants(master: int) -> tuple[dict[str, str], bool]:
    """Promise 3's first half: the two specs through the desk. Returns the
    run ids by bank name and whether the second JOINED the first's listings."""
    rows = build_specs.remote(master)
    runs: dict[str, str] = {}
    hosts: dict[str, dict] = {}
    for name in ("lora", "steer"):
        reply = submit_spec(rows[name], SUBDIR)
        runs[name] = reply["run_id"]
        hosts[name] = reply["pools"]
    joined = hosts["lora"] == hosts["steer"]
    print(f"[join] lora on {hosts['lora']} / steer on {hosts['steer']} -> "
          f"{'ONE serving host, ONE learner' if joined else 'NOT joined'}", flush=True)
    return runs, joined


def await_runs(runs: dict[str, str], timeout_s: float = 3600.0) -> dict:
    """Promise 3's second half: both ledgers reach UPDATES; the rails per
    update, reported."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        progress = ledgers.remote(list(runs.values()))
        line = {name: progress[rid]["committed"] for name, rid in runs.items()}
        print(f"[ledger] {json.dumps(line)}", flush=True)
        if all(n >= UPDATES for n in line.values()):
            break
        time.sleep(30)
    else:
        raise SystemExit("the runs did not finish within the deadline")
    rails = {}
    for name, rid in runs.items():
        gaps = [round(t.get("logprob_gap", -1.0), 4) for t in progress[rid]["train"]]
        losses = [round(t.get("loss", 0.0), 4) for t in progress[rid]["train"]]
        print(f"[{name}] {rid}: logprob_gap per update {gaps}, loss {losses}",
              flush=True)
        rails[name] = {"run_id": rid, "logprob_gap": gaps, "loss": losses}
    return rails


@app.local_entrypoint()
def check(master: int = 11) -> None:
    """THE CHECK: up, two tenants through the desk, their ledgers, then the
    desk's release with the keepalive observed to return — and, whatever
    happened, THIS venue's metal released and asserted freed. Its own, not
    the whole plane: since ADR 0007 one desk serves every venue.

    A SYNC entrypoint on purpose: the desk's admission-free verbs (status)
    are blocking Modal calls on a worker thread, which deadlock under an
    entrypoint's own event loop; the async verbs (submit, release) are
    awaited explicitly, one loop each."""
    call = metal_handle(APP).serve.spawn()
    print(f"[check] {METAL} serving: call {call.object_id}", flush=True)
    try:
        print(json.dumps(wait_for_metal(METAL), indent=1), flush=True)
        runs, joined = submit_two_tenants(master)
        rails = await_runs(runs)
        print(json.dumps({"joined": joined, "rails": rails}, indent=1), flush=True)
    finally:
        take_down(MINE, call, "steer check done")


@app.local_entrypoint()
def finish(call_id: str = "") -> None:
    """The check's second half, for runs already delivered (a driver that
    died mid-check): the runs off the desk's placements, their ledgers, then
    the takedown — with the keepalive's call id, its return observed."""
    import asyncio

    placed = asyncio.run(desk().placements())
    runs = {rid[:12]: rid for rid in sorted(placed)}
    print(f"[finish] runs on the desk: {runs}", flush=True)
    call = modal.FunctionCall.from_id(call_id) if call_id else None
    try:
        if runs:
            rails = await_runs(runs)
            print(json.dumps({"rails": rails}, indent=1), flush=True)
    finally:
        take_down(MINE, call, "steer check finished")


@app.local_entrypoint()
def knock() -> None:
    """The door back (ADR 0003 Q4, ADR 0004 Q10): a placement for the same
    demands KNOCKS the released metal awake, the container announces, the
    desk lists it carve-able again — then it is released once more."""
    import asyncio

    from rlstack.runner.desk import Demand

    before = fleet().get("metal", {}).get(METAL, {})
    print(f"[knock] before: released={before.get('released')} plane={before.get('plane')}")
    try:
        placed = asyncio.run(desk().resolve((
            Demand(pool="main", capability="inference", base=BASE, shape=TP,
                   vram_gb=MAIN_GB * TP, group=0),)))
        print(f"[knock] placed: {json.dumps(placed, default=str)[:400]}")
        after = fleet().get("metal", {}).get(METAL, {})
        print(f"[knock] after: released={after.get('released')} plane={after.get('plane')}")
        if not after.get("plane"):
            raise SystemExit("the knock did not bring the metal back")
    finally:
        take_down(MINE, None, "knock check done")


@app.local_entrypoint()
def down(call_id: str = "") -> None:
    """Hand the metal back by hand and, given the keepalive's call id, watch
    it return."""
    take_down(MINE, modal.FunctionCall.from_id(call_id) if call_id else None,
              "released by hand")
