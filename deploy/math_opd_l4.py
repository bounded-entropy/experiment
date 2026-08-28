"""OPD at task scale: an 8B student distilled from a 32B teacher on MATH.

    modal run deploy/math_opd_l4.py::tasks        # build the task set, report it
    modal run deploy/math_opd_l4.py::baseline     # the ceiling and the floor
    modal run deploy/math_opd_l4.py::shakeout     # a few updates of the real thing
    modal run deploy/math_opd_l4.py::report       # a finished run, out of its store
    modal run deploy/math_opd_l4.py::score_clock  # what the teacher costs, alone
    modal run deploy/math_opd_l4.py::full --go    # the 50-100 update run (GATED)

deploy/opd_l4.py proved the PLUMBING of on-policy distillation (#47): three
per-capability hosts, a live 32B teacher scoring the student's own draws, four
updates on toy arithmetic with twelve-token completions. This file is its
bigger sibling and changes exactly one thing — THE TASK. Hendrycks MATH at
levels 3-5, real completions, so that every number the run reports (reward,
reverse KL, and above all the wall time the INLINE teacher scoring costs per
update) is a number about a workload someone would actually run.

WHAT THIS RUN EXISTS TO MEASURE. Teacher scoring rides the Trainer's post
phase inline: an update's gradient waits on a group's worth of sequential 32B
prefills. On twelve-token completions that is invisible. On MATH completions
it is the bottleneck the async scorer daemon is designed to remove — so this
file times it (`ScoreClock`) and reports seconds per update, which is the
input to deciding whether the daemon is worth building and how much it buys.

THE TASK SET (deterministic, content-addressed):
    levels 3-5, filtered to answers THE EXISTING VERIFIER CAN CHECK — the
    boxed answer must parse as a plain integer, |answer| < 1000, because
    training/post/verifier.py matches the LAST number in the completion
    against str(meta["answer"]) and a comma or a fraction bar defeats that.
    Nothing in rlstack/ is touched to make MATH fit; the task set is chosen to
    fit the verifier that exists.

THE PROMPT (raw completion, the engine's v0 contract — no chat template):
    three worked exemplars, each ending "The answer is <n>." and terminated by
    an eos token, then the problem. THE TERMINATOR IS THE LOAD-BEARING PART:
    the environment (math_single_turn) passes NO stop strings, so a completion
    runs to max_tokens unless the model emits an eos id — and the verifier
    reads the LAST number in it, which would then come from whatever the model
    invented after answering. See DOCUMENT_END for which of Qwen3's two eos
    ids works and the measurement that settled it; `baseline` reports the
    finish-reason mix, so the scaffold is judged before any GPU-hour is
    committed to a run.

TOPOLOGY (#47's, unchanged except max_model_len):
    teacher    Qwen3-32B tp=4 on L4:4, frozen, inference regime only
    student    Qwen3-8B  tp=2 on L4:2, the sampler
    learner    Qwen3-8B  fsdp=2 on L4:2, the runner beside it
    max_model_len 2048 (was 512): prompt ~450 + 512 generated, with the
    teacher's prefill of both inside the same window. #47 measured the 32B's
    KV at 49,664 tokens, so 2048 per sequence is affordable — `shakeout`
    is where that stops being arithmetic and becomes an observation.

WHAT THE METAL SAID (2026-08-28; ::tasks, ::baseline, ::shakeout run
d43141929dc2 at 6 updates, ::score_clock). The full run has NOT been run.

    THE TASK SET   7,500 train rows -> 5,586 at levels 3-5 -> 2,925 the
                   verifier can check (52.4%); test 5,000 -> 3,669 -> 1,892
                   (51.6%). Prompts are 211-543 tokens (median 249), so
                   prompt + 512 sits at 1,055 of the 2,048 window.
    THE WINDOW     max_model_len 2048 costs the teacher nothing: 16.63 GiB of
                   weights per device and 3.03 GiB of KV = 49,664 tokens —
                   #47's number exactly, at four times the window, because KV
                   capacity is a memory fact and the window is a per-sequence
                   one. The student holds 8.27 GiB and 150,224 tokens of KV.
    THE BASELINE   greedy, 100 held-out tasks: untrained 8B student 0.600,
                   32B teacher 0.520. The teacher is BELOW the student, and
                   the reason is visible in the completions — the 32B leaves
                   the few-shot register for its post-trained one ("Okay,
                   let's try to tackle this problem step by step") and
                   truncates more often (39/100 vs 31/100 at 512 tokens).
    THE LEDGER     reward .125/.000/.562/.812/.188/.750, loss (= per-token
                   reverse KL) +.1043/.0739/.0576/.0846/.0716/.0568 nats,
                   gap .0160-.0393, ~100s per update after the first.
                   32,155 teacher-scored tokens, token-aligned in every
                   update; teacher mean logprob -0.3077 against the student's
                   -0.2335. Held-out eval (32 tasks, temperature 1.0) 0.531
                   at update 3 and 0.281 at update 6 — two points, n=32.
    THE RAILS      logprob_gap 0.0160-0.0393 with documents 40x longer than
                   #47's: the kernel floor, unmoved.
    THE CLOCK      and it is the answer this file was built for: inline
                   teacher scoring costs ~8s of a ~100s update — 16 prefills
                   over 5,485 tokens, median 0.94s each, span 8.0s, teacher
                   busy 15.4s. The span is group_size x per-prefill latency
                   (run_pipeline gathers over GROUPS and walks trajectories
                   within one sequentially), so it scales with group_size,
                   not with wave size. At this shape the async scorer daemon
                   would buy back under a tenth of an update — and the L4:4
                   teacher sits idle for the other 92%, which is the real
                   economics and a different fix.

Deployment only (I5): wiring and measurement. Image pins are modal_app.py's
plus pyarrow, added as its own layer so the pinned vllm/torch layer is reused
byte-for-byte from cache.

NOTE (modal 1.5): no `from __future__ import annotations` in this file — it
stringifies class annotations, and modal.parameter validates its fields by
their annotation OBJECT (deploy/modal_host.py carries the same note).
"""

import json
import random
import re

import modal

from probe import CHECKS, check

app = modal.App("rlstack-math-opd")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlstack-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    # its own layer: the pinned line above stays cache-identical to the other
    # deploys'. pyarrow is the whole dependency MATH costs us — the dataset
    # ships as parquet and huggingface_hub (a transformers dep) fetches it.
    .pip_install("pyarrow==25.0.1")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
          "OMP_NUM_THREADS": "1",
          # ragged documents mean every microbatch is a different shape, and
          # a caching allocator that cannot grow a segment strands the
          # difference: the learner's first OOM reported 2.32 GiB reserved
          # but unallocated out of 22. torch's own suggested remedy.
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "HF_HOME": "/hf"})
    .add_local_python_source("probe", "rlstack", "rlstack_engine")
)

STUDENT = "Qwen/Qwen3-8B"
TEACHER = "Qwen/Qwen3-32B"
STORE = "modal://rlstack-store"
VOLUMES = {"/store": store_volume, "/hf": hf_cache}

MAX_MODEL_LEN = 2048        # prompt + completion, and the teacher's prefill
MAX_TOKENS = 512            # the completion budget
PROMPT_TOKEN_CAP = 1024     # a task whose prompt exceeds this is dropped


# ---------------------------------------------------------------------------
# the task set: MATH levels 3-5, filtered to what the verifier can check
# ---------------------------------------------------------------------------

MATH_REPO = "EleutherAI/hendrycks_math"
# pinned like any other dependency (I7): the task file is content-addressed,
# so a dataset that moved under us would silently fork run identity.
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry",
                 "intermediate_algebra", "number_theory", "prealgebra",
                 "precalculus")
LEVELS = ("Level 3", "Level 4", "Level 5")

# The exemplar terminator, and the most load-bearing string in this file.
# Both Qwen3 checkpoints list TWO eos ids (151645 <|im_end|>, 151643
# <|endoftext|>), so either stops a vLLM request — but only one of them is a
# token the model was ever trained to PREDICT. Measured, not assumed:
# <|endoftext|> (the pretraining document separator) was tried first and the
# 8B ignored it in 50/50 completions, reproducing the Problem/Solution format
# perfectly while never emitting the separator between blocks — which is what
# a target masked out of the pretraining loss looks like from the outside.
# <|im_end|> is the token post-training spends its whole life predicting.
DOCUMENT_END = "<|im_end|>"      # Qwen3 id 151645

EXEMPLARS = (
    ("What is the value of $3^2 + 4\\cdot 2 - 5$?",
     "$3^2 = 9$ and $4\\cdot 2 = 8$, so the expression equals $9 + 8 - 5 = 12$.",
     12),
    ("If $2x + 7 = 19$, what is the value of $5x$?",
     "Subtracting 7 from both sides gives $2x = 12$, so $x = 6$, and $5x = 30$.",
     30),
    ("How many positive divisors does $36$ have?",
     "$36 = 2^2\\cdot 3^2$, so it has $(2+1)(2+1) = 9$ positive divisors.",
     9),
)

INTEGER = re.compile(r"^-?\d+$")


def few_shot_prompt(problem, style):
    """The raw-completion scaffold, in the two shapes worth measuring.

    Both show the same three worked exemplars and both terminate each one, so
    they differ in exactly one thing — what the target invites:

        "cot"     the prompt ends "Solution:", so the target block is
                  IDENTICAL IN SHAPE to the three above it: the model works
                  the problem, says "The answer is <n>." and terminates.
        "direct"  the prompt ends "The answer is", so the answer is the very
                  next token — the shape the plan pre-registered.

    MEASURED, same 50 held-out tasks, same greedy 8B, same verifier
    (::baseline --students-only): cot 0.500 accuracy, 35/50 clean eos, 287
    tokens mean; direct 0.040 accuracy, 9/50 eos, 421 tokens. Both halves of
    that gap are the same cause — a target block that does not match the
    exemplars is not a pattern the model completes, so it neither reasons nor
    stops. direct's answer-first guess IS right 18% of the time (the
    first-number diagnostic), but it then keeps writing and the verifier's
    last-number rule reads whatever it invented next. cot is the default.

    Style is a property of the TASK FILE (it is inside the prompt text), so
    the two are different content-addressed task sets and therefore different
    experiments — which is the honest way to compare them.
    """
    blocks = [f"Problem: {problem_}\nSolution: {work} The answer is {answer}."
              f"{DOCUMENT_END}\n"
              for problem_, work, answer in EXEMPLARS]
    tail = "Solution:" if style == "cot" else "The answer is"
    return "".join(blocks) + f"Problem: {problem}\n{tail}"


def boxed_answer(solution):
    """The content of the LAST \\boxed{...} in a MATH solution, brace-matched
    (the argument itself contains braces: \\boxed{\\frac{1}{2}})."""
    start = solution.rfind("\\boxed")
    if start < 0:
        return None
    open_brace = solution.find("{", start)
    if open_brace < 0:
        return None
    depth = 0
    for i in range(open_brace, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return solution[open_brace + 1:i]
    return None


def verifiable_answer(solution):
    """The boxed answer as an int, or None if THIS verifier cannot check it.

    The rule is training/post/verifier.py's, read backwards: it compares the
    LAST run of digits in the completion against str(meta["answer"]). So a
    fraction, a surd, an expression, or a four-digit answer the model may
    write as "1,000" are all unverifiable — not wrong, just outside what the
    existing reward can score, and this run does not modify the verifier.
    """
    boxed = boxed_answer(solution)
    if boxed is None:
        return None
    text = (boxed.strip().replace(",", "").replace("\\!", "")
            .replace("$", "").replace("\\", "").strip())
    if text.startswith("+"):
        text = text[1:]
    if not INTEGER.match(text):
        return None
    value = int(text)
    return value if abs(value) < 1000 else None


def hendrycks_math(split):
    """Every MATH row of one split, from the pinned dataset revision.

    Fetched INSIDE the container from the hub (parquet, one file per subject)
    and read with pyarrow — `datasets` would drag a resolver stack in beside
    the pinned vllm for a fourteen-file download.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rows = []
    for subject in MATH_SUBJECTS:
        path = hf_hub_download(
            MATH_REPO, f"{subject}/{split}-00000-of-00001.parquet",
            repo_type="dataset", revision=MATH_REVISION)
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def math_pool(split, style):
    """The filtered, deterministically ordered task pool for one split.

    Order is the whole point: the file's bytes must be reproducible in any
    container, so rows are sorted by (level, subject, problem) and then
    shuffled with a fixed seed. Every task set below is a PREFIX of this
    pool, which is why the eval set is inside the baseline's 100 tasks.
    """
    kept = []
    for row in hendrycks_math(split):
        if row["level"] not in LEVELS:
            continue
        answer = verifiable_answer(row["solution"])
        if answer is None:
            continue
        if "[asy]" in row["problem"]:      # Asymptote figure source, unreadable
            continue
        kept.append({"problem": row["problem"], "answer": answer,
                     "level": row["level"], "type": row["type"]})
    kept.sort(key=lambda r: (r["level"], r["type"], r["problem"]))
    random.Random(20260828).shuffle(kept)
    return [dict(r, prompt=few_shot_prompt(r["problem"], style)) for r in kept]


def task_rows(pool, split, n):
    """The first `n` tasks of a pool as store rows — {id, prompt, meta}."""
    return [{"id": f"math-{split}-{i:04d}", "prompt": row["prompt"],
             "meta": {"answer": row["answer"], "level": row["level"],
                      "type": row["type"]}}
            for i, row in enumerate(pool[:n])]


def task_bytes(rows):
    """One jsonl blob, byte-stable: sorted keys, one row per line (the shape
    probe.arith_tasks writes and runner/traffic.load_tasks reads)."""
    return "".join(json.dumps(r, sort_keys=True) + "\n"
                   for r in rows).encode()


def put_task_files(store, style, n_train, n_heldout):
    """Both task files in the store, content-addressed. Returns the two uris.

    Rebuilt from the pinned dataset in whatever container needs them rather
    than threaded through as CLI arguments: cas_put of identical bytes is
    idempotent, so "rebuild it" and "reuse it" are the same operation and the
    spec's identity cannot drift between the legs.
    """
    train = task_bytes(task_rows(math_pool("train", style), "train", n_train))
    heldout = task_bytes(task_rows(math_pool("test", style), "test", n_heldout))
    return store.cas_put(train), store.cas_put(heldout)


# ---------------------------------------------------------------------------
# the serving end: one capability per container (deploy/opd_l4.py's shape,
# with the longer context window this task needs)
# ---------------------------------------------------------------------------

def serve_one_partition(name, base, tp, memory):
    """Build the metal, wear it as a Host, expose it. The container IS the
    host — born with its Partition and its one inference Regime, attesting the
    engine it was handed against them (#43) and journaling host-up so the
    observer's `hosts` view sees a remote partition like any other."""
    from rlstack import ModalVolumeStore
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.host import Host, Partition, Regime
    from rlstack.runner.remote import HostService

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    engine = VllmEngine(base, tp=tp, gpu_memory_utilization=memory,
                        max_model_len=MAX_MODEL_LEN, max_loras=8,
                        max_lora_rank=16, serves=("lora",))
    host = Host(name, engines=(engine,), learner=None, store=store,
                partition=Partition("modal-l4", tuple(range(tp)), memory),
                regimes=(Regime(f"serve-tp{tp}", "inference", base, tp),))
    print(f"[host {name}] up: {base} tp={tp} mem={memory} "
          f"max_model_len={MAX_MODEL_LEN}")
    return host, HostService(host)


@app.cls(image=image, gpu="L4:4", volumes=VOLUMES, timeout=14400,
         scaledown_window=1800, max_containers=1, cpu=8.0, memory=65536)
@modal.concurrent(max_inputs=64)
class TeacherHost:
    """32B across four L4s, frozen. It trains nothing and owns no learner: a
    teacher is an inference capability the fleet addresses by (base, tp)."""

    base: str = modal.parameter(default=TEACHER)
    tp: int = modal.parameter(default=4)
    host_name: str = modal.parameter(default="modal-math-teacher-32b")

    @modal.enter()
    def bring_up(self):
        self.host, self.service = serve_one_partition(
            self.host_name, self.base, self.tp, 0.90)

    @modal.exit()
    def persist_cache(self):
        hf_cache.commit()

    @modal.method()
    async def call(self, verb: str, payload: dict) -> dict:
        return await self.service.serve(verb, payload)

    @modal.method()
    def ask(self, verb: str, payload: dict) -> dict:
        return self.service.answer(verb, payload)

    @modal.method()
    def status(self) -> dict:
        return {"describe": self.service.describe(),
                "status": {k: v for k, v in self.host.status().items()
                           if k != "partition"}}


@app.cls(image=image, gpu="L4:2", volumes=VOLUMES, timeout=14400,
         scaledown_window=1800, max_containers=1, cpu=8.0, memory=32768)
@modal.concurrent(max_inputs=64)
class StudentHost:
    """The sampler: 8B at tp=2, serving this run's LoRA bundles. A separate
    host from the learner because #45 measured that the two do not fit on one
    2xL4."""

    base: str = modal.parameter(default=STUDENT)
    tp: int = modal.parameter(default=2)
    host_name: str = modal.parameter(default="modal-math-student-8b")

    @modal.enter()
    def bring_up(self):
        self.host, self.service = serve_one_partition(
            self.host_name, self.base, self.tp, 0.85)

    @modal.exit()
    def persist_cache(self):
        hf_cache.commit()

    @modal.method()
    async def call(self, verb: str, payload: dict) -> dict:
        return await self.service.serve(verb, payload)

    @modal.method()
    def ask(self, verb: str, payload: dict) -> dict:
        return self.service.answer(verb, payload)

    @modal.method()
    def status(self) -> dict:
        return {"describe": self.service.describe(),
                "status": {k: v for k, v in self.host.status().items()
                           if k != "partition"}}


class ModalTransport:
    """The client end, verbatim from deploy/modal_host.py: `call` is
    `.remote.aio`, `ask` is `.remote`."""

    def __init__(self, partition):
        self.partition = partition

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.partition.call.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return self.partition.ask.remote(verb, payload)


class ScoreClock:
    """A transport that times score_tokens — the whole reason this run exists.

    Teacher scoring rides the Trainer's post phase INLINE: for each group,
    one sequential prefill per trajectory against a 32B, and the gradient
    waits for all of them. This wraps the wire (a deploy-side object; nothing
    in rlstack/ knows) and records one interval per score, which is enough to
    report BUSY TIME per update — the number the async scorer daemon would be
    built to reclaim.
    """

    def __init__(self, inner):
        self.inner = inner
        self.scores = []      # {"start", "end", "tokens"} per score_tokens

    async def call(self, verb: str, payload: dict) -> dict:
        import time

        started = time.time()
        reply = await self.inner.call(verb, payload)
        if verb == "score_tokens":
            self.scores.append({"start": started, "end": time.time(),
                                "tokens": len(payload["token_ids"])})
        return reply

    def ask(self, verb: str, payload: dict) -> dict:
        return self.inner.ask(verb, payload)

    def one_update(self, update, opened, closed):
        """The scoring calls that fell inside one update's wall-clock window:
        their token count, the SPAN from the first score's start to the last
        one's end (the time the gradient actually waited), and the summed
        per-call latency (the teacher's own busy time, which exceeds the span
        because the pipeline scores its groups concurrently)."""
        inside = [s for s in self.scores if opened <= s["start"] < closed]
        if not inside:
            return None
        return {
            "update": update,
            "calls": len(inside),
            "tokens": sum(s["tokens"] for s in inside),
            "span_s": round(max(s["end"] for s in inside)
                            - min(s["start"] for s in inside), 1),
            "busy_s": round(sum(s["end"] - s["start"] for s in inside), 1),
            "update_s": round(closed - opened, 1),
        }


class LedgerWatch:
    """When each update was committed, from the driver's own clock — and the
    place every per-update number is PRINTED AS IT LANDS.

    The ledger records what an update DID, never when, so the wall-clock
    boundary between updates has to be observed. It PEEKS (the observer's
    read-only verb): open_run would attach, and attaching sweeps unsealed
    work out from under a live run.

    Printing here rather than at the end is not cosmetic. A driver that
    reports only after its last update reports nothing at all if the
    container dies during teardown — which is exactly how the first
    six-update shakeout lost its whole summary (the fsdp child outlived
    learner.stop(), the interpreter's exit hung joining it, and Modal's
    30-second shutdown grace killed the container). Every number this file
    exists to measure is now in the log the moment it is true.
    """

    def __init__(self, store, run_id, started, clock=None):
        self.store = store
        self.run_id = run_id
        self.previous = started
        self.clock = clock
        self.boundaries = {}
        self.timing = []

    async def watch_until_cancelled(self, every=2.0):
        import asyncio
        import time

        while True:
            for entry in self.store.peek_ledger(self.run_id):
                update = int(entry["update"])
                if update not in self.boundaries:
                    now = time.time()
                    opened = self.previous
                    self.boundaries[update] = (opened, now)
                    self.previous = now
                    print(f"[commit] update {update} after {now - opened:.0f}s:"
                          f"  reward {entry['post']['reward']:.3f}"
                          f"  loss {entry['train']['loss']:+.4f}"
                          f"  gap {entry['train']['logprob_gap']:.4f}"
                          f"  grad {entry['train']['grad_norm']:.3f}"
                          f"  tokens {entry['train']['tokens']}", flush=True)
                    self._report_scoring(update, opened, now)
            await asyncio.sleep(every)

    def _report_scoring(self, update, opened, closed):
        """The inline-teacher-scoring line for one update — the measurement
        this whole file was built to take."""
        if self.clock is None:
            return
        row = self.clock.one_update(update, opened, closed)
        if row is None:
            return
        self.timing.append(row)
        share = 100.0 * row["span_s"] / row["update_s"] if row["update_s"] else 0
        print(f"[scoring] update {update}: {row['calls']} teacher scores over "
              f"{row['tokens']} tokens — span {row['span_s']}s of the update's "
              f"{row['update_s']}s ({share:.0f}%), teacher busy "
              f"{row['busy_s']}s", flush=True)


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------

def math_opd_spec(store, *, style, n_updates, n_train, n_heldout, eval_every,
                  master, max_tokens=MAX_TOKENS, group_size=8,
                  trajectories_per_wave=16, microbatch_tokens=512,
                  max_policy_lag=1, lr=1e-4):
    """One on-policy-distillation experiment over MATH, declared and no more.

    The spec says WHAT: a student that samples, a teacher pool that scores its
    draws, a loss that reads the resulting column, a verifier that scores the
    answer. It never says WHERE — three capability demands (a tp=2 sampler, a
    tp=4 teacher, an fsdp=2 learner) which placement satisfies with three
    per-capability hosts.

    microbatch_tokens is 512, not #47's 2048, and the metal chose the number.
    pack() bounds a microbatch by its token SUM, but the trainer's batched
    forward pads rows to the LONGEST document in it — a ratio the docstring
    of _batched_logprobs names and #47 never paid, because twelve-token
    completions are not ragged. MATH completions are: 50 to 512 generated
    tokens on a ~250-token prompt. At 1024 a microbatch of five documents
    padded out to ~3,500 positions and the fsdp=2 learner died in the
    forward with 19.36 of 22.03 GiB allocated (rank 1, o_proj). At 512
    almost every microbatch is ONE document, so padded positions ≈ real
    tokens: less memory AND less wasted compute, at the price of more
    passes. It is the one engineering knob (Schedule's own word) and it
    changes no estimator — the gradient is identical either way.
    """
    from rlstack import (
        AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
        OptimSpec, PolicySpec, SamplingSpec, Schedule, Seeds, TrajectorySource,
        gpus, learner, lora, pool,
    )

    train, heldout = put_task_files(store, style, n_train, n_heldout)
    return ExperimentSpec(
        policy=PolicySpec(base=STUDENT,
                          bank={"pi": lora("layers.*.self_attn.*", r=16)}),
        gen=GenSpec(env="math_single_turn", tasks=train,
                    sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                          max_tokens=max_tokens)),
        trajectories=TrajectorySource("live"),
        algo=AlgoSpec(loss="opd", post=("verifier", "teacher_logprobs"),
                      optim=OptimSpec("adamw", lr=lr),
                      schedule=Schedule(
                          group_size=group_size,
                          trajectories_per_wave=trajectories_per_wave,
                          n_updates=n_updates,
                          microbatch_tokens=microbatch_tokens,
                          max_policy_lag=max_policy_lag)),
        eval=EvalSpec(tasks=heldout, every=eval_every, n_samples=1,
                      post=("verifier",)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=2), (pool("main", tp=2, fraction=0.85),)),
            GpuGroup(gpus(n=4), (pool("teacher", base=TEACHER, tp=4,
                                      fraction=0.90),)),
            GpuGroup(gpus(n=2), (learner(fsdp=2, fraction=0.90),)),
        )),
        seeds=Seeds(master=master),
    )


# ---------------------------------------------------------------------------
# leg 1: the task set, built and described (CPU only)
# ---------------------------------------------------------------------------

@app.function(image=image, volumes=VOLUMES, timeout=1800, cpu=4.0,
              memory=8192)
def tasks(style: str = "cot", n_train: int = 512, n_heldout: int = 128
          ) -> dict:
    """Build both task files, measure them, and say what the filter cost.

    Two things are worth a container of their own before any GPU-hour: the
    FILTER YIELD (how much of MATH levels 3-5 this verifier can score) and the
    PROMPT LENGTH in real tokens against the 2048-token window every later leg
    assumes.
    """
    from collections import Counter

    from transformers import AutoTokenizer

    from rlstack import ModalVolumeStore

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT)

    stats = {}
    for split in ("train", "test"):
        raw = hendrycks_math(split)
        in_levels = [r for r in raw if r["level"] in LEVELS]
        pool = math_pool(split, style)
        stats[split] = {
            "rows": len(raw),
            "levels_3_5": len(in_levels),
            "verifiable": len(pool),
            "yield": round(len(pool) / len(in_levels), 3),
            "per_level": dict(Counter(r["level"] for r in pool)),
            "per_type": dict(Counter(r["type"] for r in pool)),
        }
        print(f"[{split}] {len(raw)} rows -> {len(in_levels)} at levels 3-5 "
              f"-> {len(pool)} the verifier can check "
              f"({stats[split]['yield']:.1%})")
        print(f"        {stats[split]['per_level']}")

    train_uri, heldout_uri = put_task_files(store, style, n_train, n_heldout)
    store_volume.commit()

    rows = task_rows(math_pool("train", style), "train", n_train)
    lengths = sorted(len(tokenizer.encode(r["prompt"], add_special_tokens=False))
                     for r in rows)
    longest = lengths[-1]
    print(f"\n[prompt tokens] min {lengths[0]}  median "
          f"{lengths[len(lengths) // 2]}  p95 {lengths[int(len(lengths) * .95)]}"
          f"  max {longest}")
    check("prompt + completion fits the context window",
          longest + MAX_TOKENS <= MAX_MODEL_LEN,
          f"{longest} + {MAX_TOKENS} vs max_model_len {MAX_MODEL_LEN}")
    check("every prompt is under the cap", longest <= PROMPT_TOKEN_CAP,
          f"longest {longest} <= {PROMPT_TOKEN_CAP}")
    ids = tokenizer.encode(DOCUMENT_END, add_special_tokens=False)
    check("the exemplar terminator is one eos id", ids == [151645],
          f"{DOCUMENT_END!r} -> {ids}")

    print(f"\n[example prompt]\n{rows[0]['prompt']}\n[/example] answer="
          f"{rows[0]['meta']['answer']}")
    print(f"\ntrain   {train_uri}\nheldout {heldout_uri}")
    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    return {"train": train_uri, "heldout": heldout_uri, "stats": stats,
            "prompt_tokens": {"median": lengths[len(lengths) // 2],
                              "max": longest},
            "failed": failed}


# ---------------------------------------------------------------------------
# leg 2: the baseline probe — the ceiling and the floor
# ---------------------------------------------------------------------------

async def measure_pool(label, pool_engine, task_list, max_tokens):
    """One model's greedy accuracy on held-out tasks, through the REAL path.

    Not a hand-rolled eval: the same environment, the same EnginePoolClient,
    the same seal, and the same registered `verifier` postprocessor the run
    itself scores with — so a number here is commensurable with a number in
    the ledger. Greedy (temperature 0) because this is a capability
    measurement, not a sample of the behavior policy.
    """
    import asyncio
    from collections import Counter

    from rlstack import Bundle, Group, SamplingSpec, Wave, derive, run_pipeline
    from rlstack.runner.traffic import EnginePoolClient, run_episode

    bundle = Bundle(f"bundle:baseline:{label}", {})
    pool_engine.add_bundle(bundle)          # payload-free: the bare base
    routes = {"main": (pool_engine, bundle)}
    sampling = SamplingSpec(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    limiter = asyncio.Semaphore(32)

    async def one(index, task):
        async with limiter:
            client = EnginePoolClient(routes, sampling,
                                      derive(0, "baseline", index))
            return await run_episode("math_single_turn", task, client)

    trajectories = await asyncio.gather(
        *[one(i, t) for i, t in enumerate(task_list)])
    wave = Wave([Group(t.id, [traj])
                 for t, traj in zip(task_list, trajectories)])
    columns = await run_pipeline(("verifier",), wave, routes, sampling, 0, 0,
                                 phase="baseline")

    rewards = columns["reward"]
    turns = [traj.turns[0] for traj in trajectories]
    finish = Counter(turn.finish for turn in turns)
    generated = [len(turn.token_ids) for turn in turns]
    first = sum(float(_first_number_matches(traj)) for traj in trajectories)
    accuracy = sum(rewards) / len(rewards)
    # A completion cut off at max_tokens scores 0 whatever the model knew, so
    # the two numbers answer different questions: `accuracy` is what the run
    # will see, `finished` is what the model can do inside the budget.
    finished = [r for r, t in zip(rewards, turns) if t.finish != "length"]
    conditional = sum(finished) / len(finished) if finished else float("nan")
    print(f"\n[{label}] verifier accuracy {accuracy:.3f} over {len(rewards)} "
          f"tasks")
    print(f"    finish reasons     {dict(finish)}")
    print(f"    accuracy | finished {conditional:.3f} "
          f"over {len(finished)} that were not truncated")
    print(f"    generated tokens   mean {sum(generated) / len(generated):.0f}  "
          f"max {max(generated)}")
    print(f"    first-number rule  {first / len(rewards):.3f} "
          f"(diagnostic: the verifier reads the LAST number)")
    for traj in trajectories[:2]:
        print(f"    --- {traj.task.id} answer={traj.task.meta['answer']}\n"
              f"    {traj.turns[0].message.content[:300]!r}")
    return {"accuracy": round(accuracy, 4),
            "accuracy_if_finished": round(conditional, 4) if finished else None,
            "first_number": round(first / len(rewards), 4),
            "finish": dict(finish),
            "mean_generated": round(sum(generated) / len(generated), 1),
            "max_generated": max(generated)}


def _first_number_matches(traj):
    """Diagnostic only: would the verifier have passed if it read the FIRST
    number instead of the last? A wide gap between the two says the model
    answered and then kept writing — i.e. the eos terminator did not take."""
    numbers = re.findall(r"-?\d+", traj.turns[0].message.content)
    return bool(numbers) and numbers[0] == str(traj.task.meta["answer"])


@app.function(image=image, volumes=VOLUMES, timeout=7200, cpu=4.0,
              memory=16384)
def baseline(style: str = "cot", n_tasks: int = 100,
             max_tokens: int = MAX_TOKENS, students_only: bool = False) -> dict:
    """The ceiling and the floor every later reward curve is read against.

    The 32B teacher and the UNTRAINED 8B student, greedy, on the same held-out
    tasks — which are a prefix of the same pool the run's eval set comes from,
    so the eval numbers a run reports sit between these two by construction.
    The two pools are measured CONCURRENTLY because eight L4s bill whether or
    not they are both busy.
    """
    import asyncio

    from rlstack import ModalVolumeStore, RemotePool
    from rlstack.runner.traffic import load_tasks

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    _, heldout = put_task_files(store, style, 1, n_tasks)
    store_volume.commit()
    task_list = load_tasks(store, heldout)
    print(f"[baseline] style={style} {len(task_list)} held-out tasks, "
          f"greedy, max_tokens={max_tokens}")

    student = RemotePool(ModalTransport(StudentHost()), base=STUDENT, tp=2)
    teacher = RemotePool(ModalTransport(TeacherHost()), base=TEACHER, tp=4)
    pools = {"student-8b-untrained": student}
    if not students_only:
        pools["teacher-32b"] = teacher

    async def both():
        return await asyncio.gather(*(
            measure_pool(label, pool_engine, task_list, max_tokens)
            for label, pool_engine in pools.items()))

    measured = dict(zip(pools, asyncio.run(both())))
    floor = measured["student-8b-untrained"]["accuracy"]
    check("the scaffold produces reward at all", floor > 0.0,
          f"untrained student {floor:.3f}")
    if not students_only:
        ceiling = measured["teacher-32b"]["accuracy"]
        check("the teacher is above the student", ceiling > floor,
              f"teacher {ceiling:.3f} vs student {floor:.3f}")
    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    print(f"\n[baseline] style={style} {measured}")
    return {"style": style, "n_tasks": len(task_list), **measured,
            "failed": failed}


# ---------------------------------------------------------------------------
# the run, shared by the shakeout and the full thing
# ---------------------------------------------------------------------------

def warm(pools):
    """Force both containers to BUILD their engines, CONCURRENTLY.

    vLLM's AsyncLLMEngine materializes on the first admitted verb, so without
    this the run's first post phase would sit through a 32B load. Concurrently
    because both partitions are billed the whole time either is loading."""
    import asyncio
    import time

    from rlstack import Bundle, Message, Role

    context = (Message(Role.USER, "What is 47+58? The answer is"),)

    async def one(label, pool_engine):
        bundle = Bundle(f"bundle:warm:{label}", {})
        pool_engine.add_bundle(bundle)
        started = time.time()
        scores = await pool_engine.score_tokens(context, (16, 17, 18),
                                                bundle.bundle_id)
        took = time.time() - started
        print(f"[warm] {label} ready in {took:.0f}s, "
              f"scores {[round(s, 3) for s in scores]}")
        return took

    async def all_of_them():
        return await asyncio.gather(*(one(label, pool_engine)
                                      for label, pool_engine in pools.items()))

    return max(asyncio.run(all_of_them()))


def teacher_column_report(run, updates):
    """What the teacher actually said, read back out of the store.

    TOKEN ALIGNMENT: every trajectory's column has one float per generated
    token of that trajectory's sealed turns. THE SIGNAL: the mean per-token
    (behavior − teacher), the RECORD's reading of the reverse KL the loss
    minimizes."""
    aligned = True
    teacher_all, behavior_all, kl_all = [], [], []
    for u in updates:
        postdata = run.read_postdata(u)
        rows = run.read_wave(u)
        column = postdata["teacher_logprobs"]
        aligned = aligned and len(column) == len(rows)
        for vector, row in zip(column, rows):
            behavior = [lp for turn in row["turns"]
                        for lp in turn["behavior_logprobs"]]
            aligned = aligned and len(vector) == len(behavior)
            teacher_all.extend(vector)
            behavior_all.extend(behavior)
            kl_all.extend(b - t for b, t in zip(behavior, vector))

    def mean(values):
        return sum(values) / len(values) if values else float("nan")

    check("teacher_logprobs is token-aligned in every update", aligned,
          f"{len(teacher_all)} scored tokens over {len(updates)} updates")
    print(f"    teacher mean logprob   {mean(teacher_all):+.4f}")
    print(f"    student mean logprob   {mean(behavior_all):+.4f}")
    print(f"    per-token reverse KL   {mean(kl_all):+.4f} nats")
    return {"tokens_scored": len(teacher_all),
            "teacher_mean": round(mean(teacher_all), 5),
            "student_mean": round(mean(behavior_all), 5),
            "reverse_kl": round(mean(kl_all), 5)}


def completion_report(run, updates):
    """How the completions ended, per update — the scaffold's own vital sign.

    A high `length` share means the model never emitted eos and ran to
    max_tokens, which is both the expensive case (every extra token is a
    token the teacher must prefill) and the case where the verifier reads a
    number the model invented after answering."""
    from collections import Counter

    finish = Counter()
    generated = []
    for u in updates:
        for row in run.read_wave(u):
            for turn in row["turns"]:
                finish[turn["finish"]] += 1
                generated.append(len(turn["token_ids"]))
    print(f"    finish reasons         {dict(finish)}")
    print(f"    generated tokens       mean "
          f"{sum(generated) / len(generated):.0f}  max {max(generated)}")
    return {"finish": dict(finish),
            "mean_generated": round(sum(generated) / len(generated), 1),
            "max_generated": max(generated)}


def three_host_run(spec, *, label, n_updates):
    """The run itself: the learner's container drives, both pools are somebody
    else's metal, and the teacher's wire is on a clock."""
    import asyncio
    import time

    import torch

    from rlstack import ModalVolumeStore, RemotePool
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.host import Host, Partition, Regime
    from rlstack.runner.learners.fsdp_torch import lead_fsdp_learner
    from rlstack.runner.loop import experiment_identity

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    schema = hf_schema(STUDENT)

    student_pool = RemotePool(ModalTransport(StudentHost()), base=STUDENT, tp=2)
    clock = ScoreClock(ModalTransport(TeacherHost()))
    teacher_pool = RemotePool(clock, base=TEACHER, tp=4)
    print(f"[pins] torch={torch.__version__} "
          f"cuda_devices={torch.cuda.device_count()}")
    warmed = warm({"student-8b-tp2": student_pool,
                   "teacher-32b-tp4": teacher_pool})

    learner = lead_fsdp_learner(2)
    host = Host(f"modal-math-opd-{label}", engines=(), learner=learner,
                store=store, partition=Partition("modal-l4", (0, 1), 0.90),
                regimes=(Regime("learner-fsdp2", "training", STUDENT, 2),))
    print(f"[chorus] rank 0 of 2, learner.fsdp={learner.fsdp}")

    started = time.time()

    rid = experiment_identity(spec, schema)
    print(f"[run] {rid} — {n_updates} updates", flush=True)

    async def submit_with_instruments():
        watch = LedgerWatch(store, rid, started, clock)
        watching = asyncio.get_running_loop().create_task(
            watch.watch_until_cancelled())
        stats = asyncio.get_running_loop().create_task(host.run_stats(30.0))
        try:
            report = await host.submit(spec, schema,
                                       remotes={"main": student_pool,
                                                "teacher": teacher_pool})
        finally:
            watching.cancel()
            stats.cancel()
        return report, watch.timing

    try:
        report, timing = asyncio.run(submit_with_instruments())
    finally:
        learner.stop()
    elapsed = time.time() - started
    store_volume.commit()

    run = store.open_run(report.run_id)
    entries = run.read_ledger()
    print(f"\nrun_id={report.run_id}  updates={report.updates_completed}  "
          f"wall={elapsed:.0f}s (+{warmed:.0f}s warming)")
    for entry in entries:
        print(f"  update {entry['update']}: reward {entry['post']['reward']:.3f}"
              f"  loss {entry['train']['loss']:+.4f}"
              f"  ratio {entry['train']['mean_ratio']:.4f}"
              f"  gap {entry['train']['logprob_gap']:.4f}"
              f"  grad {entry['train']['grad_norm']:.3f}"
              f"  tokens {entry['train']['tokens']}")

    # the per-update lines were printed as they landed (LedgerWatch); this is
    # the average, which is the number the scorer-daemon decision turns on
    if timing:
        print(f"\n[inline teacher scoring] mean per update: span "
              f"{sum(r['span_s'] for r in timing) / len(timing):.1f}s of "
              f"{sum(r['update_s'] for r in timing) / len(timing):.1f}s, "
              f"teacher busy {sum(r['busy_s'] for r in timing) / len(timing):.1f}s "
              f"over {sum(r['tokens'] for r in timing) // len(timing)} tokens")

    updates = [int(e["update"]) for e in entries]
    check("ledger complete", updates == list(range(1, n_updates + 1)),
          f"{len(updates)}/{n_updates}")
    gaps = [e["train"]["logprob_gap"] for e in entries]
    check("logprob_gap at the kernel floor", bool(gaps) and max(gaps) < 0.15,
          f"max {max(gaps):.4f}" if gaps else "no updates")
    rewards = [e["post"]["reward"] for e in entries]
    check("the scaffold produces reward under the behavior policy",
          bool(rewards) and rewards[0] > 0.0,
          f"update 1 {rewards[0]:.3f}" if rewards else "no updates")
    losses = [e["train"]["loss"] for e in entries]
    check("update 1's KL is the bare 8B-vs-32B distance", bool(losses),
          f"{losses[0]:+.4f} nats/token" if losses else "no updates")
    column = teacher_column_report(run, updates)
    completions = completion_report(run, updates)
    check("evals present",
          all(run.has_eval(u) for u in updates
              if u % spec.eval.every == 0))
    for u in updates:
        if run.has_eval(u):
            print(f"    eval@{u}  {run.read_eval(u, 'summary.json')}")

    print("\n[learner host]", host.status())
    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    print(f"\n[{label} checks] {sum(ok for _, ok, _ in CHECKS)} passed, "
          f"{len(failed)} failed: {failed}")
    return {"run_id": report.run_id, "wall_s": round(elapsed, 1),
            "warm_s": round(warmed, 1),
            "rewards": rewards, "losses": losses,
            "max_gap": max(gaps) if gaps else None,
            "scoring": timing, **column, **completions, "failed": failed}


# ---------------------------------------------------------------------------
# leg 3: the shakeout — the few updates that answer the questions the full
# run would otherwise discover expensively
# ---------------------------------------------------------------------------

@app.function(image=image, gpu="L4:2", volumes=VOLUMES, timeout=14400,
              cpu=8.0, memory=65536)
def shakeout(n_updates: int = 8, eval_every: int = 4, style: str = "cot",
             master: int = 91, max_tokens: int = MAX_TOKENS,
             microbatch_tokens: int = 512) -> dict:
    """A short run of the real three-host thing, for the five questions.

        the window    does an 8B tp=2 sampler and a 32B tp=4 teacher both
                      serve at max_model_len=2048, with a prompt of ~450
                      tokens and 512 generated?
        the scaffold  is the untrained student's verifier reward > 0 — i.e.
                      does this prompt make MATH a task the run can score?
        the clock     how many seconds per update does INLINE teacher scoring
                      cost, and what share of the update is it?
        the signal    what is the reverse KL at update 1 (the bare 8B-vs-32B
                      distance on these completions)?
        the rails     is logprob_gap still at the kernel floor when the
                      documents are 40x longer than #47's?

    Eight updates, not ten: eight L4s bill for the whole run, and the answers
    above are all visible by update three — the budget is better spent holding
    a retry in reserve, because first contact usually finds something.
    """
    from rlstack import ModalVolumeStore

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    spec = math_opd_spec(store, style=style, n_updates=n_updates,
                         n_train=512, n_heldout=32, eval_every=eval_every,
                         master=master, max_tokens=max_tokens,
                         microbatch_tokens=microbatch_tokens)
    store_volume.commit()
    return three_host_run(spec, label="shakeout", n_updates=n_updates)


# ---------------------------------------------------------------------------
# leg 4: the full run — BUILT, GATED. Samarth's explicit go only.
# ---------------------------------------------------------------------------

@app.function(image=image, gpu="L4:2", volumes=VOLUMES, timeout=28800,
              cpu=8.0, memory=65536)
def full(n_updates: int = 60, style: str = "cot", master: int = 47,
         max_tokens: int = MAX_TOKENS, microbatch_tokens: int = 512,
         go: bool = False) -> dict:
    """The Phase A run: 50-100 updates, lag=1, eval every 10 on held-out MATH.

    PRE-REGISTERED SUCCESS (the plan, unchanged): reverse KL trends DOWN;
    verifier reward does not degrade against the teacher-baseline probe; the
    gap rail stays at the kernel floor. `loss` in the ledger IS the per-token
    reverse KL, so the first criterion reads straight off the ledger.

    THE GATE IS DELIBERATE. Eight L4s live for hours is the most expensive
    thing in this repo, and the run is only worth starting once the shakeout's
    numbers say the scaffold works and the update count is chosen against the
    measured per-update wall time. Pass --go when that decision is made:

        modal run deploy/math_opd_l4.py::full --go --n-updates 60

    WHAT THE SHAKEOUT PRICES IT AT: ~100s per update, so 60 updates is about
    1h45 of eight L4s and 100 updates about 3h. n_heldout is 64 here and the
    evaluator walks its tasks SEQUENTIALLY (one wave of n_samples at a time,
    rlstack/runner/daemons/evaluator.py) — mid-run evals hide inside training,
    but the LAST one is a tail of roughly n_heldout x 20s with nothing left to
    overlap. Drop n_heldout to 32 to halve that tail.
    """
    from rlstack import ModalVolumeStore

    if not go:
        raise RuntimeError(
            "deploy/math_opd_l4.py::full is gated: it burns eight L4s for "
            "hours. Read ::shakeout's per-update timing, choose n_updates "
            "against it, then re-run with --go.")
    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    spec = math_opd_spec(store, style=style, n_updates=n_updates,
                         n_train=512, n_heldout=64, eval_every=10,
                         master=master, max_tokens=max_tokens,
                         microbatch_tokens=microbatch_tokens)
    store_volume.commit()
    return three_host_run(spec, label="full", n_updates=n_updates)


@app.function(image=image, volumes=VOLUMES, timeout=1800, cpu=4.0,
              memory=16384)
def report(n_updates: int = 6, eval_every: int = 3, style: str = "cot",
           master: int = 91, max_tokens: int = MAX_TOKENS,
           microbatch_tokens: int = 512) -> dict:
    """Read a finished run back out of its store, on a CPU container.

    IDENTITY IS COMPUTED, NEVER TYPED (I3): this takes the same arguments the
    run took and DERIVES the same run_id, so no id has to be copied from a log
    — which is the whole reason it exists. The first six-update shakeout's
    driver was killed during teardown and its summary never reached the log;
    everything in that summary except the driver's own stopwatch was already
    sealed in the store, and this is how it comes back.
    """
    from rlstack import ModalVolumeStore
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.loop import experiment_identity

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    spec = math_opd_spec(store, style=style, n_updates=n_updates,
                         n_train=512, n_heldout=32, eval_every=eval_every,
                         master=master, max_tokens=max_tokens,
                         microbatch_tokens=microbatch_tokens)
    rid = experiment_identity(spec, hf_schema(STUDENT))
    print(f"[run] {rid}")

    entries = store.peek_ledger(rid)
    for entry in entries:
        print(f"  update {entry['update']}: reward {entry['post']['reward']:.3f}"
              f"  loss {entry['train']['loss']:+.4f}"
              f"  ratio {entry['train']['mean_ratio']:.4f}"
              f"  gap {entry['train']['logprob_gap']:.4f}"
              f"  grad {entry['train']['grad_norm']:.3f}"
              f"  tokens {entry['train']['tokens']}"
              f"  microbatches {entry['train']['microbatches']}")
    updates = [int(e["update"]) for e in entries]
    run = store.open_run(rid)
    column = teacher_column_report(run, updates)
    completions = completion_report(run, updates)
    for summary in store.peek_eval_summaries(rid):
        print(f"    eval@{summary['update']}  {summary}")
    node = [c for c in store.peek_dictionary(rid)["columns"]
            if c["name"] == "teacher_logprobs" and c["phase"] == "post"][0]
    check("the run describes its own teacher channel",
          node["granularity"] == "token" and node["feeds_loss"]
          and "loss:opd" in node["consumers"], json.dumps(node))
    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    return {"run_id": rid, "updates": len(updates),
            "rewards": [e["post"]["reward"] for e in entries],
            "losses": [e["train"]["loss"] for e in entries],
            "gaps": [e["train"]["logprob_gap"] for e in entries],
            "evals": store.peek_eval_summaries(rid),
            **column, **completions, "failed": failed}


@app.function(image=image, volumes=VOLUMES, timeout=5400, cpu=4.0,
              memory=16384)
def score_clock(updates: str = "3,6", n_updates: int = 6, eval_every: int = 3,
                style: str = "cot", master: int = 91,
                max_tokens: int = MAX_TOKENS, microbatch_tokens: int = 512
                ) -> dict:
    """Time the INLINE teacher scoring of a finished run's real waves, with
    only the teacher host live. The measurement this file exists to take.

    It is not a simulation of the post phase, it IS the post phase: the sealed
    wave is read back out of the store, and run_pipeline drives the registered
    `teacher_logprobs` processor over it through a RemotePool — the same
    groups, the same flatten-order walk, the same one-prefill-per-turn against
    the same 32B. The only thing missing is the student and the learner, and
    the teacher is dedicated during the real post phase anyway, so nothing
    contends for it there either.

    Four L4s for four minutes instead of eight for forty: the cheap way to
    ask what an update's gradient waits for, and the input to whether the
    async scorer daemon is worth building.
    """
    import asyncio
    import time

    from rlstack import (
        Bundle, ModalVolumeStore, RemotePool, SamplingSpec, run_pipeline,
        wave_from_rows,
    )
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.loop import experiment_identity

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    spec = math_opd_spec(store, style=style, n_updates=n_updates,
                         n_train=512, n_heldout=32, eval_every=eval_every,
                         master=master, max_tokens=max_tokens,
                         microbatch_tokens=microbatch_tokens)
    rid = experiment_identity(spec, hf_schema(STUDENT))
    run = store.open_run(rid)
    print(f"[run] {rid}")

    clock = ScoreClock(ModalTransport(TeacherHost()))
    teacher = RemotePool(clock, base=TEACHER, tp=4)
    bundle = Bundle("bundle:score-clock", {})
    teacher.add_bundle(bundle)
    routes = {"main": (teacher, bundle), "teacher": (teacher, bundle)}
    sampling = SamplingSpec(temperature=1.0, top_p=1.0, max_tokens=max_tokens)

    rows = []
    for update in [int(u) for u in updates.split(",")]:
        wave = wave_from_rows(run.read_wave(update))
        before = len(clock.scores)
        started = time.time()
        columns = asyncio.run(run_pipeline(
            ("teacher_logprobs",), wave, routes, sampling, master, update))
        span = time.time() - started
        calls = clock.scores[before:]
        scored = sum(len(v) for v in columns["teacher_logprobs"])
        latencies = sorted(round(c["end"] - c["start"], 2) for c in calls)
        print(f"[scoring] update {update}: {len(wave.groups)} groups x "
              f"{len(wave) // len(wave.groups)} trajectories, {len(calls)} "
              f"prefills, {scored} scored tokens")
        print(f"    span {span:.1f}s   teacher busy "
              f"{sum(c['end'] - c['start'] for c in calls):.1f}s   "
              f"per prefill min {latencies[0]}s median "
              f"{latencies[len(latencies) // 2]}s max {latencies[-1]}s")
        rows.append({"update": update, "span_s": round(span, 1),
                     "prefills": len(calls), "scored_tokens": scored,
                     "busy_s": round(sum(c["end"] - c["start"]
                                         for c in calls), 1),
                     "median_prefill_s": latencies[len(latencies) // 2]})

    mean_span = sum(r["span_s"] for r in rows) / len(rows)
    check("inline teacher scoring is a measurable share of an update",
          mean_span > 0, f"mean span {mean_span:.1f}s per update")
    print(f"\n[teacher host] {TeacherHost().status.remote()}")
    return {"run_id": rid, "updates": rows, "mean_span_s": round(mean_span, 1),
            "failed": [(n, d) for n, ok, d in CHECKS if not ok]}


@app.function(image=image, timeout=600)
def describe() -> dict:
    """What the two serving partitions advertise, fetched over the wire."""
    return {"teacher": TeacherHost().status.remote(),
            "student": StudentHost().status.remote()}


@app.local_entrypoint()
def main() -> None:
    """No default leg: every one of them spends real money, in one order."""
    print(__doc__.split("\n\n")[1])
    print("Run one leg at a time; ::full needs --go.")
