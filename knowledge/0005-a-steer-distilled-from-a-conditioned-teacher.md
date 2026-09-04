# ADR 0005 — A steering vector distilled from a prompt-conditioned teacher: the trajectory set as content, the conditioned-teacher channel, and a window the bank declares

| | |
|---|---|
| **Date** | 2026-09-04 |
| **Status** | Proposed |
| **Author** | Claude Fable 5.1 (session: the SPAR introspection paper, 2026-09-04) |
| **Touches** | `data/tasks/` (one new dataset file, one new content verb), `inference/environments/` (two one-file environments), `training/post/` (two processors in the existing shape), `policy/adapters/steer.py` + `steer_torch.py` + `steer_vllm.py` (the window: a symbolic edge, a bank-declared default), `policy/adapters/rollout.py` (`Request.prompt_len`), `policy/adapters/replay.py` (`ReplayRows.prompt_lens`), `runner/learners/torch_learner.py` (threads it, adapter-blind), `runner/engines/vllm_engine.py` + `runner/fakes.py` (fill `prompt_len`), `runner/measure.py` (a measurement may route to a second pool), `spec/specs.py` (sugar), `deploy/concept_steer.py` (the venue), `tests/` |
| **Invariants** | I3 (the hint and the window are content and spec, so they hash), I6 (a replayed row the policy never sampled has no record — what fills it is declared, never guessed), I9 (a processor scores through a declared pool under privileged conditioning — hinted + teacher, combined), I5 (a 32B student is new metal territory; the spec says nothing about where) |
| **CONTEXT** | extends #40 (the scoring verb, hinted logprobs), #47 (the teacher is a pool), #60 (a dataset becomes content), #70 (measurement outside the run), #77 / ADR 0004 (the steer; Q2's "no directive = every position" becomes "no directive = the bank entry's window, default every position") |

## Original prompt

> ok read this pdf. one problem that i had with this research is that it's unclear whether the model is doing some sort of post-hoc reasoning on the final residual, instead of actually doing some internal computation for the layer experiment.
>
> here is one way we can test this hypothesis: we can instead make the loss objective a distillation objective wrt a reference model conditioned on a system prompt to talk about a specific concept. let's keep "happiness" for now. so the system prompt is going to talk about happiness, and we are going to train the residual stream vector to output a logit distribution similar to that of this conditioned model (just via SFT). do how many ever SFT examples are necessary (you can be the judge of that), im thinking we can do 1000 token long trajectories over a variety of different sorts of prompts. let's try this on qwen32b, since that's the model which performed very well in our benchmarks.
>
> so what i mean here, is train a residual stream vector injected at specific anchor layers, to output logit distirbution similar to reference, and then see if the model can still learn via ICL (meaning it's doing some sort of ICL in intermediate layers).
>
> if the residual stream approach doesn't work, then we can try doing small LoRA adapters at the specified layers instead, but let's try this first.
>
> could you create a plan which matches up with the primitive structures of this codebase?

The paper is "Reasoning and learning about injected concepts in language
models" (SPAR). Its Experiment 1 injects a mean-difference concept vector at an
early / middle / late layer (Qwen3-32B: layers 10 / 32 / 54 of 64) at every
token of the user turn during prefill, and asks the model in-context which
region it was; Qwen3-32B reaches near-perfect p(correct) with enough examples
and generalizes to unlabeled layers. The paper's own "proving real privileged
access" direction names the confound this ADR is built to test: the model may
be reading an induced property of its FINAL hidden state rather than anything
about where the intervention happened.

## Context / problem

**The hypothesis and the test.** A mean-difference vector is an off-manifold
perturbation whose footprint at the final residual may itself be
layer-dependent (how far the stream was pushed, how much of it the later
layers repaired). A vector TRAINED so that the steered, unconditioned model
matches the output distribution of the same model conditioned on a happiness
system prompt is constrained at its output to look like something a real
prompt would do. If the model can still classify the injection layer
in-context with such vectors, the layer signal is internal; if the ability
collapses, Experiment 1 was reading the artifact. Everything the test needs
is: three trained vectors (one per anchor layer), a number that says the
distillation worked, and the paper's own ICL harness run over the trained
vectors instead of the extracted ones.

**What exists.** The steer adapter type is the residual lever (ADR 0004,
#77): one `d`-wide vector per matched boundary `resid_pre.<n>`, zero as its
exact identity, served by the engine image's hook and replayed by
`SteerSite`; TP=2 and FSDP=2 proven at 0.6B (#78). The loss zoo has the two
objectives this experiment can use: `sft` ("behavior cloning on the sealed
record: the data source IS the curriculum", `training/losses/sft.py`) and
`opd` (sampled-token reverse KL against a `teacher_logprobs` column,
`training/losses/opd.py`). The teacher channel is built: `teacher_logprobs`
scores the student's own draws through a "teacher" pool (#47, 8B ← 32B on
three hosts), and `hinted_logprobs` scores through a pool under a hint the
sampler never saw (#40, `hint_for` reads `task.meta["hint"]`). The plan
already has the leaf for training on trajectories sealed elsewhere:
`Replay("cas://<sha>#<i>")` names a row of "a fixed file of sealed trajectory
rows" (`runner/refs.py:9`), and `Plans.rollout=None` means "the run samples
nothing". Measurement lives outside the run (#70, `runner/measure.py`). A
32B teacher pool at tp=4 on L4:4 fits with room to prefill (#47).

**What does not exist, file and line.**

1. *Nothing writes a trajectory set.* `refs.py:97` reads `cas://<sha>` as
   jsonl rows and `write_tasks` (`data/tasks/base.py:27`) writes a TASK set,
   but no verb turns a sealed `Wave` into the cas file a Replay leaf reads —
   the inverse of the reader was never needed, because every replay so far
   pointed into another run's `waves/`. A generation-only run cannot make
   one either: `loop.py:115` refuses `algo=None` (the Sealer daemon is an
   open thread).
2. *No environment samples from a non-policy pool under a hint and seals the
   hint OUT.* `math_single_turn` samples the policy; `teacher_logprobs` and
   `hinted_logprobs` SCORE. The teacher's completions have to be sampled
   with the system prompt in context and sealed with the user prompt alone,
   so the student trains on exactly what it will be asked at eval.
3. *The steer's replay refuses a row that recorded no window.*
   `steer_torch.py:recorded_window` raises "every turn served by a steer
   bundle records the window it applied" — correct for the policy's own
   rollouts, and exactly wrong for a Replay leaf into a trajectory set the
   TEACHER sampled: those turns pin `bundle:base:teacher`, carry no
   `steer_window`, and the student must still replay them WITH the vector
   applied somewhere. Where is not a fact (nothing was recorded), it is a
   choice, and today no primitive holds that choice.
4. *"Prompt-only" is not sayable without a tokenizer.* `SteerWindow(start,
   end)` speaks real-token coordinates; the paper injects at the user turn's
   tokens only, so the window is `[0, len(prompt))` per request, and an
   environment holds the prompt's TEXT — `PoolClient` has no tokenize verb,
   and the steer venue's probe tokenized by hand (`deploy/steer_l4.py:666`).
5. *A measurement routes to one pool.* `measure.py:109` builds
   `routes = {"main": (pool, bundle)}`, so a measurement whose post pipeline
   scores through "teacher" would KeyError at the first group.
6. *No teacher processor takes a hint.* `teacher_logprobs` walks the sealed
   stream under the same conditioning the student had; `hinted_logprobs`
   prepends the hint but scores through "main". The conditioned teacher is
   the one walk with both — six lines, but a processor of its own by the one-
   file rule.
7. *No prompt corpus.* The only task sets are DAPO-Math (#60). The teacher
   needs "a variety of different sorts of prompts" that draw ~1000-token
   answers, chat-formatted for Qwen3 with thinking off, and carrying the
   system prompt as content.

**Unmeasured.** The 32B as a STUDENT: an fsdp=2 learner over a frozen 32B has
never been built (#78 leaves "larger bases" untried); the throughput cost of
eager mode on a 32B steer-serving build (#77 names it unmeasured); and how
many teacher samples a 5120-wide vector needs to saturate.

## Decision

Two runs of machinery, in the repo's vocabulary. First, **the teacher's
trajectory set becomes content**: a `concept_prompts` task set whose tasks
carry the user prompt (chat-templated, no system block, thinking off) and the
happiness system block as `meta["hint"]`; a `conditioned_teacher` environment
that samples the "teacher" pool under hint + prompt and seals prompt + turn;
`write_trajectories` putting the sealed wave in the CAS as the jsonl the ref
reader already reads; a venue door that runs `sample_wave` over the corpus
through a bare Qwen3-32B and prints the uri. Second, **the student is an
ordinary SFT run**: `Plans.rollout=None`, a train plan of `Replay` leaves into
that uri, loss `sft`, an empty post pipeline, a bank of ONE steer entry at ONE
anchor boundary (`resid_pre.10` / `.32` / `.54`, `d=5120`), one run per anchor,
the same plan bytes in all three so only the bank differs. The steer gains
what item 3–4 above need: a **bank-declared window** (`init["window"]`, one of
`all` / `prompt` / `completion`, default `all` — absent from `init` when
default, so every existing steer spec keeps its identity), which the rollout
lowering applies when the caller passes no directive and the replay lowering
applies when a row recorded no window; and `SteerWindow` gains a symbolic
edge (`Edge.PROMPT`) resolved by the bus against `Request.prompt_len`, so
"the user turn" is sayable without a tokenizer on either side. The
distillation is **measured outside the run** (#70): a Measurement whose env
samples the student on held-out prompts (the entry's window applies by
default) and whose pipeline is `conditioned_teacher_logprobs` → `reverse_kl`,
through a `measure_run` that routes "teacher" beside "main" on the same
engine under the base bundle. The **ICL test is the paper's harness**, fed
the exported vectors (`adapters/<name>@<v>.bin` is already safetensors keyed
by boundary path); rlstack serves one window per request and the k-shot
prompt needs k different injections, which is a directive rule this ADR does
not change (Q10). On-policy distillation (`opd` + the same conditioned-teacher
processor, a live rollout plan) is the second arm, one spec edit away, and
the fallback if SFT's target proves too loose.

### Touched / untouched

- **Touched** — `rlstack/data/tasks/concept_prompts.py` (new): one function,
  `concept_prompt_tasks(concept, base)`, reading an instruction corpus (Q6)
  into Tasks: `id = "<corpus>/<row id>"`, `prompt` = the Qwen3 template over
  ONE user message with `enable_thinking=False` and NO system block, `meta =
  {"hint": <the template's rendering of the system message alone>,
  "concept": "happiness", "category": ...}`. The builder ASSERTS at build
  time that `render([system, user]) == meta["hint"] + prompt` on this
  tokenizer, so hint + prompt concatenated raw (the engine's v0 prompt rule,
  #60) is byte-for-byte what the chat template would have produced. Heavy
  imports inside the function (rule 7); registered in `__main__.BUILDERS`.
- **Touched** — `rlstack/data/tasks/base.py`: `write_trajectories(store,
  wave) -> str` beside `write_tasks` — `wave_to_rows` as canonical jsonl,
  `cas_put`, the sha is the set's identity. The reader is `refs.py`'s CAS
  branch, unchanged (Q4 asks whether the verb is filed right).
- **Touched** — `rlstack/inference/environments/conditioned_teacher.py`
  (new): `sample` the "teacher" pool with `[hint_for(task), prompt]`, return
  `Rollout(task, messages=[prompt, turn.message], turns=[turn],
  env_extras={"hint": hint.content})`. The hint is in the record as
  provenance and out of the message stream by construction, so `flatten`
  never tokenizes it into the student's document.
- **Touched** — `rlstack/inference/environments/single_turn.py` (new): the
  one-sample environment under an honest name — `math_single_turn`'s body is
  exactly this, but its name says math and this corpus is not, and a word
  used with a different meaning is a finding (ARCHITECTURE.md). The old name
  stays for the runs that hash it.
- **Touched** — `rlstack/training/post/conditioned_teacher_logprobs.py`
  (new): `teacher_scores`' walk with `hint_for(traj)` at the head of the
  context — `produces = token_level = ("teacher_logprobs",)`, `pools =
  ("teacher",)`. The SAME column name as `teacher_logprobs`, so `opd`
  requires it unchanged (a different conditioning is a different processor,
  never an edit to a loss — #47's rule, applied to the hint).
- **Touched** — `rlstack/training/post/reverse_kl.py` (new): pool-less,
  `consumes = ("teacher_logprobs",)`, `produces = ("reverse_kl",)`: per
  trajectory, the mean over generated tokens of (recorded behavior logprob −
  teacher logprob) — the one-sample reverse-KL estimate, the number the
  measurement reports and the number `opd`'s ledger `loss` already is.
- **Touched** — `rlstack/policy/adapters/steer.py`: `Edge` (a `StrEnum`,
  `PROMPT`), `SteerWindow.start/end: int | Edge`, `Window` (the bank-declared
  default: `ALL` / `PROMPT` / `COMPLETION`, a `StrEnum` in `init`),
  `resolve_window` resolving an edge against `request.prompt_len` and
  defaulting to the entry's window; `Steer.record_directive` unchanged in
  shape (records the resolved numbers). `spec/specs.py`: `steer(..., window=
  Window.ALL)` writes `init["window"]` ONLY when not ALL.
- **Touched** — `rlstack/policy/adapters/steer_vllm.py`: `apply` passes the
  entry's window as the default directive (read off the attached bundle's
  entry init — `attach` receives the payloads today, so the init rides in
  the payload's safetensors metadata, or `attach` grows the entry's init;
  Q3).
- **Touched** — `rlstack/policy/adapters/steer_torch.py`: `window_mask`
  reads the record when a row has one and the entry's `Window` resolved
  against `rows.prompt_lens[row]` when it has none. The "recorded nothing"
  refusal moves to the seal side (a test pins that both buses record for
  every bundle carrying a steer); the "two windows in one row" refusal stays.
- **Touched** — `rlstack/policy/adapters/rollout.py`: `Request.prompt_len:
  int` (sample: `len(token_ids)`; score: the context's length, so an edge
  under score traffic means the same thing). `replay.py`:
  `ReplayRows.prompt_lens: tuple[int, ...] | None`, adapter-blind like
  `facts`. `runner/learners/torch_learner.py:_rows_of` threads it from the
  batch (each document's first `loss_mask == 1` position, off `doc_starts`).
  `runner/engines/vllm_engine.py:_levers_for` and `runner/fakes.py` fill
  `prompt_len`.
- **Touched** — `rlstack/runner/measure.py`: `measure_run(..., pools:
  Mapping[str, Engine] = {})` — every extra name routes to its engine under
  a payload-free base bundle, exactly `loop.py`'s `base_bundles` rule, so
  "teacher" on the same engine as "main" is one dict entry. `Measurement`
  gains nothing.
- **Touched** — `deploy/concept_steer.py` (new venue, `stress_fleet.py`'s
  desk shape): doors `prompts` (build the task set), `distill_set` (the
  teacher's trajectory set through the standing serving host, prints the
  uri), `train --layer --window` (submit one arm through the desk), `measure`
  (one `measure_run` pass, by hand or cron), `export --run_id --version`
  (the vector's safetensors to the volume for the paper's harness), and
  `sweep` / `status` as every venue has. Everything semantics-bearing is in
  the spec (I5).
- **Untouched** — `training/losses/sft.py` and `opd.py`: the objectives are
  exactly right as written; SFT's rails on foreign rows read as the distance
  to the teacher (see non-promises), and `opd` requires the column the new
  processor produces. `training/post/teacher_logprobs.py` and
  `hinted_logprobs.py`: the new processor is beside them, not inside them.
  `data/plan.py`, `runner/refs.py`, `runner/assemble.py`: the Replay leaf
  and the CAS ref were built for this and need nothing. `runner/loop.py`,
  `runner/daemons/*`: an SFT run with no rollout plan and an empty post
  pipeline plans a Trainer alone, byte-for-byte the daemon it was; the loop
  still requires a "main" engine (reachability, `tokenize` for injected
  spans, `add_bundle`), which is why the topology declares one even though
  the SFT run never samples. `rlstack_engine/steer.py` + `steer_worker.py`:
  the hook reads `[start, end)` from `extra_args` and does not care who
  resolved them. `spec/validate.py`: no new gate — the window enum is
  refused where it is parsed, in the adapter type. `data/flatten.py`: the
  hint never reaches it.

### Promises / non-promises

- **Promises** — (1) A steer spec written before this ADR has the same
  `run_id` after it: `init` gains no key unless `window` is set. (2) An SFT
  run over a trajectory set is resume-equivalent: the plan is Replay leaves,
  the rows are content, the row plan and the window are pure functions of
  (spec, rows), so a killed-and-resumed run's directory is byte-identical to
  a straight one — `tests/test_resume.py` gains this shape. (3) The fakes
  suite is green; the torch-gated steer tests gain: a row with no record
  replays under the entry's window, a row with a record still replays the
  record, `PROMPT` masks exactly the positions before the first generated
  token, `Edge.PROMPT` resolves to the same number on the fake bus and in
  `SteerSite`. (4) The parity probe on Qwen3-0.6B (`steer_l4::probe`)
  reproduces #77's numbers with the default window, and a `PROMPT`-window
  bank agrees engine-vs-trainer within the kernel floor on a windowed
  prompt. (5) A trajectory set's uri is its identity: rebuilding the SAME
  rows returns the same uri; the SFT spec pins it by uri, so the fingerprint
  names what was trained on (I3). (6) The measurement writes
  `measurements/<run_id>/distill/` and the run directory gains not one byte
  (#70's rule, pinned). (7) Every venue door that acquires metal ends with
  the desk's `release` and the plane asserted empty (ADR 0003 / #77).
- **Non-promises** — The trajectory set is NOT byte-reproducible across
  engine batch compositions: vLLM's seeded sampling is deterministic per
  request only up to batch noise (#46 saw it), so regenerating the set may
  give different text; the uri pins what WAS sampled, which is what identity
  needs. The SFT ledger's `logprob_gap` on foreign rows is NOT a parity
  alarm: `behavior_logprobs` are the TEACHER's under the hint, so the rail
  reads mean |student − teacher| — the distillation distance, which is
  useful, and the observer will still label it as the parity rail (stated,
  not renamed). No claim about the ICL result: this ADR builds the vectors
  and the harness inputs; the experiment's outcome is the paper's harness's
  to report. The 32B learner's memory and pace are UNMEASURED until the
  shakeout; the eager-mode serving cost too. Nothing here proves a steer
  under tp=4 (#77 names tp>1 unproven; #78 proved tp=2 at 0.6B). The
  "recorded nothing under a steer bundle" alarm in `window_mask` is
  deliberately retired (promise 3 replaces it with a seal-side test).

### Interfaces

- **Content:** `cas://<sha>` for the task set (declared in `GenSpec.tasks`
  only by runs that sample — the SFT run declares `gen=None`); `cas://<sha>`
  for the trajectory set, named by `Replay("cas://<sha>#<i>")` leaves in the
  train plan, checked resolvable at the submit gate as every CAS ref is.
- **Pools:** the SFT run declares `main` (idle but required); the
  measurement and the on-policy arm route `main` (the student's bundle) and
  `teacher` (`bundle:base:teacher`, payload-free) to ONE engine object — the
  Routes contract already says "one engine may back many pool names".
  Through the desk, a second inference demand with the same (base, tp)
  JOINS the same listing (`covers` is capability equality).
- **The directive rule (ADR 0004 Q2, extended):** no directive = the bank
  entry's declared window (default `ALL`, which is what "no directive" meant
  before); a directive overrides per request; what applied is RECORDED
  either way; replay reads the record, else the declaration. The bus's
  `Request` carries `prompt_len` so an edge resolves on the engine side and
  the recorded numbers are the truth replay reads.
- **The gate:** nothing new; `check_sites_reachable_on` already holds the
  steer's boundary against the main engine's inventory, so the SFT run's
  serving host must be a steer-serving build (eager, our worker class) even
  though it only tokenizes — a fact of the spec, not of the venue.
- **`observe/`:** the SFT run's ledger has `loss` (the cross-entropy against
  the teacher's samples) and `logprob_gap` (mean |student − teacher|); the
  forward-KL proxy is `loss − (−mean behavior_lp)`, both already columns.
  `measurements/<run_id>/distill/points` carries `reverse_kl` and
  `teacher_logprobs` means per measured version, rendered as a dashed series
  like every measurement.
- **Export:** `adapters/<name>@<v>.bin` is `steer_torch.emit`'s safetensors
  keyed by boundary path (`model.layers.10`); the `export` door copies it
  out. The paper's harness adds the tensor under that key at the output of
  `model.layers[10]`, which is what `resid_pre.10` names (Q8 confirms the
  convention).

### Sketches

```python
# policy/adapters/steer.py — the window a request asks for, and the one a bank declares
class Edge(StrEnum):
    PROMPT = "prompt"          # the request's prompt length, resolved by the bus

class Window(StrEnum):
    ALL = "all"                # every position (ADR 0004's default, unchanged)
    PROMPT = "prompt"          # [0, prompt_len): the user turn — the paper's protocol
    COMPLETION = "completion"  # [prompt_len, end)

@dataclass(frozen=True)
class SteerWindow(Directive):
    adapter_type = "steer"
    start: int | Edge = 0
    end: int | Edge | None = None

def resolve_window(directive: SteerWindow | None, default: Window,
                   request: Request) -> tuple[int, int | None]:
    """The request's window in ENGINE coordinates: the directive if the caller
    passed one, else the bank entry's declared window; edges resolve against
    request.prompt_len; then the occupied offset — and a window that cannot
    fit is refused with the request named, never clamped."""

# spec/specs.py — sugar; `window` lands in init ONLY when not ALL (identity-stable)
def steer(site, d, tie=False, init_std=0.0, window: Window = Window.ALL) -> AdapterSpec: ...

# policy/adapters/replay.py — one more adapter-blind row fact
@dataclass(frozen=True)
class ReplayRows:
    slots: tuple[Mapping[str, Any], ...]
    index: torch.Tensor
    facts: tuple[tuple[Mapping[str, Any], ...], ...] | None = None
    prompt_lens: tuple[int, ...] | None = None   # row r's first generated position

# data/tasks/base.py — the trajectory set, write_tasks' sibling
def write_trajectories(store: Store, wave: Wave) -> str:
    """Put a sealed wave in the CAS as the jsonl a Replay("cas://<sha>#<i>")
    leaf reads; the sha IS the set's identity."""

# inference/environments/conditioned_teacher.py
@environment("conditioned_teacher")
class ConditionedTeacher(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        turn = await client.pool("teacher").sample([hint_for(task), prompt])
        return Rollout(task=task, messages=[prompt, turn.message], turns=[turn],
                       env_extras={"hint": task.meta["hint"]})

# the student, one arm: everything semantics-bearing, in one value
ExperimentSpec(
    policy=PolicySpec(base="Qwen/Qwen3-32B",
                      bank={"v": steer("resid_pre.10", d=5120, window=Window.PROMPT)}),
    gen=None,
    plans=Plans(train=<cas: 64 waves x 32 Replay("cas://<set>#i") leaves>, rollout=None),
    algo=AlgoSpec(loss="sft", post=(), optim=OptimSpec("adamw", lr=5e-3),
                  schedule=Schedule(microbatch_tokens=4096, max_policy_lag=0)),
    topology=Topology(hosts=(HostSpec((pool("main", tp=2), learner(fsdp=2))),)),
    seeds=Seeds(master=11))

# the measurement, outside the run
Measurement(name="distill", env="single_turn", task_ids=<128 held-out>, samples=1,
            every=8, post=("conditioned_teacher_logprobs", "reverse_kl"), seed=3,
            temperature=1.0, max_tokens=1024)
```

The on-policy arm is the same spec with `gen=GenSpec(envs=("single_turn",),
tasks=(<corpus>,), sampling=SamplingSpec(1.0, 1.0, 1024))`, a rollout plan of
`Sample` leaves, `loss="opd"`, `post=("conditioned_teacher_logprobs",)`, and
`teacher` declared beside `main` (Q1).

## Questions

**Q1. SFT on the teacher's samples, or on-policy reverse KL?**
Recommendation: SFT first, as asked — sequence-level distillation, one
trajectory set generated ONCE and reused by every arm (three anchors, two
windows), a lower-variance gradient than `opd`'s score-function estimator,
and a learner-only training phase (the serving host sleeps). Its target is
looser: it matches the teacher's samples, not the teacher's distribution at
the student's own draws. The on-policy arm is one spec edit away and needs
no new machinery beyond the conditioned-teacher processor this ADR builds
for the measurement anyway; it is the fallback if the SFT vectors' measured
reverse KL stalls well above the scoring floor (~0.02 nats, #47).
If the other branch: build the on-policy arm first; the trajectory-set verb
and the `conditioned_teacher` environment are then not needed, and the
replay-window fallback (Q2) is not needed either, since the student samples
its own rows and records its window.

> **Samarth:**

**Q2. Which positions does the trained vector cover — the user turn only, or
every position?**
Recommendation: `Window.PROMPT` as the main arm. The paper injects at the
user turn's tokens during prefill and nowhere else, so a vector trained with
the same coverage is trained on the objective the ICL harness will test: make
the prefix's KV look as if a happiness system prompt preceded it. A vector
trained under `ALL` may do its work on the completion positions directly,
which the harness never steers — a train/eval shift that would muddy a null
result. `ALL` is the comparison arm (the same set, the same plan, one enum in
the bank).
If the other branch (`ALL` only): no `Edge`, no `prompt_lens`, no `Request.
prompt_len` — the window changes shrink to the replay fallback alone.

> **Samarth:**

**Q3. Where does the window live for a row the policy never sampled?**
Recommendation: on the bank entry — `init["window"]`, identity-bearing,
absent when `ALL` so old specs keep their hashes — used by the rollout as the
default when no directive is passed and by the replay when no record exists,
the record winning whenever there is one. This retires the "recorded nothing
under a steer bundle" refusal in `window_mask` (a seal-side test pins that
both buses always record). The engine side needs the entry's init at
`apply`: `attach` today receives payloads only, so either the init rides as
safetensors metadata in the emitted payload (self-describing bundle, no
contract change) or `RolloutLowering.attach` grows an `inits` argument — I
recommend the metadata, since a bundle should carry what serving it needs.
If the other branch (refuse foreign rows, keep ADR 0004's rule intact): SFT
on the teacher's set is impossible for a steer, and the experiment is the
on-policy arm only (Q1's other branch).

> **Samarth:**

**Q4. Is the trajectory set filed and made right — `write_trajectories` beside
`write_tasks`, made by a venue door over `sample_wave`?**
Recommendation: yes. The ref grammar and reader exist; the missing verb is
the writer, and a task set and a trajectory set are the two kinds of content
a plan names (by id, by ref), so `data/tasks/base.py` is the honest home even
though the folder's name says tasks (a rename to `data/content/` is a
separate, cheap decision). The door is `deploy/concept_steer.py::distill_set`
— `sample_wave` over the corpus with routes `{"main": (engine, base_bundle),
"teacher": (engine, base_bundle)}` (the client is constructed for "main" even
when the env only addresses "teacher"), then `write_trajectories`, then the
uri printed; the same shape as `measure_run` and the campaign `evaluate`
doors, i.e. any process against a pool. The generation-only run (a Sealer
daemon) stays an open thread.
If the other branch (build the Sealer now): a run with `algo=None` gets a
committing daemon and the set becomes `store://<run_id>/waves/<u>#<i>` refs —
a bigger change whose value is identity for generation runs, which this
experiment does not need.

> **Samarth:**

**Q5. The hint is task content: `meta["hint"]` holds the rendered system
block, one task set per concept.**
Recommendation: yes — it is what the teacher was told, so it hashes into
every run that names the set (I3), it is the convention `hinted_logprobs`
already reads, and a second concept is a second cas uri, not a knob (#60's
thinking-mode precedent). The proposed text, to be replaced by yours:
"You are a helpful assistant who is deeply preoccupied with happiness.
Whatever the user asks, relate your answer to happiness, joy and
contentment: return to the theme, draw examples from it, and let it color
your tone throughout." The builder asserts the raw concatenation equals the
templated rendering (Decision), which also pins that Qwen3's template emits
no default system block when none is given.
If the other branch (the hint as an environment/processor constant): it
hashes through `code_hashes` instead, and every concept is a source edit.

> **Samarth:**

**Q6. Which prompts, how many, sampled how?**
Recommendation: `HuggingFaceH4/no_robots` (10k human-written prompts with
categories), keeping the categories that invite prose — Generation, Open QA,
Brainstorm, Chat, Rewrite, Summarize — and dropping Coding, Classify, Closed
QA, Extract; 2048 for the set and 128 held out by `split_tasks` (per-task
draw, so the held-out stays held out if the corpus grows). Sampling at
temperature 1.0, top_p 1.0, `max_tokens=1024`: the target distribution is the
teacher's own, and a tempered teacher is a different target; the paper's
1000-token trajectories are the length. A 5120-wide vector will saturate
early — 2048 × ~1000 tokens is generous, and the ledger's forward-KL proxy
(Interfaces) says when it has. If the answers come back short, add Dolly's
creative-writing and brainstorming rows; if they degrade late at T=1.0,
drop to Qwen's recommended 0.7 / 0.8 and say so in the set's meta.
If the other branch: a different corpus is a different builder function and
nothing else.

> **Samarth:**

**Q7. Metal: one A100-80GB:2 (or H100:2), one HostSpec alternating `main`
tp=2 with the learner fsdp=2?**
Recommendation: yes for the SFT arms. The 32B is 32.5 GiB per device at
tp=2 or fsdp=2, so engine and learner cannot co-reside on 80 GB with a
1000-token activation budget and must alternate (sleep mode, a build fact;
`max_policy_lag=0`, which an SFT run does not care about). The learner is
new territory (#78): fsdp=2 over a frozen 32B with `checkpoint_the_blocks`,
a trainable vector at one layer — autograd stops at that layer's input, so
the late arm backpropagates through ten blocks and the early one through
fifty-four, both with recompute; the shakeout measures rank-0 peak the way
`dapo_grpo` does. The serving build pays the steer's demands (eager, our
worker, V1 runner) — required by the gate even for the SFT arms — and its
eager-mode decode cost lands on `distill_set`, which is where I would first
consider a SEPARATE plain build if the number is bad. Rough cost: the set
~40 min, each SFT arm ~40 min, the measurement passes minutes each — three
to four A100-hours for the six-arm matrix. The on-policy arm wants
co-residence (lag 1) and so an L4:8 or H100:4 shape; that is that arm's
question when it comes.
If the other branch (L4:8, tp=4 + fsdp=4 co-resident): every 32B number
above halves per device and #77's tp>1 steer serving gets its first proof at
tp=4, at the cost of #47's 3 GiB-per-device KV budget at 1000-token prompts
plus completions.

> **Samarth:**

**Q8. Anchors and the hook convention: `resid_pre.10 / .32 / .54`, and does
your harness add at the OUTPUT of `model.layers[ℓ]`?**
Recommendation: the paper's own anchors for Qwen3-32B (10 / 32 / 54 of 64,
Appendix B), one run per anchor (three run_ids, identical plans, banks
differing in one site). `resid_pre.<n>` is "the stream leaving layer n"
(ARCHITECTURE.md) — the output of `model.layers[n]`, which is where a
forward hook on the decoder layer adds. If your harness hooks the layer's
INPUT, the matching site is `resid_pre.<n-1>` and the export names it. A
fourth arm at `final_hidden` is the sharpest version of your hypothesis —
an injection that can ONLY be read at the final residual — and costs one
more run; I would add it once the three anchors train.
If the other branch (a per-layer sweep for the generalization variant): the
same venue with `--layer` over every fourth boundary, sixteen runs.

> **Samarth:**

**Q9. Learning rate 5e-3 and one epoch (64 updates × 32 trajectories)?**
Recommendation: yes as the shakeout's starting point. AdamW moves each
coordinate ~lr per step, so a 5120-wide vector moves ~lr·√5120 ≈ 0.36 in
norm per step at 5e-3; mid-stack residual norms on Qwen3-32B are in the
hundreds, and a vector of norm 20–50 is the magnitude class a useful
injection has (the paper's calibrated α sits at a fraction of ‖h‖), so
5e-3 reaches it in ~100 steps and 1e-3 would take the whole run — #46's
1/√(params) rule, read the other way. `microbatch_tokens=4096` packs ~3
documents per forward. Success is read off the ledger: `loss − (−mean
behavior_lp)` (the forward-KL proxy) falling toward the scoring floor, and
the measurement's `reverse_kl` at every 8th version. A second epoch is a
warm-started child run (`extend`'s shape), never a longer plan.
If the other branch: a number is a number; only `OptimSpec.lr` changes.

> **Samarth:**

**Q10. The ICL test runs in the paper's harness over exported vectors, not in
rlstack.**
Recommendation: yes. A k-shot prompt injects k different (vector, layer)
pairs at k user turns; rlstack serves one bundle and one window per request,
and making a request carry a window per bank entry is a change to ADR 0004's
directive rule ("at most one per adapter type per request") that deserves its
own ADR if the ICL eval should ever be a Measurement. Two injection modes in
the harness, both reported: the vector AS TRAINED (an absolute add, what the
objective produced) and the vector NORMALIZED and scaled by the paper's
live-norm α at the paper's α* (the like-for-like comparison against the
extracted vectors). The interpretation table goes in the CONTEXT entry, not
here: p(correct) vs k for trained vectors against the paper's curve, with the
trained vectors' norm ratio to the calibrated injection beside it.
If the other branch: the per-entry directive ADR comes first, and this
experiment waits on it.

> **Samarth:**

**Q11. Byte-identity and the crash-midway state, stated once.**
The SFT run: the train plan is Replay leaves into content, `realize` is a
pure function, the row plan is one slot with `prompt_lens` derived from the
batch, and the window is the entry's declaration — so resume-equivalence
holds by the same argument as every replay run, and `tests/test_resume.py`
gains the shape (promise 2). The trajectory set: `distill_set` is a door,
not a run; a crash mid-way leaves no cas object (the write is the last
step) and a rerun samples again — possibly different text (non-promises),
and a different uri if so, which is honest. The measurement: idempotent by
`measure_run`'s contract; a crash between a point's sampling and its append
loses only that point, backfilled next pass. The identity of EXISTING steer
runs: unchanged, because `init` gains no key at the default (promise 1) and
`SteerWindow`'s new field types change no recorded number. Agree that these
are the obligations, and that the one deliberate loss — the replay-side
"recorded nothing" alarm — is acceptable against a seal-side test?

> **Samarth:**

## Outcome

Filled at implementation.
