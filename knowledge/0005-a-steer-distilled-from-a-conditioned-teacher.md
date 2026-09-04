# ADR 0005 — A steering vector distilled from a prompt-conditioned teacher: the trajectory set as content, the conditioned-teacher channel, and a window the bank declares

| | |
|---|---|
| **Date** | 2026-09-04 |
| **Status** | **IMPLEMENTED 2026-09-04 — CONTEXT #81** (venue written and UNRUN). Accepted at the 2026-09-04 review; Q5, Q6, Q9 and Q11 stood on their recommendations; Q3 REDACTED in part (a row with no record replays at every position) and that redaction is the whole of the steer change |
| **Author** | Claude Fable 5.1 (session: the SPAR introspection paper, 2026-09-04) |
| **Touches** | `data/tasks/` (one new dataset file), `inference/environments/` (two one-file environments), `training/post/` (two processors in the existing shape), `runner/measure.py` (a measurement may route to a second pool), `deploy/concept_steer.py` (the venue), `tests/`, and ONE rule in `policy/adapters/steer_torch.py` (Q3, as redacted: a row with no record replays at every position). No `rollout.py`, no `replay.py`, no learner or engine file (Q2, Q3) |
| **Invariants** | I3 (the hint is content, so it hashes; the teacher run's identity is the set's), I6 (a record, when one exists, is the truth; a row without one replays at the adapter type's default — nothing is declared in the bank), I9 (a processor scores through a declared pool under privileged conditioning — hinted + teacher, combined), I5 (a 32B student is new metal territory; the spec says nothing about where) |
| **CONTEXT** | extends #40 (the scoring verb, hinted logprobs), #47 (the teacher is a pool), #60 (a dataset becomes content), #70 (measurement outside the run), #77 / ADR 0004 (the steer, UNCHANGED: the record is the only truth — Q3). DEPENDS ON ADR 0006 (a run is daemons with resources): the teacher's trajectory set is a generation-only RUN, not a content file |

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

1. *The teacher cannot sample as a RUN of its own.* The right shape for
   the trajectory set is a generation-only run — a spec with `algo=None`, a
   rollout plan, an empty bank — whose sealed `rollouts/` the SFT runs
   replay by ref. The spec allows it and the store keeps it (attach never
   sweeps rollouts), but nine places in the runner hard-code the experiment
   shape — the `algo=None` refusal at `loop.py:115` is only the first —
   and the ref grammar cannot name another run's rollouts. That audit and
   its fix are ADR 0006, which this ADR depends on; no Sealer is needed,
   because `write_rollout` is already atomic and never refused.
2. *No environment samples from a non-policy pool under a hint and seals the
   hint OUT.* `math_single_turn` samples the policy; `teacher_logprobs` and
   `hinted_logprobs` SCORE. The teacher's completions have to be sampled
   with the system prompt in context and sealed with the user prompt alone,
   so the student trains on exactly what it will be asked at eval.
3. *The steer's replay refuses a row that recorded no window.*
   `steer_torch.py:recorded_window` raises "every turn served by a steer
   bundle records the window it applied" — correct, and the rule the review
   kept in part (Q3: nothing is declared in the bank; a record, when one
   exists, is the truth) and redacted in part: a trajectory the policy never
   sampled carries no record and is NOT a bug — it replays at every
   position, the default. One rule in `steer_torch.py` changes; the bank-
   declared window is withdrawn.
4. *"Prompt-only" is not sayable without a tokenizer.* Moot: the review
   chose every position (Q2), which is ADR 0004's default and needs no
   coordinates. Left as the record of why a prompt edge was considered.
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

*(Rewritten after the 2026-09-04 review: Q2 chose every position, Q3 ruled
that the record is the only truth and nothing is declared in the bank, and
ADR 0006 Q6 gave a learner-less run its own init. The window machinery the
first draft proposed — a symbolic prompt edge, a bank-declared default, prompt
lengths threaded to replay — is gone. One steer rule changes, at Samarth's
redaction of Q3: a row with no record replays at every position.)*

Two runs of machinery, in the repo's vocabulary. First, **the teacher's
trajectory set is a generation-only run** (ADR 0006 Part B): a
`concept_prompts` task set whose tasks carry the user prompt (chat-templated,
no system block, thinking off) and the happiness system block as
`meta["hint"]`; a `conditioned_teacher` environment that samples `main`
under hint + prompt and seals prompt + turn; a spec with `algo=None`, an
EMPTY bank (so `main` is the bare Qwen3-32B — the conditioned teacher, no
steer anywhere), and a rollout plan of `Sample` leaves over the corpus. Its
turns carry no `steer_window`, and need none: the hint is stripped at the
seal, and at replay a row with no record is steered at every position — the
adapter type's default, the same one the rollout applies when no directive is
passed, declared in no bank (Q3, as redacted). Its sealed `rollouts/` are the
set and its `run_id` is the set's identity.
Second, **the student is an ordinary SFT run**: `Plans.rollout=None`, a train
plan of `Replay("store://<teacher_run>/rollouts/<r>#<i>")` leaves, loss
`sft`, an empty post pipeline, the same steer entry trainable at ONE anchor
boundary (`resid_pre.10` / `.32` / `.54`, `d=5120`), one run per anchor, the
same plan bytes in all three so only the bank differs. Replay reads each
row's recorded window when there is one and applies the default when there
is none — ADR 0004's record rule, with its "recorded nothing" refusal
replaced by the default; a seal-side test keeps the alarm (every bundle
carrying a steer records a window on both buses). The distillation is
**measured outside the run** (#70): a Measurement whose env samples the
student on held-out prompts and whose pipeline is
`conditioned_teacher_logprobs` → `reverse_kl`, through a `measure_run` that
routes "teacher" beside "main" on the same engine under the base bundle. The
**ICL test is the paper's harness**, fed the exported vectors
(`adapters/<name>@<v>.bin` is already safetensors keyed by boundary path);
rlstack serves one window per request and the k-shot prompt needs k
different injections, a directive rule this ADR does not change (Q10).
On-policy distillation (`opd` + the same conditioned-teacher processor, a live
rollout plan) is the second arm, one spec edit away, and the fallback if SFT's
target proves too loose.

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
- **Touched** — `rlstack/inference/environments/conditioned_teacher.py`
  (new): `sample` the `main` pool with `[hint_for(task), prompt]` and no
  directive, return `Rollout(task, messages=[prompt, turn.message],
  turns=[turn], env_extras={"hint": hint.content})`. The hint is in the
  record as provenance and out of the message stream by construction, so
  `flatten` never tokenizes it into the student's document. The turn carries
  no steer record — the teacher run's bank is empty — and needs none.
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
- **Touched** — `rlstack/policy/adapters/steer_torch.py`: ONE rule —
  `recorded_window` answers `(0, None)` for a row whose turns recorded no
  window (the default the rollout applies with no directive), instead of
  refusing; the "two windows in one row" refusal stays. The "recorded
  nothing under a steer bundle" alarm moves to the seal side: a test pins
  that both buses record a window for every bundle carrying a steer.
- **Touched** — `rlstack/runner/measure.py`: `measure_run(..., pools:
  Mapping[str, Engine] = {})` — every extra name routes to its engine under
  a payload-free base bundle, exactly `loop.py`'s `base_bundles` rule, so
  "teacher" on the same engine as "main" is one dict entry. `Measurement`
  gains nothing.
- **Touched** — `deploy/concept_steer.py` (new venue, `stress_fleet.py`'s
  desk shape): doors `prompts` (build the task set), `distill_set` (submit
  the teacher's generation-only spec through the desk and print its run_id;
  the SFT specs name it), `train --layer` (submit one arm through the desk),
  `measure` (one `measure_run` pass, by hand or cron), `export --run_id
  --version` (the vector's safetensors to the volume for the paper's
  harness), and `sweep` / `status` as every venue has. Everything
  semantics-bearing is in the spec (I5).
- **Untouched** — `policy/adapters/steer.py`, `steer_vllm.py`,
  `rollout.py`, `replay.py`, `runner/learners/torch_learner.py`,
  `runner/engines/vllm_engine.py`: the rollout side records exactly as
  before, the directive is unchanged, nothing is declared in the bank (Q3);
  only the replay's missing-record rule moves, in `steer_torch.py`. `spec/specs.py`: the
  `steer(...)` sugar is unchanged; no window enters `init`.
  `training/losses/sft.py` and `opd.py`: the objectives are exactly right as
  written; SFT's rails on foreign rows read as the distance to the teacher
  (see non-promises), and `opd` requires the column the new processor
  produces. `training/post/teacher_logprobs.py` and `hinted_logprobs.py`:
  the new processor is beside them, not inside them. `data/plan.py`,
  `runner/assemble.py`: the Replay leaf was built for this and needs nothing
  (`runner/refs.py` gains the rollouts location under ADR 0006, not here).
  `runner/loop.py`, `runner/daemons/*`: an SFT run with no rollout plan and
  an empty post pipeline plans a Trainer alone, byte-for-byte the daemon it
  was; the loop still requires a "main" engine (reachability, `tokenize` for
  injected spans, `add_bundle`), which is why the topology declares one even
  though the SFT run never samples. `rlstack_engine/steer.py` +
  `steer_worker.py`: untouched. `spec/validate.py`: no new gate here (ADR
  0006's learner-less-bank gate is that ADR's). `data/flatten.py`: the hint
  never reaches it.

### Promises / non-promises

- **Promises** — (1) One steer rule changes and no spec does: every steer
  spec keeps its identity, and every row that recorded a window replays it
  exactly as before. (2) An SFT run over the teacher's rollouts is
  resume-equivalent: the plan is Replay leaves, the rows are sealed content,
  the row plan and the recorded window are pure functions of (spec, rows),
  so a killed-and-resumed run's directory is byte-identical to a straight one
  — `tests/test_resume.py` gains this shape. (3) A row with no record replays at
  every position and a row with one still replays its record (torch-gated,
  in the image); the fake bus and the vLLM bus record a window for every
  bundle carrying a steer (the seal-side alarm, on fakes). (4) The fakes suite is green. (5) The teacher
  run's `run_id` is the set's identity — its spec, its code, its corpus —
  and the SFT plans name its rollouts by ref, so the fingerprint names what
  was trained on (I3) and the submit gate refuses a rollout that is not
  sealed yet. (6) The measurement writes `measurements/<run_id>/distill/`
  and the run directory gains not one byte (#70's rule, pinned). (7) Every
  venue door that acquires metal ends with the desk's `release` and the
  plane asserted empty (ADR 0003 / #77).
- **Non-promises** — The teacher run's rollouts are NOT byte-reproducible
  across engine batch compositions: vLLM's seeded sampling is deterministic
  per request only up to batch noise (#46 saw it), so a second teacher run
  of the same spec (a second store, say) may seal different text; within
  ONE store the run resumes by skipping sealed rollouts and never re-samples
  one, which is what identity needs. **The coverage confound is accepted
  (Q2):** the vector trains over every position, prompt and completion,
  while the paper's harness injects at the user turn only during prefill; a
  null ICL result therefore has a train/eval shift among its explanations,
  and the CONTEXT entry says so beside the number. The SFT ledger's
  `logprob_gap` on foreign rows is NOT a parity alarm: `behavior_logprobs`
  are the TEACHER's under the hint, so the rail reads mean |student −
  teacher| — the distillation distance, which is useful, and the observer
  will still label it as the parity rail (stated, not renamed). No claim
  about the ICL result: this ADR builds the vectors and the harness inputs;
  the experiment's outcome is the paper's harness's to report. The 32B
  learner's memory and pace are UNMEASURED until the shakeout; the
  eager-mode serving cost too. Nothing here proves a steer under tp=4 (#77
  names tp>1 unproven; #78 proved tp=2 at 0.6B).

### Interfaces

- **Content:** `cas://<sha>` for the task set (declared in `GenSpec.tasks`
  by the teacher run; the SFT runs declare `gen=None`); the teacher run's
  `rollouts/<r>`, named by `Replay("store://<run_id>/rollouts/<r>#<i>")`
  leaves in the SFT train plan (ADR 0006's ref), checked resolvable at the
  submit gate as every store ref is.
- **Pools:** the teacher run declares `main` alone, with an empty bank (the
  bare base under the hint is the conditioned teacher); the SFT runs declare `main` (idle but
  required); the measurement and the on-policy arm route `main` (the
  student's bundle) and `teacher` (`bundle:base:teacher`, payload-free) to
  ONE engine object — the Routes contract already says "one engine may back
  many pool names". Through the desk, a second inference demand with the
  same (base, tp) JOINS the same listing (`covers` is capability equality).
- **The directive rule (ADR 0004 Q2), one clause added.** No directive =
  every position; what applied is RECORDED; replay reads the record — and a
  row without one (a trajectory the policy never sampled) replays at every
  position, the same default. Nothing is declared in the bank. The environment's control is per request, decided before
  decode; a per-token decision made DURING decode is not expressible in the
  engine's request model and is not needed here (Q3's note).
- **The gate:** nothing new here; `check_sites_reachable_on` already holds
  the steer's boundary against the main engine's inventory, so both the
  teacher run's and the SFT run's serving host must be a steer-serving build
  (eager, our worker class) — a fact of the spec, not of the venue.
- **`observe/`:** the SFT run's ledger has `loss` (the cross-entropy against
  the teacher's samples) and `logprob_gap` (mean |student − teacher|); the
  forward-KL proxy is `loss − (−mean behavior_lp)`, both already columns.
  `measurements/<run_id>/distill/points` carries `reverse_kl` and
  `teacher_logprobs` means per measured version, rendered as a dashed series
  like every measurement.
- **Export:** `adapters/<name>@<v>.bin` is `steer_torch.emit`'s safetensors
  keyed by boundary path (`model.layers.10`); the `export` door copies it
  out. The paper's harness adds the tensor under that key at the output of
  `model.layers[10]`, which is what `resid_pre.10` names (Q8, confirmed).

### Sketches

```python
# the teacher: a generation-only run (ADR 0006 Part B) — its rollouts ARE the set
ExperimentSpec(
    policy=PolicySpec(base="Qwen/Qwen3-32B", bank={}),
    gen=GenSpec(envs=("conditioned_teacher",), tasks=(<corpus>,),
                sampling=SamplingSpec(temperature=1.0, top_p=1.0, max_tokens=1024)),
    plans=Plans(train=None, rollout=<cas: 64 waves x 32 Sample(task, "conditioned_teacher")>),
    algo=None,
    topology=Topology(hosts=(HostSpec((pool("main", tp=2),)),)),
    seeds=Seeds(master=5))
# one teacher run serves all three student arms: its rollouts carry no steer
# record, and each arm's replay applies its own vector at every position

# inference/environments/conditioned_teacher.py
@environment("conditioned_teacher")
class ConditionedTeacher(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        turn = await client.sample([hint_for(task), prompt])      # main: the bare base
        return Rollout(task=task, messages=[prompt, turn.message], turns=[turn],
                       env_extras={"hint": task.meta["hint"]})

# the student, one arm: everything semantics-bearing, in one value
ExperimentSpec(
    policy=PolicySpec(base="Qwen/Qwen3-32B",
                      bank={"v": steer("resid_pre.10", d=5120)}),
    gen=None,
    plans=Plans(train=<cas: 64 waves x 32 Replay("store://<teacher_run>/rollouts/<r>#<i>") leaves>,
                rollout=None),
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
If the other branch: build the on-policy arm first; the teacher run and the
`conditioned_teacher` environment are then not needed, and the
replay-window fallback (Q2) is not needed either, since the student samples
its own rows and records its window.

> **Samarth:** agree — SFT first (2026-09-04).

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

> **Samarth:** ALL only (2026-09-04). *Folded: no prompt edge, no prompt lengths, no comparison arm; the coverage shift against the harness is a stated confound (non-promises).*

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

> **Samarth:** "it definitely shouldnt be declared in the bank. for replay purposes, the thing that should be logged should be the positions at which steering occurred (for replay). for the engine, when generating a new token, the environment should be able to specify whether to steer or not. for learning, the learner simply injects at the positions recorded alongside the sealed rollout (in meta information or whatever)" (2026-09-04). *Folded: ADR 0004's rule stands untouched — the record is the only truth and the refusal stays. The teacher's turns get a real record by being sampled under the student's v0 bank (zero steer = the base, #77); with ALL the record is coordinate-free after the hint is stripped. Per-request control via the window is what exists today; a per-token decision DURING decode is not expressible in the engine's request model and is not needed under Q2.* — then, REDACTED in part (2026-09-04): "i want to redact that the sft traces have to seal the tokens. just make the default that it applied for all the window." *Re-folded: a trajectory the policy never sampled carries no record and replays at EVERY position — the adapter type's own default, the same one the rollout applies with no directive, declared nowhere in the bank. The teacher run therefore needs no bank at all, and `steer_torch.py` changes in one rule: a row with no record is the default, not a refusal.*

**Q4. The teacher's trajectory set is a generation-only RUN (ADR 0006), and
this ADR waits on that one.**
Recommendation: yes. A run gives the set what a content file cannot: a
manifest naming the corpus, the environment and the hint by hash; a
dictionary; resume by `already_sealed`; the desk's supervision; and a place
in the observer. The SFT plans then name `store://<run_id>/rollouts/<r>#<i>`
— a ref the gate checks before the student starts. The teacher spec declares
`main` and `teacher` on the same bare base (both bind to one engine; the
client is constructed for "main" even though the env addresses "teacher"),
an empty bank, `algo=None`, and a rollout plan whose length is the run's
extent. The cost is sequencing: ADR 0006 lands first.
If the other branch (do not wait): a content verb `write_trajectories` beside
`write_tasks` and a venue door over `sample_wave` — the shape `measure_run`
and the old `evaluate` doors already have — with `Replay("cas://<sha>#<i>")`
leaves; about forty lines, no supervision, no resume, and a second way of
making trajectories that ADR 0006 would then retire.

> **Samarth:** follows from Q3 and ADR 0006 Q5/Q6 — the set is a generation-only run (not asked separately, 2026-09-04).

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

> **Samarth:** not raised in the 2026-09-04 review; the recommendation stands unless Samarth objects. The prompt text is a draft for him to replace.

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

> **Samarth:** not raised in the 2026-09-04 review; the recommendation stands unless Samarth objects.

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

> **Samarth:** agree — A100-80GB:2, alternating (2026-09-04).

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

> **Samarth:** agree — the output of `model.layers[ℓ]`, sites `resid_pre.10 / .32 / .54`, three runs (2026-09-04). The `final_hidden` arm was offered and not taken.

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

> **Samarth:** not raised in the 2026-09-04 review; the recommendation stands unless Samarth objects.

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

> **Samarth:** agree — the paper's harness, both injection modes (2026-09-04).

**Q11. Byte-identity and the crash-midway state, stated once.**
The SFT run: the train plan is Replay leaves into content, `realize` is a
pure function, the row plan is one slot, and the window is each row's own
record (Q3) — so resume-equivalence
holds by the same argument as every replay run, and `tests/test_resume.py`
gains the shape (promise 2). The teacher run: a generation-only run under
ADR 0006's obligations — atomic rollouts, resume by skipping sealed ones,
done when the last rollout is sealed — and an SFT run submitted before the
teacher finishes is refused at the gate for the rollouts it names, never
left waiting. The measurement: idempotent by
`measure_run`'s contract; a crash between a point's sampling and its append
loses only that point, backfilled next pass. The identity of EXISTING steer
runs: unchanged, because `init` gains no key at the default (promise 1) and
`SteerWindow`'s new field types change no recorded number. Agree that these
are the obligations, and that the one deliberate loss — the replay-side
"recorded nothing" alarm — is acceptable against a seal-side test?

> **Samarth:** not raised in the 2026-09-04 review; the obligations stand as stated. (The seal-side alarm clause STANDS: Q3's redaction replaced the replay refusal with the default, and the alarm moved to the seal side as the clause says.)

## Outcome

### Landed 2026-09-04, recorded as CONTEXT #81

**What landed.** `data/trajectory.hint_for(task)` — moved out of
`training/post/hinted_logprobs.py` so the conditioned-teacher ENVIRONMENT may
reach it without `inference/` importing `training/`; `HintedLogprobs`' class
source is untouched, so no run's identity moved.
`data/tasks/concept_prompts.py` (`concept_prompt_tasks`, `renderer`,
`system_block`, `user_prompt`, `check_hint_concatenates`, `is_prose`,
`task_from_row`, `prompt_splits`, `SYSTEM_PROMPT`, `PROSE_CATEGORIES`),
registered as `concept_prompts` in `__main__.BUILDERS`.
`inference/environments/conditioned_teacher.py` and `single_turn.py`.
`training/post/conditioned_teacher_logprobs.py` and `reverse_kl.py`.
`policy/adapters/steer_torch.py`: `recorded_window` answers `(0, None)` for a
row with no record, with `steer.py`'s and `steer_torch.py`'s docstrings
recoded to say the rule. `runner/measure.py`: `measure_run(..., pools=)`,
`base_bundles`, plus `token_level_columns` / `column_mean` under
`reduce_point`. `deploy/concept_steer.py` (new venue, 856 lines, UNRUN).
`ARCHITECTURE.md`'s Directive and Measurement entries, one clause each.

**Tests.** 973 green on fakes (from 936 at ADR 0006 Part B), 117 torch-gated
skips, ~7.7 s. New: `tests/test_conditioned_teacher.py` (19 — the two
environments, the two processors, the split, and the teacher→student SFT end
to end including resume-equivalence), `test_tasks.py` (+11 over the builder's
pure half with a fake tokenizer), `test_steer.py` (+2 seal-side, +2
torch-gated replay, one refusal case narrowed to disagreement alone),
`test_measure.py` (+3). `tests/test_resume.py` is untouched and green.

**What the answers changed.** Q2 (ALL only) removed every window machine the
first draft proposed and left the coverage confound as a stated
non-promise. Q3's redaction is the whole of the steer change: nothing is
declared in the bank, a record is the truth where there is one, and a row
without one replays at the default — which is what makes SFT on a foreign
trajectory set possible for a steer at all. Q4 made the trajectory set a
generation-only RUN, which is why this ADR waited on 0006 and why the teacher
needs no bank.

**What the shape's own questions decided, at implementation.** (a) `hint_for`
went to `data/trajectory.py` beside `Task` rather than `data/tasks/base.py`:
it is a read of what a task SAYS, not a verb for building a set, and
`tasks/base.py` imports the Store. (b) The concatenation assertion runs PER
ROW, not once on a probe, because each Task's own `meta["hint"] + prompt` is
what the engine will concatenate for that row. (c) `reduce_point` had to
learn token-level columns before a distillation measurement could be
written at all — fixed by reading `PostDef.token_level`, never a value's
shape. (d) The venue's split ASKS for 2304 train prompts to guarantee the
plan's 2048, because counts are drawn and a 2048/8850 draw falls short half
the time.

**Unproven.** No metal: nothing here has run on a GPU, and every number in
the venue is a guess. The 32B student's rank-0 peak and pace, and the
eager-mode serving cost of a steer-serving 32B build, are unmeasured. The
chat-template concatenation assertion is only exercised on the volume; the
no_robots schema was read off the hub API, not off a download. FOUND WHILE
WRITING THE VENUE and stated rather than fixed: `FsdpTorchLearner.sleeps` is
`ranks.width == 1`, so at fsdp=2 the alternating HostSpec's arbiter wires no
sleep hook for the learner and only the engine hands its share back — whether
Q7's one-A100-80GB:2 treaty holds is the shakeout's first finding. *Since
CONTEXT #82 the sharded learner sleeps too (`sleeps` is a probe of the pinned
torch, not the width), so both members hand the device back and Q7's partition
sizes for the largest member rather than the sum — conditional on
`stress_fleet.py::learner_sleep`, which is written and unrun.* The
coverage confound (Q2) and the SFT ledger's `logprob_gap` reading as
distillation distance rather than parity stand exactly as the non-promises
state them.

**One contradiction in this document, left standing rather than edited
away.** Q11's closing note says "(The seal-side alarm clause is moot: the
replay refusal was kept, Q3.)" — written before Q3's redaction, which
replaced that refusal with the default. The Decision, as rewritten after the
review, and Q3's own redaction are what was implemented; Q11's note is stale
and is left as the record of the order the answers arrived in.
