<!-- ADR formatting: keep each prose paragraph and list item on one source line. Do not hard-wrap prose at a fixed column width. -->

# ADR 0018 — A routed bank entry: one adapter type holds a dreamer and N memories, rows route by the plan leaf's role, requests route by a directive

| | |
|---|---|
| **Date** | 2026-09-17 |
| **Status** | Proposed, implemented on the recommendations (see Outcome) — implementation proceeds on the recommendations under Samarth's 2026-09-17 goal ("if a primitive doesn't align, mark down what you changed with the primitive and fix it before you implement the fix"); every question stays open for his `agree`/`disagree`, and a `disagree` is a follow-up change |
| **Author** | Claude Fable 5.1, the dreams session (research-log post 0007) |
| **Touches** | `policy/adapters/` (one new adapter type: `dream_bank.py`, `dream_bank_torch.py`, `dream_bank_vllm.py`), `runner/assemble.py` (`realize` stamps the leaf's role into the row's turn facts), `runner/roles/scorer.py` (the pin for a wave whose sampled turns disagree), `training/losses/` (`dream_stream.py`, `anchor_calibrate.py`), `training/post/` (`dream_effect.py`, `dream_advantage.py`, `squad_answers.py`, `gold_logprob.py`, `dream_diagnostics.py`, `squad_judge.py`), `inference/environments/` (`answer.py`), `data/tasks/` (`squad_split.py`, `lamp_dreams.py`), `runner/dreams.py` (the plan builders), `data/stores/strangeloop.py` (the existence-probe retry ported from the lamp worktree), `examples/strangeloop-dreams.json`, `tests/` |
| **Invariants** | I2 (one adapter type, two lowerings; the served payload is a container of peft fragments), I3 (the anchor arrives by WarmStart and rides the payload, so identity is unchanged), I6 (a replayed row's route is a fact stamped at realization, byte-identical on the Scorer and the Trainer; sampled rows carry the route the engine recorded), I8 (per-request selection is a directive; requests under different routes batch together), I9 (every evaluation is a postprocessor's pool traffic; the loss is pure and reads roles off `doc_turn_extras`) |
| **CONTEXT** | extends #47 (the teacher is a pool), ADR 0004 Q2 (directives), ADR 0005 (the conditioned teacher), ADR 0006 (self refs answer "not yet"), ADR 0009 (a bank of routed states inside one entry: learned_tasks' posterior bank is the precedent), ADR 0014 (cadence; a learner-only run has no wire), ADR 0015–0017 (Strange Loop). Entry number at implementation. |

## Original prompt

> sure this sounds good. write it in, and then let's get started. fully execute the training workflow. you can spawn up to 12 gpus to parallelize this work as much as possible. please use all rlstack primitives (the desk) and please use strangeloop. if a primitive doesnt align, mark down what you changed with the primitive and fix it before you implement the fix and change it.
>
> i bet there are going to be lots of obvious bugs at first with how the dreamer works, etc. it is your job to resolve them.
>
> make code very clean. folow the design primitives of rlstack, and make deploy scripts very clean, etc.

And, the design it executes (research-log post 0007, `research/lamp-dreams-2026-09-16/PLAN.md` v5): "the dreamer is learning to dream by minimizing NLL over new facts"; "we should still be doing the GRPO NLL random assignment etc etc stuff"; "for the actual dreams, i think we should SFT on only the dreams, not the full trajectory"; "we should not do forwards in the learner for greedy decoding. that's just not correct. we should definitely do them in the actual engine."

## Context / problem

**The experiment.** One run per arm. A stream of arrivals (half a SQuAD paragraph, or a lamp history's events). Sixteen memory LoRAs all train on every arrival's text; a dreamer LoRA writes four dreams per arrival with the arrival in context; each dream is assigned to two memories, which also train on the dream's text with the arrival absent. When the second half of a passage arrives, its NLL is scored under every memory before any update; a 16-row ridge contrast turns those NLLs into an effect per dream of the first half, and the dreamer takes one clipped off-policy GRPO step on those four dreams. Every question is answered by generation in the engine under one memory's adapter and graded; nothing but the arrivals' text and the dreams' text ever trains a memory.

**What exists.** A bank entry is one delta per site (`loop.py:_claim_slot`, `validate.check_sites_do_not_overlap`) and a run's bundle is every servable entry's payload merged into one punica adapter (`lora_vllm.attach` → `merge_fragments`, disjoint keys). Per-request selection is by BUNDLE (a tenant's) and by DIRECTIVE within a bundle (`adapters/base.py:Directive`, `rollout.py:Request.directives`, `vllm_engine._levers_for`), and what a directive did is recorded into `Turn.turn_extras` (`Levers.turn_extras`, `record_directive`). The trainer routes rows by SLOT (one tenant's deltas) and hands each row its recorded turn facts (`ReplayRows.facts` ← `TokenBatch.doc_turn_extras` ← `Flat.turn_extras`), which is how `learned_tasks` picks a posterior row per sequence (`learned_tasks_torch.task_latents`). A plan leaf carries a `role` that "the gradient path understands one of (v0)" (`data/plan.py:TRAIN`) and `realize` re-tags rows with THIS plan's group keys (`assemble.py:realize`). The Scorer pins a wave to the one bundle its turns recorded and refuses a wave pinning two (`scorer.py:pinned_bundle`). Every evaluation number the design needs is `score` or `sample` traffic a postprocessor sends (`post/base.py`, `hinted_logprobs.py` is the walk), and the engine's LoRA slot budget is `max_bundles × (max_members + 1)` (`rollout.py:ServingBuild.slots`).

**What does not exist, and why the existing primitives do not answer it.**

1. *Seventeen LoRAs at the same sites in one run.* Seventeen `lora` entries violate the bank rule and would merge into one adapter on the engine. Seventeen tenant RUNS would need dynamic cross-run data (dreams sampled mid-stream, consumed by other runs), which the ref grammar forbids at Phase 0 by design (`refs.py`). The precedent is `learned_tasks`: ONE entry holding a bank of states, the row's recorded fact choosing which applies.
2. *A row that the plan replays under several routes.* A dream is sampled once (recorded route: the dreamer) and trained by two memories; the same sealed row must reach the forward with two different routes in two different waves. The record cannot carry that; the PLAN can — its leaf already has a `role`, unused beyond `train`.
3. *Scoring under the current memories.* The probe (P2's NLL under each memory) and the QA generation must run under the policy as of the previous arrival's last update. The Scorer's pin rule refuses a wave whose sampled turns disagree (P1's dreams, sampled 8–64 arrivals ago, replayed beside the arrival's text), and no rule says what "current" means to a runner that may run ahead or behind.
4. *A covariance anchor the learner can load.* The anchor (about 100 MB of per-site factors) has no channel to the learner except `WarmStart` (a parent's payload → `load`) or re-emission at every version — the cost ADR 0014 measured and deferred (its "frozen half of `emit`").
5. *The lamp store fix.* The bounded existence-probe retry that kept the lamp pipeline alive (`research/lamp-prefix-2026-09-13/STRANGELOOP-CHANGES.md` #18) is uncommitted in the `strangeloop-backend` worktree and absent here.

## Decision

One adapter type, `dream_bank`, is a bank of LoRA sets inside one entry: a dreamer set and N memory sets, plus an optional frozen anchor (rank-r-plus-diagonal factors of the base's input covariance per site) and, in calibrate mode, the accumulators that produce one. Its replay lowering applies to each row the set named by the row's `route` fact; its rollout lowering attaches every set as its own punica adapter (N+1 slots per bundle, declared through `max_members = N`) and `apply` selects the set a `Route` directive names, recording it. A plan leaf's `role` becomes the routing fact of a replayed row: `realize` stamps `role` into every turn's facts and, when the role names a set, `route` as well, on the Scorer and the Trainer alike. The Scorer pins a wave whose sampled turns disagree to the bundle committed at update u−1, and waits for that commit. The anchor arrives by `WarmStart` from a calibrate run and rides every emitted payload (the deferred frozen half is not built; the cost is measured, not hidden). Losses dispatch by `role` off `doc_turn_extras`: memory rows clone (`sft`, plus the anchor penalty once per update), dreamer rows take the clipped GRPO surrogate over `advantage`, eval rows are masked. Everything else is postprocessors and plan builders in existing shapes.

### Touched / untouched

- **Touched** — `policy/adapters/dream_bank.py` (declaration: `serving = PUNICA`, `provides = {anchor_penalty, memory_delta_norm, dreamer_delta_norm}`, `directive = Route`, `records = ("route",)`), `dream_bank_torch.py` (sets, routing by fact, penalty, calibrate accumulation, emit/load of the container payload), `dream_bank_vllm.py` (attach N+1 adapters from the container, `apply` by directive, `align` zero, `detach`).
- **Touched** — `runner/assemble.py:realize`: rows come back tagged with the group key AND, in each turn's facts, `role` (the leaf's) and `route` (when the role names a set). A leaf whose role is `train` stamps nothing, so every existing plan realizes to the same bytes.
- **Touched** — `runner/roles/scorer.py:pinned_bundle`: unanimous turns pin as today; disagreeing turns pin the ledger's bundle at u−1, and `next_rows` additionally waits for that commit. Waves whose turns agree are byte-for-byte what they were.
- **Touched** — `training/losses/dream_stream.py`, `anchor_calibrate.py`; `training/post/dream_effect.py`, `dream_advantage.py`, `squad_answers.py`, `gold_logprob.py`, `dream_diagnostics.py`, `squad_judge.py`; `inference/environments/answer.py` (sample under the route the task names, plain-text prompt); `data/tasks/squad_split.py`, `lamp_dreams.py`; `runner/dreams.py` (the specs: stream, calibrate, frozen-base dreams; the arrival waves; the assignment; the derived meta); `examples/strangeloop-dreams.json`; `data/stores/strangeloop.py` (#18's `_still_there`); `tests/`.
- **Untouched** — `policy/compile.py` (one entry, one payload, one bundle id: unchanged), `policy/adapters/lora*.py` (composed, not edited), `runner/engines/vllm_engine.py` (the bus loops adapter types; nothing new to loop), `runner/roles/trainer.py` and `generator.py` (the update, the wire, the pacing are what they were), `data/flatten.py` and `data/trajectory.py` (turn facts already travel), `spec/` (no new field: `init` carries the type's knobs, roles are strings the plan already has), `runner/measure.py` (unused: evaluation is in-run), `runner/learners/` (facts already reach the forward).

### Promises / non-promises

- **Promises** — With `memories = 0` and no anchor, a `dream_bank` entry's dreamer set is a rank-r `lora` in bytes and numerics, served through the same punica path. A row routed to set s gets exactly set s's delta and no other; a row with no route fact gets the dreamer's (the default, recorded). The penalty equals ½ tr(ΔW C ΔWᵀ) densely at full rank, and its gradient matches finite differences. `realize` on the Scorer and the Trainer produce identical rows for every plan, including every plan that exists today. A wave whose turns agree pins as before. The calibrate run's single checkpoint restores as an anchor in a stream run through `WarmStart`, and a stream run resumes from any checkpoint with the anchor intact. The suite is green with fakes; the torch-gated tests pass locally on CPU.
- **Non-promises** — No claim about the science; nothing on metal is promised here. The frozen half of `emit` is not built: the anchor rides every payload, and the pilot measures what that costs per update. Population replacement (copy the best memory over the worst) is not built in pass 1. The judge pool's placement on a second metal is the desk's business and is measured, not promised.

### Interfaces

- **Spec**: `AdapterSpec("dream_bank", site, init={"r": 8, "memories": 16, "lam": λ, "anchor_rank": 128, "mode": "stream" | "calibrate", "seed"})`. Sugar `dream_bank(site, r, memories, lam, anchor_rank, mode)`.
- **Routes**: set names `dreamer`, `memory:00`…`memory:15`; `base` is the bare base (engine: no adapter; replay: pass-through). `Route(name)` is the directive; `record_directive` writes `{"route": name}` into the turn's facts.
- **Roles on leaves**: `train` (default, nothing stamped), `dreamer`, `memory:<j>`, `eval`. `realize` stamps `role` and, for `dreamer`/`memory:<j>`, `route`.
- **Losses**: `dream_stream` (requires `advantage`, `anchor_penalty`), `anchor_calibrate` (requires nothing; `0·Σlp`).
- **Processors**: `dream_effect` (pools `main`; scores each doc row under its route with a `Route` directive; fits the contrast from the arrival task's meta; produces `prequential_nll`, `dream_reward`), `dream_advantage` (inline; z-scores `dream_reward` over the group's dreamer rows; produces `advantage`), `squad_answers` (inline; grades sampled answers against `meta["answers"]`; `em`, `f1`), `gold_logprob` (pools `main`; scores the gold span under the row's route; `gold_lp`), `dream_diagnostics` (pools `main`; the base's NLL of a dream given the arrival, its length; `dream_base_nll`, `dream_tokens`), `squad_judge` (pools `judge`; SEAL's yes/no prompt; `judged`).
- **Environments**: `conditioned_teacher` (unchanged) samples dreams under the default route; `answer` samples a short answer under `Route(meta["route"])`.
- **Engine build**: `max_members = memories` so the slot budget is `max_bundles × (memories + 1)`; `serves` includes `dream_bank`.
- **Store keys**: nothing new. The anchor is the calibrate run's `adapters/pi@<v>`.
- **`observe/`** sees `anchor_penalty`, `memory_delta_norm`, `dreamer_delta_norm` on every ledger line and the processors' columns in postdata.

### Sketches

```python
# policy/adapters/dream_bank_torch.py
@dataclass
class DreamBankState:
    r: int
    sets: dict[str, LoraState]              # "dreamer", "memory:00", ...
    anchor: Anchor | None                   # frozen: per path U [d_in, k], e [k], delta
    accumulate: bool                        # calibrate mode: sum x xᵀ per path
    sums: dict[str, torch.Tensor]; count: int

def route_of(facts_for_row) -> str:        # the row's `route` fact, else "dreamer"

class DreamBankSite(SiteWrapper):
    def forward(self, x):                   # per row: inner(x) + delta of sets[route_of(row)]

def provide(state) -> {"anchor_penalty": lam * Σ_s ½ tr(ΔW_s C ΔW_sᵀ) over memory sets, ...}

# runner/assemble.py
def realize(entry, reader):                 # rows tagged group=key; turns' facts gain role (and route)

# runner/roles/scorer.py
def pinned_bundle(self, update, wave):     # unanimous → that bundle; else the ledger's bundle at update-1
```

## Questions

1. **One entry with routed sets, versus per-entry bundles.** Per-entry bundles would let every set be its own bundle id at the cost of a compile-contract change (an entry emitting several servable payloads) and N+1 bundle pushes per update. Recommend the one-entry container: identity and the wire are untouched, and `max_members` already prices N+1 slots per bundle. The other branch is cleaner for measurement outside the run, which this design does not use.
2. **The route as a plan-leaf role stamped at `realize`, versus a new leaf field.** A new field (`Replay(ref, route=…)`) is more explicit and changes the plan codec and every reader of it; the role already exists, already hashes into the plan bytes, and its v0 meaning ("train") is preserved as the default. Recommend the role. Consequence: `role` gains vocabulary the loss reads; the plan doc says so.
3. **Stamping into turn facts, versus a new per-document channel.** Facts already reach the forward, the Scorer's realize and the Trainer's realize agree by construction, and `waves/<u>` records exactly what trained. A new channel would touch `flatten`, `TokenBatch`, both learners and the fakes. Recommend facts. Consequence: a wave's rows now differ from the rollouts they replay by two keys, which the wave browser shows.
4. **The Scorer's pin for a disagreeing wave: the ledger's bundle at u−1.** Under `max_policy_lag = 0` this equals the recorded bundle for any freshly sampled wave, so no existing run's postdata changes; it is exactly "the policy before this update", which is what a probe means; and it is deterministic on resume because it is a function of the update index. The alternative, refusing as today, makes the design impossible without a second pool on a second engine. Recommend the u−1 rule, with the Scorer waiting for that commit.
5. **Anchor delivery by WarmStart, riding every payload.** The deferred frozen-half contract would save about 100 MB per push; building it touches the trainer's checkpoint, restore and the initial blobs. Recommend WarmStart now, measure the push cost in the pilot, and build the frozen half only if it dominates.
6. **Evaluation as sampled leaves in the rollout plan, masked from the loss by role `eval`.** Alternative: `measure_run` at checkpoints only. In-run keeps question-level records in the run's own rollouts and postdata every arrival at the cost of masked forwards over short rows. Recommend in-run.
7. **Dreams' memory rows are the dream rollouts themselves, replayed under a memory role**, so the memory trains on the dream turn's tokens with the dream prompt (the instruction) as its injected context and the arrival absent — because the conditioned teacher seals the hint out. Alternative: derived supervised rows under a different prompt. Recommend replaying the rollout: no derivation, and the row that trained is the row that was dreamed.
8. **Replacement (copy best over worst) deferred.** It breaks the contrast's randomization and needs a learner-side copy hook. Recommend deferring to pass 2.
9. **The `_still_there` port.** A bug fix, not a design choice; recorded because it is a change to a primitive on this branch. Recommend porting as is.

## Order of work

adapter type and its tests → realize's stamp and the scorer's pin, with resume-equivalence tests → losses, processors, environment, task families → plan builders and specs → desk config and the research driver → CPU end-to-end with fakes → Strange Loop: desk up with the 0.5B, smoke every stage → the 7B: calibrate, frozen-base dreams, pilot, stage A, the gate, stage B.

## Outcome

Implemented on `claude/dreams` (2026-09-17) on the recommendations; every question still awaits Samarth's `agree`/`disagree`, and a `disagree` is a follow-up change. CONTEXT entry #93. Fakes suite 1520 OK (205 skipped); the torch-gated bank, loss and plan tests OK under the factual-env interpreter; four smoke shapes (none, own with the dreamer trained, base replaying a teacher run, calibrate) through the desk on Qwen2.5-0.5B; on Qwen2.5-7B the calibrate run (2000 chunks × 256 tokens, rank 128), the base-dreams teacher (82 rollout waves) and the stage A/B arms on Strange Loop, one per metal.

What the answers changed on metal: Q5 (the anchor rides every payload) made the wire payload ~260 MB per update, which the desk's relayed wire cannot carry — the stream topology became one alternating host per arm (engine sleep mode on) rather than the frozen-half contract, which stays the fallback; Q6 (in-run evaluation as rollout leaves) needed the Trainer to drop `eval` rows before the learner sees them; Q4's u−1 pin held as written. Added beyond the ADR, each a primitive fix recorded in CONTEXT #93 part E: scratch deadlines that grow with the bytes and read retries on 5xx, the provider's SSH wait and ended-lease rule, the desk's archived placement mode, `--solo`/`--delivery` on the deploy door, one update per BATCH of arrivals (D), text and dream groups apart (D), and `anchor_raw`.

Later the same day (CONTEXT #93 part F): a five-try read budget in the scratch client, the desk reconnecting saved leases before its runtime ticks, the archived placement mode replayed on reroute, the padded-footprint microbatch budget in `pack`, `prequential_base` and the `base` eval route. The first arms at memory lr 2e-4 collapsed within forty passages; the campaign runs at lr 5e-5.

Then (CONTEXT #93 part G): the store's refused-write retry and lost-acknowledgement readback, and the dream instruction as a setting with SEAL's implications prompt; at lr 5e-5 the memories neither drifted nor learned and restatement dreams read null over 60 dreams, so the controls are SEAL's.

Unproven: the penalty's effect, the trained dreamer's reward beyond the smoke, the judge pool on metal, the lamp family.
