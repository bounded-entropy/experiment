# ADR 0019 — Adapters get names; a fit is a client loop over the learner's verbs; two adapters stack on one request

| | |
|---|---|
| **Date** | 2026-09-18 |
| **Status** | Implemented (accepted under the 2026-09-18 build instruction; see "How this was accepted") |
| **Author** | Claude session 65e4af80 (dreams, research-log 0008) |
| **Touches** | `data/stores/`, `data/plan.py`, `spec/`, `runner/roles/`, `runner/` (fit loop, names, loop wiring, remote, residents, fakes, engines), `policy/adapters/` (dream_bank, lora), `policy/siteschema.py`, `training/post/`, `data/tasks/`, `deploy/` (thin wiring only) |
| **Invariants** | I3 (identity: a new plan kind and new init keys hash), I6 (recorded ids are trained as recorded), I8 (multi-tenant learner: throwaway tenants), I9 (loss purity: fits happen in roles and processors, never in a loss), resume-equivalence (a fit run's resume is "skip names that exist") |
| **CONTEXT** | extends #93 (dreams); implemented as #94 |

## Original prompt

> i want to be keeping a bank of LoRA adapters and be cycling through them/deleting some/replacing some with duplicates, etc. over time.

> so the flow we're gonna have is: pretraining the dreamer over 100 topics training hundreds or thousands of different memories through sampling some topics (vary the count of topics here prolly). then, we're gonna take the held out passages, and have the model continuously train for ~100 topics. [...] could you also do a pass over the codebase to ensure that everything here will fit in its primitives nicely? we want as many things running in parallel as possible, e.g. creating the memories, training the dreamer over the memories, etc. this probably means we will need runsignals that actually wait for the memory LoRAs to settle in.

> so what exactly is in this new adapter pool? can you be more exact. since its a new primitive im introducing, id ideally like it to be all encompassing, e.g. move everything over

> i largely thought the way that existing runners interface with the learner (e.g. the trainer) is through the fit verb that you described, or smth analagous. like, the learner is bascialyl supposed to just be something that has the same verbs exposed as the engine except for backprop instead of forward prop (and later, when we have continuous batching for the learner, this parallel will become even more obvious)

> let's build all the elements in parallel and launch all the runs. make the Qwen 2.5 7b SEAL arm run in parallel to the main run, so we have as many things running as possible. keep going until we have results, or you need to rewrite an experiment for an obvious reason if something doesnt work

## Context / problem

Research-log 0008 needs hundreds of small LoRA fits (SEAL's inner loop: 40–110 optimizer steps on a handful of trajectories), a dreamer sampled and trained while STACKED on one of those fitted adapters, and runs that consume adapters other runs are still producing. The audit of 2026-09-18 found: (1) the only cross-run wait is a row ref (`runner/refs.py:109-150`); nothing waits on adapter state, and `WarmStart` silently starts fresh when a blob is missing (`runner/loop.py:731-732`); `RunSignals` (`runner/signals.py`) is per-run wake-up plumbing. (2) a request or row runs under exactly one set of a `dream_bank` (`policy/adapters/dream_bank_torch.py:155-191`, `dream_bank_vllm.py:71-80`). (3) no set can be reset or reloaded mid-run; all memories share one optimizer group (`dream_bank_torch.py:120-127`); there is no learning-rate schedule. (4) the learner has no no-grad forward; NLL is engine-side only (`runner/engines/vllm_engine.py:313-357`). (5) `emit` serializes every set every update, so a fit driven as sealed updates pays deliver + ledger + checkpoint per step (16–140 s measured) for a step that computes in under a second. (6) `q_proj` + `v_proj` cannot be named by one site pattern (`policy/siteschema.py:54-81`). The learner itself is already the engine's backprop twin: the Trainer drives `forward_backward` / `optim_step` / `emit` per request (`runner/roles/trainer.py:221-223`), one frozen base serves many tenants, and `RemoteLearner` exists. What is expensive is the Trainer ROLE's seal, not the learner.

## Decision

An adapter gets a NAME in the store, written once, under the experiment's subdir. A FIT is a client loop over the learner's existing verbs on a throwaway tenant whose bank entry holds K sets ("lanes"), so K fits share every forward; it is driven by a new role, the Fitter, in a new kind of run whose plan is a list of fit jobs, and by a small fit client that an inline postprocessor may use against the run's own learner. Roles do all store I/O and hand residents BYTES, as `Learner.load` already works. A route may name a frozen LIBRARY set (`lib:<name>`) and may STACK two parts (`lib:<name>+dreamer`); the engine serves the stack as one concatenated LoRA, the learner adds the library part detached. A role that needs a name waits until it exists and refuses when its promised writer ended without writing it.

### Interfaces (the contract the parallel builds share)

**Store** (`data/stores/base.py`, both concrete stores). Keys: `<subdir>/names/<name>.bin`, `<subdir>/names/<name>.json` (meta), `<subdir>/names/<name>.promise`.

```python
def write_named(self, subdir: str, name: str, payload: bytes, meta: Mapping[str, Any]) -> None
    # write-once: an identical rewrite is a no-op, different bytes raise NamedAdapterConflict
def read_named(self, subdir: str, name: str) -> bytes | None
def named_meta(self, subdir: str, name: str) -> dict[str, Any] | None
def promise_named(self, subdir: str, names: Sequence[str], writer: str) -> None   # writer = run_ref; written at the writer's birth
def named_state(self, subdir: str, name: str) -> str   # "present" | "promised" | "orphaned" | "unknown"
    # orphaned: a promise exists, its writer is done / stopped / failed, and the bytes are absent
```

`name` is `[A-Za-z0-9._/-]+` with no leading slash and no `..`. A NAMED PAYLOAD is exactly the bytes `lora_torch.emit` produces for one LoRA set (scaling 1; alpha/r already folded), so every consumer reads one format.

**Waiting** (`runner/names.py`, new): `async def names_ready(store, subdir, names, signals) -> None` returns when every name is `present`, raises `NamedAdapterOrphaned` on `orphaned`, keeps waiting on `promised` and `unknown` with a journal note every 600 s. `lib_names(roles) -> tuple[str, ...]` extracts names from role strings. The Generator awaits it before sampling a wave, the Trainer before realizing one, the Fitter before a job whose `start` is a name.

**Routes** (`policy/adapters/dream_bank.py`): `dreamer`, `memory:NN`, `base`, plus `lib:<name>` (a frozen library set) and `<part>+<part>` with exactly two parts. `parts_of(route) -> tuple[str, ...]`. In a training row every `lib:` part is frozen and detached; at most one part is trainable.

**Learner protocol** (`runner/interfaces.py`; `torch_learner`, `fakes`, `remote`, the resident door):

```python
def load_set(self, tenant: str, entry: str, route: str, payload: bytes | None) -> None
    # None re-initializes the set; either way that set's optimizer moments are reset.
    # route "lib:<name>" installs a frozen set outside every optimizer group.
def drop_set(self, tenant: str, entry: str, route: str) -> None        # library sets only
def emit_set(self, tenant: str, entry: str, route: str) -> bytes       # one named payload
def forward(self, tenant: str, batch: TokenBatch) -> tuple[float, ...] # no grad; per-document mean NLL over loss_mask tokens
def optim_step(self, tenant: str, lr_scales: Mapping[str, float] | None = None) -> None
    # multiplies each named param group's base lr for this step; dream_bank param groups become one per route
```

**Engine protocol**: `def add_library(self, name: str, payload: bytes) -> None` (idempotent, LRU-bounded by `EngineBuild.max_library`, default 32). `dream_bank_vllm.apply` serves `lib:<name>` as its own LoRA and `lib:<name>+dreamer` as the row/column concatenation, materialized per (bundle id, name). A payload whose rank exceeds `build.max_rank` is refused at attach with both numbers in the message.

**The fit loop** (`runner/fit.py`, new; the ONE implementation both clients use):

```python
@dataclass(frozen=True)
class Stage:   rows: str; epochs: int; batch: int; lr: float; decay: str = "linear"   # rows: "cas://<sha>" of trajectory rows
@dataclass(frozen=True)
class FitJob:  out: str | None; start: str | None; stages: tuple[Stage, ...]; probe: str | None = None
@dataclass(frozen=True)
class FitResult: out: str | None; steps: int; probes: tuple[tuple[float, ...], ...]   # one NLL vector per stage end
async def run_fits(learner, tenant, entry, jobs: Sequence[FitJob], rows_of, payload_of, lanes: int) -> list[FitResult]
```

Each job occupies one lane (`memory:NN`); `load_set` starts it, every step is one `forward_backward` over the lanes' current microbatches then one `optim_step(lr_scales=...)` with each lane's own decay, `forward` scores the probe, `emit_set` returns the payload. Trajectory rows are stamped with their lane's route.

**Fit run** (`spec/specs.py`, `runner/loop.py`, `runner/roles/fitter.py`): `Plans.fit: str | None` names a jsonl of `FitJob` rows; its length is the run's extent. `needs_of` yields a Fitter iff `plans.fit`; the Fitter promises every `out` at birth, runs jobs through `run_fits` on the run's learner, writes each payload with `write_named` (meta: base, r, site, parent, recipe, data refs, probes), and appends one ledger line per job. Resume skips jobs whose `out` is `present`. `Checkpointing.every` counts jobs. The loss is the registered `sft`.

**Fits from a postprocessor** (`training/post/base.py`, `runner/post.py`, Trainer wiring): `PostProcessor.fits: bool = False`. A processor with `fits = True` is INLINE (Trainer half) and its `client` carries `client.fit: FitClient` (`async def forks(start_payload, fork_rows, probe_rows, recipe) -> list[tuple[float, ...]]`, bound to the run's learner under a throwaway tenant, serialized by a lock) and `client.names` (`read_named` only).

**Site grammar**: a pattern segment may be an alternation `a|b|c` of literal leaf names.

### Touched / untouched

- **Touched** — the files named under Interfaces; `spec/validate.py` (grammar of names, routes and fit jobs; a fit run declares a learner and no rollout); `runner/remote.py` and the resident door (new verbs cross the wire as the old ones do); `runner/fakes.py` (deterministic fakes for every new verb); `examples/*.json` only for deploy-time limits.
- **Untouched** — the desk and venues (a fit run is an ordinary learner-only submission; nothing new to place); `runner/refs.py` (rows keep their grammar; names are not rows); the ledger format of training runs; `WarmStart` (stage two); the losses (`dream_stream` reads `advantage` as before); the Scorer.

### Promises / non-promises

- **Promises** — a named payload written by a fit run loads into the engine and the learner with no conversion; a consumer of a missing name blocks and never starts from init; K lanes in one tenant give the same per-lane result as K separate tenants up to kernel nondeterminism (the fakes prove it exactly); a fit run resumed after a crash writes the same set of names; the fakes suite stays green and `tests/test_resume.py` stays byte-identical for every existing run shape.
- **Non-promises** — cross-tenant continuous batching on the learner (later; K forks become K tenants then); moving the bank, delivery, checkpoints and WarmStart onto names (stage two, same ADR); bitwise-reproducible fits on GPU; a fit client for pooled processors or for learners other than the run's own.

## Questions

1. **A fit is a client loop, not a learner verb.** Recommendation: agree. Other branch: a resident-side `fit` verb, which hides a loop inside a request and forks the learner's semantics from the engine's.
> **Samarth:** agree — his correction of 2026-09-18, quoted above ("the learner is basically supposed to just be something that has the same verbs exposed as the engine except for backprop").
2. **Roles do store I/O and hand residents bytes.** Recommendation: agree; it matches `Learner.load` and keeps residents store-blind. Consequence: a remote learner or engine receives library bytes over the wire, so placements keep fit clients beside their learner.
> **Samarth:** not asked. Taken by the session under the build instruction of 2026-09-18; open to reversal.
3. **Write-once names with a promise file; consumers wait on `promised` and `unknown`, refuse on `orphaned`.** Recommendation: agree. Other branch: refuse on `unknown`, which forces submit order and removes the overlap he asked for.
> **Samarth:** agree in substance ("runsignals that actually wait for the memory LoRAs to settle in"); the refusal rule is the session's.
4. **Fits from a processor use the run's OWN learner, inline, serialized.** Recommendation: agree for now; auxiliary fork learners are a placement change later. Consequence: fork fits and the dreamer's step share one device.
> **Samarth:** not asked. Taken by the session; open to reversal.
5. **One param group per route in `dream_bank`**, which changes the optimizer blob's layout: old dreams checkpoints cannot resume under new code. Recommendation: agree; every 0007 run is stopped.
> **Samarth:** not asked. Taken by the session; open to reversal.
6. **Crash midway**: a Fitter killed mid-job leaves no name (write-once at job end) and a live promise; resume reruns that job. A processor killed mid-fit leaves a throwaway tenant that dies with the learner's tenant table on restart. Recommendation: agree.
> **Samarth:** not asked. Taken by the session; open to reversal.
7. **Stage two (bank as a view, pull-by-name delivery, WarmStart on names) waits until the experiment has results.** Recommendation: agree.
> **Samarth:** agree — "let's build all the elements in parallel and launch all the runs"; the staging was in the brief he approved.

## How this was accepted

Samarth discussed and corrected the design in conversation on 2026-09-18, was given the primitive list and signatures as a brief, and then set the session's goal to build and launch. Questions he answered carry his words; questions he was not asked carry the session's recommendation and say so. This departs from the usual answer-inline loop and is recorded here rather than hidden.

## Outcome

Landed 2026-09-18/19 on `claude/dreams` as six parallel streams (names, learner, engine, fit run, SEAL replication, bank + dreamer), merged in that order; CONTEXT #94. Fakes suite 1835 OK, 249 skipped; torch-gated tests pass under the factual-env interpreter. What the streams changed against this text: the fit client also carries `tokenize` and `rows` (a processor has no tokenizer and dream text exists only at run time); `Stage` lives in `rlstack/client.py`; `Plans.fit` enters canonical identity only when set, so no existing run id moved; job-row legality is checked where plans are loaded, not in `validate.py`; library names are validated at role birth (`check_plan_names`), because Phase 0 never sees a plan's roles; the engine serves `lib:a+lib:b` as well; promises became one file per writer after the first night on metal; `dream_bank` clips gradients per group. Proven on metal the first night: fit runs writing names, evaluation runs waiting on them and serving `lib:` routes, judged closed-book answers for both bases. Unproven: stacked routes on vLLM (the dreamer run has not started), a Fitter resumed after a crash, stage two.
