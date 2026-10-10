<!-- ADR formatting: keep each prose paragraph and list item on one source line. Do not hard-wrap prose at a fixed column width. Preserve intentional line breaks in fenced code, tables, and diagrams. Apply this convention to all future ADRs. -->

# ADR 0023 — What changes results is declared, typed and hashed

| | |
|---|---|
| **Date** | 2026-10-10 |
| **Status** | Proposed |
| **Author** | Claude (codebase review session, 2026-10-10) |
| **Touches** | `registry.py`, `spec/` (specs, canonical, validate, flow), `training/post/base.py`, `training/losses/` (signatures), `inference/environments/base.py`, `runner/loop.py` + `runner/roles/` (instantiation), every registered entry that reads settings from `task.meta`, `tests/test_architecture.py` |
| **Invariants** | I3 (identity covers the code that runs), I4 (wiring validates at submit), I11 (runs self-describe) |
| **CONTEXT** | Extends #38 (loss purity) and the I3/I4 text; first of a cleanup series (0023–0027, see end) |

## Original prompt

> i want to ask you: what do you think about the state of this codebase? i havent read the codebase in a while, so im not sure whether our deploy scripts are following standard practices, whether any invariants were messed up in the process, or any of the numerous issues we could have. could you do a deep dive and see what might need fixing/

> im mainly interested in the acutal experimnt code (all code involved in the experiment) and seeing whether the design principles are respected there

> lets have multiple ADRs for generally cleaning this stuff up. the code that im writing now will be pushed to the github repo, and the github repo doesnt store experiment-specific stuff. as such, we just need to make a couple of targeted ADRs to fix all the general issues. could we start with the first ADR?

## Context / problem

I3 promises that "editing a registered function's body changes identity", and I4 promises that all wiring validates at submit. The 2026-10-10 review found three ways results change without the spec saying so.

1. **Settings have no typed home, so they ride in `task.meta`.** `AlgoSpec.post` and `GenSpec.envs` are tuples of bare names (`spec/specs.py`), and `AlgoSpec.loss` is a bare name. A processor or environment that needs a knob reads it from each task's meta dict:
   - `ask_memory.py:85-128` reads `pieces`, `route`, `tool_route`, `past_cap`, `max_asks`, `answer_tokens` and `answer_chars`.
   - `program_correctness.py` reads `tests`, `sql` and `types`.
   - Losses can't be configured at all: `grpo_reference_kl(…, clip_eps=0.2, beta=0.04)` takes its coefficients as function defaults.
   - Nothing checks these keys at submit. A typo'd key becomes `meta.get(…) → None` (a silently disabled feature) or a `KeyError` after a GPU rollout. On the dreams branch, `fork_mcq`'s `study_budget_tokens` and `reward_weighting` went unchecked this way. This breaks STYLE rules 2 and 6.
2. **Identity hashes a class body, not the code that runs.** `registry.source_hash` is `inspect.getsource(cls)`. Examples:
   - `grpo_reference_kl` calls `grpo()` and `opd()` from sibling modules, so editing `grpo` changes its loss with the same run_id.
   - `PostDef.produces` is not hashed by value.
   - On the dreams branch, four reward changes (`e7c072d`, `2239d5b`, `0d526f9`, `55cc38b`) landed in unhashed helpers.
   - On resume, `write_dictionary` rewrites `dictionary.json` with the new columns, so one run_id can hold postdata from two different reward functions (I11).
3. **Environment variables can change results.** In this branch every `os.environ` read is a resource: transports, venues, device pinning, rendezvous. On the dreams branch `RLSTACK_FORK_COALESCE_S` decides which lanes share a bf16 microbatch, so rewards depend on worker env and request timing. No rule or test keeps such knobs out.

## Decision

Every registered entry (loss, postprocessor, environment, maker) may declare a frozen `Settings` dataclass. The spec names each use of it as a `Use(name, settings)`, and the submit gate constructs the settings (unknown key, wrong type or bad value → refused before any GPU). Values that are genuinely per-task (a test suite, a gold answer's token ids) stay in task data, but the entry declares a `TaskFields` record that the gate checks against every task row. An entry's identity hash covers the source of the modules its definition imports within the semantic regions, plus its declaration values. Environment variables are confined by an architecture test to regions whose values are resources.

### Touched / untouched

- **Touched**
  - `registry.py`: Def records gain `settings: type | None` and `task_fields: type | None`. `source_hash(obj)` becomes `code_hash(entry)`, the import closure.
  - `spec/specs.py`: a `Use` record; `AlgoSpec.loss`, `AlgoSpec.post`, `GenSpec.envs` and `GenSpec.makers` hold `Use`s.
  - `spec/canonical.py`: a `Use` with default settings canonicalizes to the bare name.
  - `spec/validate.py`: two checks, `check_settings_construct` and `check_task_fields`.
  - `spec/flow.py`: each use's settings go into `dictionary.json`.
  - `training/post/base.py`, `inference/environments/base.py`, `registry.loss`: a `settings=` argument.
  - `runner/loop.py` + `runner/roles/`: one instance per `Use`, `cls(settings)`. A loss receives its settings as keyword arguments. The resume path compares the regenerated `dictionary.json` bytes.
  - Migration of every entry that reads run-level knobs from `task.meta`: `ask_memory`, `routed_turn`, `closed_book`, `answer`, `reflect_retry`, `gold_logprob`, `dream_effect`, `dream_advantage`, `verifier`, `final_answer`, `program_*`, and losses with keyword hyperparameters.
  - `tests/test_architecture.py`: the environment-variable rule.
  - Docs: `rl-stack-spec.md` I3/I4 text, ARCHITECTURE.md (`Use`, `Settings`, `TaskFields`), and a CONTEXT entry.
- **Untouched**
  - `data/` and its stores, the membrane (I1): task rows keep their shape. Validation and the resume comparison live in `spec/` and `runner/`, and the store stays dumb.
  - Transports and the wire: the spec JSON carries settings like any other field.
  - `observe/`: it reads `dictionary.json`, which only gains fields.
  - Adapter types: `AdapterSpec` is already typed.
  - Venues and `deploy/`.

### Promises / non-promises

- **Promises**
  - A misspelled, ill-typed or out-of-range setting, or a task row missing a declared field, is refused at submit with the entry's name.
  - Editing any module in an entry's closure, or its `produces/consumes/requires`, changes the run_id of every spec that uses it, and the host's code-skew check names it.
  - A resume never rewrites `dictionary.json` to different bytes; it refuses instead.
  - No `os.environ`/`getenv` read exists outside the allow-listed regions (test).
  - Specs using only default settings have the same canonical JSON as today. Only the hash values differ.
- **Non-promises**
  - Third-party versions (torch, vLLM, transformers, model revision) are not hashed. Run provenance is ADR 0027.
  - Experiment-specific entries (`fork_mcq`, `self_quiz`, …) are not migrated here; they follow this ADR when and where they live (ADR 0026).
  - GPU numerics are not made deterministic. This ADR only guarantees they are not changed by undeclared knobs.

### Interfaces

- **Registration:** `@postprocessor("name", settings=S, task_fields=T)`, the same shape for `@environment`, `@task_maker`, `@loss`.
- **Spec:** `Use("name", {"k": v})` in Python, `{"name": …, "settings": {…}}` in JSON, a bare string when the settings are the defaults.
- **Gate:** two named checks in `validate.CHECKS`, in spec order after the existing registry checks.
- **Identity:** `code_hashes(spec)` keeps its keys (`"<kind>:<name>"`); values become closure hashes.
- **Observe:** `dictionary.json` lists each use's settings beside its columns.

### Sketches

```python
@dataclass(frozen=True)
class Use:
    name: str                                  # a registered entry
    settings: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AskMemorySettings:
    max_asks: int = 3
    answer_tokens: int | None = None
    past_cap: Literal["end", "refuse_once"] = "end"

@environment("ask_memory", settings=AskMemorySettings, task_fields=AskMemoryTask)
class AskMemory(Environment):
    def __init__(self, settings: AskMemorySettings) -> None: ...

def code_hash(entry) -> str:
    """sha256 over: the entry's declaration values, then the source of every
    module in its static import closure inside SEMANTIC_REGIONS, in path order."""

SEMANTIC_REGIONS = ("rlstack/training/", "rlstack/inference/", "rlstack/policy/", "rlstack/data/tasks/")
ENV_READ_REGIONS = ("runner/transports/", "runner/venues/", "runner/residents.py",
                    "runner/learners/ranks.py", "runner/exporters/", "__main__.py")
```

## Questions

**Q1. Where do knobs live: on the spec's `Use`, or in task data?**
Recommendation: run-level knobs (one value for every task of the run) go on the `Use`. Per-task values (a test suite, a gold answer, a memory name) stay in task rows, declared by `TaskFields`. The rule a reader applies: if every task would carry the same value, it is a setting.
If everything goes in task data with a schema: there is one mechanism, but run-level knobs stay invisible in the spec and in `dictionary.json`.

> **Samarth:**

**Q2. How far does the identity hash reach?**
Recommendation: the static import closure of the entry's defining module, restricted to `SEMANTIC_REGIONS`. That catches `grpo()`-style helpers without hashing the runner, transports or observer, which are I5-neutral.
- If only the defining module: cheaper and fewer spurious changes, but cross-module helpers (`grpo`, `mcq_answers`) still escape.
- If the whole package tree: nothing escapes, but every unrelated commit changes every run_id, and resume after any edit is refused.

> **Samarth:**

**Q3. One identity break, or a versioned identity scheme?**
Recommendation: one declared break. Every run_id changes once, when the closure hash lands. Old runs stay readable in `observe/` and in their stores but cannot be resumed by new code, which the code-skew check already enforces for any edit.
If versioned (an `identity_version` field, old runs hashed the old way): old runs can be resumed, but there are two hash schemes forever, for runs that are all finished today.

> **Samarth:**

**Q4. One instance per `Use`, or settings passed into every call?**
Recommendation: `cls(settings)` once per `Use` at run start, with the registry keeping the class. Settings are then attributes and every call site is unchanged.
If passed per call (`process(group, data, client, settings)`): instances stay shared, but every signature in `post/` and `environments/` changes.

> **Samarth:**

**Q5. When are task fields checked?**
Recommendation: at submit, every row of the tasks file. The file is already read for the data fingerprint (I3), so the extra cost is a pass over rows already in memory.
If only the first wave's rows are checked: submit is faster on huge task files, but a bad row deep in the file fails mid-run, which is exactly what I4 forbids.

> **Samarth:**

**Q6. What does the environment-variable rule allow?**
Recommendation: reads only in the `ENV_READ_REGIONS` above, all of which carry resources: transport limits, device pinning, rendezvous, tokens, exporters. A tuning knob that can change numerics, such as fork coalescing's microbatch grouping when it is ported, becomes a typed field of the build or Schedule and is recorded in the manifest.
If env knobs stay allowed anywhere with a "document it" rule: less churn, but nothing stops the next one.

> **Samarth:**

**Q7. What happens on resume when `dictionary.json` would change?**
Recommendation: refuse, naming the columns that differ. Under Q2 this can only happen if identity also changed, so a mismatch means a bug, never a legitimate resume.
If it is overwritten, as today: the run's self-description can silently stop matching its earlier waves.

> **Samarth:**

## Cleanup series (for orientation; each its own ADR)

- **0023 (this):** what changes results is declared, typed and hashed.
- **0024:** a model's chat template belongs to the policy. Tasks carry messages and flags; rlstack renders them and derives tool pieces once per base model, so drivers never load a tokenizer.
- **0025:** a metal's books are honest and its processes die with it. Foreign VRAM is measured per device against the total; residents get a parent-death signal; stopping a worker is verified to the GPU.
- **0026:** experiments live outside the library. A driver-side campaign client beside `runner/campaign.py` (idempotent submit keyed by run_id, the run record, state and wait), so drivers stop re-implementing submit, auth and record I/O. The experiment modules (dream and SQuAD tasks, fork rewards, rehearsal spec builders) live in a separate package that registers into rlstack, and the boundary is enforced by `test_architecture.py`.
- **0027:** a run records the environment it ran in. One locked GPU image, model revisions pinned, and versions, CUDA, driver and model snapshot written to the manifest. The `banks/` store region is formalized with an exclusive create.

Correctness fixes found by the same review go in as ordinary commits with tests: dp>1 fit gradients, free-VRAM fractions, stale-error replica death, pool key growth, the bank-version race, and the flaky blob retry.

## Outcome

Filled at implementation.
