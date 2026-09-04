# ADR 0006 — A run is daemons with resources; the experiment is loop.py's special case

| | |
|---|---|
| **Date** | 2026-09-04 |
| **Status** | Proposed — **Part A (the learner is a routable resident; the Trainer need not share its host) ACCEPTED 2026-09-04 by Samarth's instruction and in implementation**; Part B (generation-only runs: Q1, Q3–Q8) open |
| **Author** | Claude Fable 5.1 (session: the SPAR introspection paper, 2026-09-04) |
| **Touches** | Part A: `runner/remote.py` (`HostService` forwards the learner verbs; `RemoteLearner` over any transport), `runner/host.py` (`resolve_routes` learns the learner; `check_fit` accepts a routed one; custody journaled), `runner/loop.py` (`attach_residents`: a remote learner is a free resident; the tenant is uninstalled at the end of `submit`), `runner/interfaces.py` (`Learner.uninstall`), `runner/learners/torch_learner.py` + `runner/fakes.py` (the verb), `runner/campaign.py` (the learner demand yields a route; the anchor is a choice), `runner/desk.py` (`deliver` routes the learner demand), `deploy/` (one door: the runner beside the sampling host, the learner on another metal — written, unrun), `ARCHITECTURE.md` ("Resident / daemon", "Learner", "Wire" recoded), `tests/`. Part B: `runner/loop.py` (Phase 1 split, `plan_daemons` → the needs), `runner/daemons/generator.py` (the pacing rule takes its buffer from the caller), `spec/specs.py` (`Plans.train` optional; the extent), `runner/campaign.py` (the anchor rule), `runner/desk.py` (`finished`), `runner/host.py` (`submit` on a learner-less host; the journal's plan), `runner/refs.py` (`store://<run_id>/rollouts/<r>`), `observe/views.py` + `observe/series.py` (progress off the extent), `spec/validate.py` (one gate: a learner-less run's bank), `tests/` |
| **Invariants** | I1 (a rollout is sealed data whether or not anything trains on it), I3 (`Plans.train=None` is a new value; every existing run's identity is unchanged), I10 (the store is the run: done-ness must be readable off it for every run kind), I12 (a learner-less run anchors on an inference host) |
| **CONTEXT** | extends #27 (the blackboard: "plan_daemons derives one daemon per GPU responsibility"), #59 (the plan is data), #43/#69 (the desk is workload-blind; the anchor is the campaign layer's rule); closes the "generation-only runs" open thread — WITHOUT a Sealer |

## Original prompt

> "and a generation-only run is still refused." this should very much not be the case. could you look through the worktrees and see if this is actually still the case? i thought i downgraded loop.py to just be a special case of spawning these daemons, and spawning these individual daemons was still a valid thing to do...

> yes, the structure should be: runs should require some daemons with some resources. experiments are just a special case of this that's expressed in loop.py. are there any abstractions preventing this? this should be a common pattern across the repo.

## Context / problem

**What was checked.** All nine branch tips (main and the eight worktrees),
every working tree (clean), the stash list (empty), and the full history of
every branch (`git log --all`). The refusal is in every tree at the top of
`run_experiment_async`; it entered with the import on 2026-08-27 (`ac05290`)
and was only reworded on 2026-08-28 (`9cc7ec2`). No commit anywhere ever
defined a Sealer or a generation-only entry point, and every tree's CONTEXT
still lists the thread as open.

**The seam Samarth remembers is real.** `run_experiment_async` is Phase 0
(identity) + Phase 1 (setup) + `plan_daemons` + one `TaskGroup`
(`loop.py:212-227`). `plan_daemons` is a separate exported function
returning `list[Daemon]`; the daemon base says in its docstring that a
condition method is meant to be overridden (`daemons/base.py:6-8`); daemons
synchronize through store predicates alone (`signals.py`). The spec already
allows the shapes: `algo: AlgoSpec | None  # None = generation-only run`
(`specs.py:251`), the gate skips every algo rule when it is None
(`validate.py`, pinned by `test_validate.py:111,407` and
`test_registry.py:236`), `code_hashes` and `flow_graph` and
`data_fingerprint` all guard it. And the store already keeps a
generation-only run's output: `write_rollout` is atomic and "never refused
against the ledger" (`stores/base.py:708-712`), and attach's
`_discard_unsealed` sweeps `waves/`, `postdata/` and blob versions past the
ledger tail — `UPDATE_SECTIONS` at `stores/base.py:62` — and never
`rollouts/`. So the "committing Sealer daemon" the open thread asks for is
not needed: a Generator alone leaves exactly the durable artifact a replay
reads. What is missing is that nothing lets it run alone, and nothing can
name what it wrote.

**What prevents it, file and line.** Nine places hard-code the experiment
shape (a Trainer, a learner, a ledger, a train plan):

1. `loop.py:115` — `run_experiment_async` refuses `algo=None` before
   anything runs.
2. `loop.py:173,189` — Phase 1 calls `learner.install` and `learner.emit`
   unconditionally, and `parameterization_of` (`loop.py:247-249`) reads
   `spec.algo.optim` and `spec.algo.loss`. `Host.submit` hands `self.learner`
   (`host.py:119`: `Learner | None`), so on a serve-only host a learner-less
   spec passes `check_fit` (which only demands a learner when a
   `LearnerMember` is declared) and then dies on `None.install`.
3. `loop.py:410` — `attach_residents` attaches the learner to the arbiter
   unconditionally.
4. `loop.py:294-343` — `plan_daemons` builds the Trainer always and reads
   the Generator's `due_at` off `plans["train"]`.
5. `generator.py:39` — the Generator's buffer is `spec.algo.schedule.
   max_policy_lag`; `generator.py:67` — `may_generate` waits on
   `committed()`, the ledger tail, which a run with no Trainer never
   advances. With no train plan `due_at` is empty, `consumed_by(index)` is
   `index` ("dead weight, paced as its own update"), so only the first
   buffer + 1 rollouts ever generate and the daemon waits forever.
6. `specs.py:84` — `Plans.train: str` is mandatory: "its LENGTH is the run's
   length". `loop.py:229` — `RunReport.updates_completed = len(plans[
   "train"])`; `loop.py:79` — the fingerprint reads it.
7. Done-ness is the ledger against the train plan in four places:
   `desk.py:1008` `finished` (the reaper's "is this still work" — a run
   with no ledger is parked and rerouted forever once its host dies),
   `observe/views.py:218` ("THE LEDGER IS TRUTH"), `observe/series.py:25`,
   and the host journal's attach event (`host.py:361`, `"plan":
   spec.plans.train`) that the observer's progress reads.
8. `campaign.py:44-48` — `demands_of` marks the LEARNER demand as the anchor
   ("the learner is never remote, said once, here"); `desk.py:679-683` —
   `submit` requires exactly one anchor. A learner-less spec has none and is
   refused at the desk's door. *(Dissolved by Part A: the learner is
   routable, so the anchor is a choice, and the deeper rule behind this line
   — that a run's Trainer must sit beside its learner — is gone with it.)*
9. `refs.py:26-28,90-96` — the store ref grammar is `store://<run_id>/
   waves/<u>` only. A Generator writes `rollouts/`; only a Trainer's
   `next_rows` materializes `waves/`. Another run cannot name a
   generation-only run's output.

One more, an edge rather than a blocker: a learner-less run whose bank is
not empty has nowhere to BUILD its deltas — Phase 1 gets the initial
payloads from `learner.emit`, and `AdapterType.params` needs torch. Serving
a trained steer from a generation-only run therefore needs its payloads to
come from somewhere sealed (Q6).

**What is already the pattern, and what the audit says.** The daemons
themselves are clean: the Trainer and Scorer read `spec.algo` because they
only exist when there is one; the Generator reads `spec.gen`. The Measurement
(#70) is already "a thin loop any process runs against a pool" — a daemon by
another name, outside the run. The desk is already workload-blind: it places
demands and delivers a frame to an anchor, and does not know a Trainer from
a Generator. So the pattern Samarth names is two-thirds present; the
remaining third is that the RUN — what Phase 1 sets up, what "done" means,
where the frame lands, what a ref can name — is spelled once, for the
experiment, in the loop and in the four readers of its ledger.

## Decision

### Part A — the learner is a routable resident (accepted 2026-09-04)

> **Samarth:** ok yes. let's actually make the learner fully remote just like
> the engine, and remove the constraint that the trainer must live on the same
> process.

**The learner is reached exactly as an engine is.** The wire already exists
(`RemoteLearner` / `LearnerService`, five JSON verbs, byte-identity tested —
ADR 0002); what the amendment adds is the plumbing that let only the local
host use it:

- **The door.** `HostService.serve` forwards `install`, `forward_backward`,
  `optim_step`, `emit`, `load` and the new `uninstall` to the host's
  `LearnerService`, each frame ADMITTED at that host's arbiter under the
  learner resident — the `sample_tokens` / `score_tokens` shape, so the
  learner's alternation group (a multi-member HostSpec) is honored where the
  learner lives. The verbs stay synchronous frames (ADR 0002 Q6, "later"):
  over a real venue the transport's `ask` runs on its own thread, the shape
  every venue's `blocking_ask` already has.
- **Fit and routes.** `Host.resolve_routes` resolves a `"learner"` route into
  `RemoteLearner(transport, fsdp=member.fsdp)` exactly as it resolves a pool
  address into `RemotePool`; `check_fit` accepts a `LearnerMember` served by
  a routed learner as it accepts a routed pool; `run_experiment_async` takes
  the learner it is handed, local or remote, and `check_members_match_their_
  shape` attests `fsdp` off the proxy's hello as before.
- **Attach.** `attach_residents` attaches a remote learner as a zero-footprint
  free resident (the `RemotePool` rule): local admission is bookkeeping, the
  real admission happens at the learner's host per frame.
- **The desk and the anchor.** `deliver` routes the learner demand under the
  key `"learner"` (the name `place` already uses); `demands_of(spec, anchor)`
  marks the chosen demand — the learner's when the spec declares one and
  nothing else is asked, `main` otherwise or when asked — so a run's Trainer
  lives where the run is anchored, by choice. The desk stays workload-blind.
- **Tenant lifecycle.** `Learner.uninstall(tenant)` joins the protocol (the
  `_remove` that already exists behind `install`'s reset, made a verb);
  `Host.submit` uninstalls its tenant when the run ends, done or failed, so a
  learner shared by runs anchored elsewhere does not accumulate the dead.
  Resume is unchanged: a re-adoption installs and restores from blobs, as
  every attach already does. The learner's host journals `learner-attach` /
  `learner-detach` events per tenant so the observer sees who is on it.

**Promises (Part A).** (1) Every existing venue and test that hands a local
learner behaves as before: the learner's host is still the default anchor,
and a local `RemoteLearner` still rides the local transport. (2) On fakes: a
run anchored on the `main` pool's host with its learner on ANOTHER host
produces a run directory byte-identical to the same spec anchored on the
learner's host (`test_resume.py`'s shape, across two `Host` objects and a
`LocalTransport`). (3) Two runs anchored on two different hosts join ONE
learner listing through the desk, both commit, and the learner's roster
shows both then neither. (4) A learner-less spec is no longer refused at the
desk for want of an anchor; whether it RUNS is Part B. (5) The fakes suite
is green.

**Non-promises (Part A).** No metal proof in this ADR — the venue door is
written and left unrun for Samarth (it costs a second metal). The
per-microbatch frame and the per-update `emit` frame now cross a real wire
when the learner is remote: the cost is the open thread's "4 GB on the wire"
at large banks and is measured, not fixed. Frames stay synchronous. The
NCCL transfer verb (ADR 0002 Q3) stays a later ADR.

### Part B — a run is daemons with resources (open)

**A run is a set of daemon NEEDS, each naming the daemon, its plan, and the
residents it admits; `run_experiment_async` derives the experiment's needs
from its spec and is otherwise the generic runner.** Concretely:

- `needs_of(spec) -> tuple[DaemonNeed, ...]` replaces the inside of
  `plan_daemons`: a `TrainerNeed` iff `algo` is not None (plan = train,
  admits the learner and the inline half's engines), a `ScorerNeed` iff the
  pipeline has a pooled half, a `GeneratorNeed` iff a rollout plan exists
  (plan = rollout, admits the main engine, `buffer` = the lag when a Trainer
  exists and `None` when nothing consumes — unpaced). Phase 2 is one
  TaskGroup over the needs, as today.
- **Phase 1 splits in two.** "The policy's initial bundle" is every run's
  (the main engine must serve the bank): with a learner it is `install` +
  `emit` as today; without one the bank must be EMPTY or every entry FROZEN
  with sealed payloads (a `WarmStart` from a run that trained it — the blobs
  ARE the payloads, read by `_warm_start`'s path, no torch on the runner's
  side) — refused at the gate otherwise. "The learner's install" is the
  TrainerNeed's alone. `attach_residents` attaches what the needs admit.
- **The extent.** A run's length is its `extent` plan: the train plan when
  there is one, else the rollout plan. `Plans.train` becomes `str | None`
  (a rollout-only run leaves it None; the fingerprint already writes `-`);
  `Plans.extent` is a property, not a field, so no existing spec's canonical
  bytes change. Done-ness is ONE store predicate, `run_done(store, run_id)`:
  the ledger tail against the train plan when the manifest has one, the last
  sealed rollout against the rollout plan otherwise. `desk.finished`,
  `views._progress`, `series` and the host journal's `plan` field read it
  or record it; `RunReport.updates_completed` becomes `completed` beside an
  `extent` name.
- **The anchor.** `demands_of` anchors on the learner when there is one and
  on the `main` pool's demand otherwise: the runner is thin, the store is the
  run, and a Generator beside its serving host is the natural seat.
  `Host.submit` with `learner=None` is legal exactly when the spec declares
  no `LearnerMember` (already what `check_fit` checks).
- **The ref.** `store://<run_id>/rollouts/<r>#<i>` joins the grammar: a
  sealed rollout of another run, resolvable before this run starts, so the
  submit gate checks it like every store ref.
- **Nothing else moves.** No Sealer: the rollout's atomic write IS its seal.
  No daemon changes its four beats. Measurement stays outside.

### Touched / untouched

- **Touched** — `runner/loop.py`: `needs_of`, the Phase 1 split, the generic
  Phase 2; `run_experiment_async` keeps its name and signature (learner
  becomes `Learner | None`), and its body reads as "the experiment's needs,
  run". `runner/daemons/generator.py`: `buffer: int | None` from the caller;
  `may_generate` is `True` when None. `spec/specs.py`: `Plans.train: str |
  None`, `Plans.extent`. `runner/campaign.py`: the anchor rule.
  `runner/desk.py`: `finished` calls the one predicate. `runner/host.py`:
  `submit` tolerates `learner=None` for a learner-less spec; the attach event
  journals the extent plan. `runner/refs.py`: the rollouts location.
  `observe/views.py`, `observe/series.py`: progress off the extent.
  `spec/validate.py`: `check_learnerless_bank_is_sealed` — a spec with no
  `LearnerMember` may not carry a trainable entry, and its frozen entries
  must have a `WarmStart` source. `tests/`: a generation-only run on fakes,
  its resume-equivalence, the desk placing and finishing one, the ref
  resolving, the gate refusing a trainable entry without a learner.
- **Untouched** — `runner/daemons/trainer.py`, `scorer.py`: they exist only
  under their needs and read what they always read. `runner/daemons/base.py`,
  `runner/signals.py`, `runner/arbiter.py`: the four beats and the
  blackboard are already generic. `data/stores/base.py`: `write_rollout`
  and attach already do the right thing; the predicate is a read over
  existing keys. `runner/measure.py`: outside the run, already a free-
  standing loop. `runner/desk.py`'s placement, carve, release: workload-
  blind, and stay so. Every existing run's identity: `Plans.train` present
  hashes as before.

### Promises / non-promises

- **Promises** — (1) Every existing spec has the same `run_id` (no field
  added to a hashed record; `extent` is derived). (2) A spec with `algo=None`
  and a rollout plan runs on fakes: manifest, dictionary, plans, and one
  sealed rollout per planned wave; killed and resumed, its directory is
  byte-identical to a straight run's (`test_resume.py` gains the shape).
  (3) The desk places it (anchor = main), a listing that dies is reaped, and
  `finished` reads true once the last rollout is sealed — no eternal
  parking. (4) An SFT run whose train plan replays
  `store://<gen_run>/rollouts/<r>#<i>` passes the gate and trains, on fakes.
  (5) The fakes suite is green; `run_experiment_async` for a training spec
  plans exactly the daemons it planned before (pinned by the existing
  `test_loop.py` and `test_daemons.py`).
- **Non-promises** — No generation-only run on metal until ADR 0005's venue
  uses one. No Sealer, no ledger for a generation-only run: its extent is
  rollouts, and readers that want "commits" see zero. `migrate` refuses a
  generation-only run as it refuses any run with no ledger (it has nothing
  to warm-start from), stated rather than fixed. A learner-less run cannot
  INITIALIZE a delta (Q6): the bank is empty or sealed elsewhere.

### Interfaces

- **`needs_of(spec)`** — the one place a spec becomes daemons; `plan_daemons`
  becomes the executor of needs and keeps its name for the callers that have
  it. A `DaemonNeed` is a typed record (STYLE rule 6): the daemon class, its
  plan kind, the pool names and whether it admits the learner, and the
  buffer.
- **`run_done(store, run_id)`** — beside `peek_ledger` in `data/stores/base.py`
  or in `observe/`? It is a read over the manifest, the ledger and the
  rollouts keys, used by the desk (runner), the observer and the host — the
  data layer is the one place all three may import (Q3).
- **The gate** — one new check, named above. `check_traffic_routes_to_
  declared_pools` already demands `main` when `gen` exists; a generation-only
  run has `gen`, so it is covered.
- **The wire and the desk** — unchanged frames: `demands_of` sets `anchor`
  on a different demand; the desk never knew why.
- **`observe/`** — a run whose extent is rollouts shows `rollouts sealed /
  planned` where a training run shows `committed / planned`; the ledger
  panels are empty and say so.

### Sketches

```python
# runner/loop.py — the run as needs; the experiment as the derivation
@dataclass(frozen=True)
class DaemonNeed:
    daemon: type[Daemon]
    plan: str                          # "train" | "rollout"
    pools: tuple[str, ...]             # residents it admits, by pool name
    learner: bool = False              # admits the learner
    buffer: int | None = None          # Generator only: None = unpaced

def needs_of(spec: ExperimentSpec) -> tuple[DaemonNeed, ...]:
    """The experiment's daemons, read off its spec: a Trainer iff algo, a
    Scorer iff the pipeline has a pooled half, a Generator iff a rollout
    plan — paced by the lag when a Trainer consumes it, unpaced otherwise."""

# spec/specs.py
@dataclass(frozen=True)
class Plans:
    train: str | None                  # None: nothing trains; the run's extent is its rollouts
    rollout: str | None = None
    @property
    def extent(self) -> str: ...       # "train" if self.train is not None else "rollout"

# data/stores/base.py — done-ness for every run kind, one predicate
def run_done(store: Store, run_id: str) -> bool:
    """Train extent: ledger tail >= wave_count(train plan). Rollout extent:
    rollouts/<wave_count(rollout plan)> is sealed. No plan on record: work."""

# runner/refs.py — one more location
#   store://<run_id>/rollouts/<r>     another run's sealed rollout r
```

## Questions

**Q1. An unconsumed rollout plan is UNPACED: `buffer=None` and
`may_generate` is always true. Or should a Generator with no Trainer still
pace itself against its own sealed rollouts (an in-flight bound)?**
Recommendation: unpaced. The lag buffer exists to bound staleness against a
moving policy; with no Trainer the policy never moves and staleness is not a
thing. Throughput is bounded by `max_inflight` and the engine, as it is for
the Generator today within the buffer.
If the other branch: `buffer` stays an int and `committed()` is replaced by
"rollouts sealed", which pins the Generator to a sequential one-wave-at-a-
time shape for no reason the store needs.

> **Samarth:**

**Q2. The anchor of a run is a choice — REFOLDED by Part A.** With the
learner routable, no demand HAS to anchor: `demands_of(spec, anchor)` marks
the learner's host when the spec declares a learner and nothing else is
asked, and `main` otherwise or on request. A learner-less run therefore
anchors on `main` without a special rule, and a pure-client anchor (the
submitting process) stays refused for the reason given before: the run
would die with the client, unseen by the desk's supervision.

> **Samarth:** (answered by the 2026-09-04 instruction — the learner is
> fully remote; Part A implements this fold)

**Q3. `run_done` lives in the data layer, beside the peeks.**
Recommendation: yes — the desk, the host and the observer all read it, and
`data/` is the one package all three import (STYLE rule 8; `observe/` may
import nothing else). It is a read over manifest + ledger + rollouts keys,
estimator-free, which is the data layer's charter.
If the other branch (in `runner/`): the observer cannot import it and grows
a copy, which is the duplication this ADR exists to remove.

> **Samarth:**

**Q4. `Plans.train` becomes optional and `extent` is a derived property, so
no hashed record gains a field.**
Recommendation: yes. A rollout-only run writes `train=None`; the fingerprint
already renders a missing plan as `-`; the manifest's `plans` copy (I11)
carries whatever exists. Identity of every existing run is untouched
(promise 1). The alternative — an explicit `extent` field with a default —
would change canonical bytes for every existing spec unless the default is
omitted from the row, which is the same trick ADR 0005 uses for the window
and is fragile twice.
If the other branch: a migration of every sealed manifest's spec row, or a
decoder rule that tolerates the absence — the decoder already tolerates
extra fields (#70), so absence is the cheaper direction anyway.

> **Samarth:**

**Q5. The ref `store://<run_id>/rollouts/<r>#<i>` — and a Generator-only run
never writes `waves/`.**
Recommendation: yes to both. Waves are the Trainer's realization of a plan
(group keys assigned at assembly); a rollout is the Generator's sealed
output under the ROLLOUT plan's keys; a replay into another run's rollouts
gets THIS run's group keys at `realize`, exactly as replaying its waves
does. The submit gate checks a store ref resolvable before the run starts,
so a rollout that does not exist yet is a Phase-0 refusal, as for waves.
If the other branch (the Generator also writes waves so the grammar stays
one word): two copies of every rollout in a generation-only run's directory
and a second writer of `waves/` — one writer per artifact is the rule.

> **Samarth:**

**Q6. A learner-less run's bank is EMPTY or SEALED: frozen entries only,
their payloads from a `WarmStart` source, refused at the gate otherwise.**
Recommendation: yes for v1. Building a delta needs `AdapterType.params`,
which needs torch, which the runner's side of a serve-only host may not
have; a sealed source's blobs are already the payloads and `_warm_start`
already reads them. This is also the case ADR 0005 wants (serve a trained
steer to generate with it) and the ordinary teacher case (an empty bank).
The gate check names the rule so a trainable entry without a learner is
refused as text, not as `None.install`.
If the other branch (build deltas on the runner's host): torch becomes a
runner-side dependency of generation-only runs, and zero-init-only would be
the honest limit (a seeded init needs the same code path a learner runs).

> **Samarth:**

**Q7. `RunReport.updates_completed` becomes `completed` + `extent`; the host
journal's attach event records the extent plan; the roster's
`updates_completed` follows.**
Recommendation: yes. These are observability (journal, roster, report) and
never identity, so the rename costs a UI label and a few tests. Keeping
"updates" for a run that has none would make the observer lie.
If the other branch: keep the field names, define `updates_completed` as
"extent completed", and accept the word.

> **Samarth:**

**Q8. Crash and resume, stated once.** A generation-only run resumes by
`already_sealed` skipping every rollout on disk — the Generator's existing
rule — and re-running Phase 0/1 (identity, manifest match, dictionary, an
empty or sealed bank re-compiled to the same content-addressed bundle). A
crash mid-rollout leaves nothing: `write_rollout` is atomic. The desk's
reaper, seeing the anchor host die, parks the run and re-delivers it; the
new host's adoption is the same resume. The one obligation this ADR adds:
`run_done` must read true at the last sealed rollout WITHOUT a ledger, so
the reaper stops re-delivering — pinned by a test that reaps a finished
generation-only run's host and asserts nothing is parked. Agree these are
the obligations, and that none requires a Sealer?

> **Samarth:**

## Outcome

Filled at implementation.
