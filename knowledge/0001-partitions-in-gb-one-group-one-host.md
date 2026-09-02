# ADR 0001 — The partition speaks GB, and one group is one host

| | |
|---|---|
| **Date** | 2026-09-01 |
| **Status** | Implemented (2026-09-01; CONTEXT #75) — every question answered; Q5b–Q5d and the Q6/Q8–Q10 defaults resolved by the agent at Samarth's delegation |
| **Author** | Claude Opus 5 (session: gsm-campaign review) |
| **Touches** | `spec/`, `runner/` (desk, host, campaign), `deploy/`, `tests/` |
| **Invariants** | I3 (identity is computed), I5 (topology is semantics-neutral), I12 (a host is atomic and never reshaped) |
| **CONTEXT** | extends #43 (the fleet), #55 (the vocabulary rename), #68 (the metal plane); recorded as **#75** |

## Original prompt

> take a look at the gsm-campaign worktree. i noticed smth weird, where there
> was actually one host responsible for both the learner and engine. i think
> this contradicts how i was initially envisioning things: i was envisioning
> things like the learner and engine were on different hosts, since this
> obviously yields better matching if a future jobs wants to join in. i feel
> like a host should consist of a single partition in the sense that if there
> are two very clearly separate gpu use cases for the partition, then they
> should just be different hosts. does that make sense? so i dont think
> concurrent should be a feature in gpu group, cuz why not just make it two
> different hosts at that point? i feel like the gpuset gpugroup, etc.
> primitives should be more aligned with the host primitives. i dont think they
> align up right now nicely, and that's causing some confusion (theres some
> assymetry)

> i feel like the system should speak GB, right?

> cuz obviously 0.625 on an L4 is much different than 0.625 on an H100

And, answering the first four questions of this ADR in session:

> 1) change the name to alternate
> 2) accept the break. i think we should probably delete the entire modal store
> anyway (all the data in it isnt too useful so this should be fine)
> 3) should probably be total across all shards?
> 4) what is the factory??

## Context / problem

Four defects, one root. The spec side (`GpuSet`, `GpuGroup`, `sharing`,
`fraction`) and the metal side (`Metal`, `Partition`, `Regime`, `Host`) were
designed against each other but never made 1:1, so the spec carries provider
facts and the metal carries demand facts.

**1. `sharing="concurrent"` claims a colocation the Host primitive cannot
model.** `5ae6460` made a group place as ONE unit whatever its sharing, so a
concurrent group carves one host wearing both regimes. `desk.py:374` then sizes
that carve at `max(d.memory)`. gsm declares `0.45 + 0.40`
(`deploy/gsm_a100.py:589`); the metal books **0.45** and reports 0.55 residual
while the container holds a vLLM engine at 0.45 plus an uncapped learner. A
second carve for 0.5 is accepted and OOMs. `MetalService` exists to make
double-booking impossible and the fusion walks around it.

That commit's stated reason was a failure — "the anchor self-dialed its own
container per sample, observed parked." That was a venue bug, not a primitive
bug: `fleet_a100.py:244` resolves same-container addresses with
`LocalTransport`, `gsm_a100.py:277` sends them out through Modal RPC and back
into the same container. The primitive was reshaped to route around a one-line
difference between two deploys.

**2. A member's declared memory is discarded.** One `Host` is one `Partition` is
one `memory` number, so `LEARN_FRACTION=0.40` travels to the desk as a Demand,
is collapsed by `max()`, and is applied by nobody —
`set_per_process_memory_fraction` appears only in `host.py:51`'s docstring and
is never called.

**3. A fraction is a provider fact wearing a demand's clothes.** 0.625 of an L4
is 15 GB and of an H100 is 50 GB. `host.py:52` already concedes this, as the
justification for a patch: "`gpu` is descriptive, never decisive: the carve
carries the kind down from the Metal it drew on, because a fraction alone cannot
tell 0.5 of an L4 from 0.5 of an H100." `fraction_for_gb` (`desk.py:61`) names GB
as the human unit and the fraction as the partition's unit — and has **zero
callers**, because the conversion happens at the keyboard instead of at the
metal.

This is live, not hypothetical. `7f3e1ff` opened the gsm venue to
`GPUS = ["A100-40GB", "L40S", "A100-80GB", "H100"]` while the metal still
registers hardcoded `OwnedMetal(name, "A100-40GB", 1, 40.0)`
(`gsm_a100.py:266`, `:293`). Land on an H100 and the books say 40 GB against a
card holding 80. Nothing anywhere calls `get_device_properties` — the metal
*declares* its VRAM and cannot notice it is wrong.

It also costs run identity. `gpus`, `fraction` and `sharing` are all in the
canonical bytes (`tests/test_canonical.py:118`), and `run_id = h(spec ⊕ code ⊕
data)`. The same experiment sized for an L4 fleet and for an H100 fleet is two
different `run_id`s, differing only in where it landed — the thing I5 exists to
forbid. `vram_gb=25` hashes identically on both.

**4. `GpuSet` is inert.** Placement reads none of it: `need_devices` comes from
`tp`/`fsdp` (`desk.py:374`), `need_memory` from the member fractions.
`GpuSet.nodes` and `PoolMember.n` are read by nobody at all. `GpuSet` hashes
into `run_id` and does nothing else — the extra spec noun with no metal-side
counterpart, which is the asymmetry the prompt names.

*Unmeasured:* the throughput cost of splitting gsm's fused host into two carves
on one A100. The wire between them is same-container and should resolve local;
that resolution is a deploy's to get right today (see non-promises).

## Decision

**One `GpuGroup` is one `Partition` is one `Host` is one placement unit, and the
spec states memory in GB.** `sharing` dies as a field and survives as arity: a
group with several members *alternates* them on one partition (today's `sleep`,
one host wearing masks, lag 0); a group with one member is a dedicated host. The
concurrent-on-one-card shape becomes two groups, two carves, two honest
bookings. `GpuSet` is deleted. Members declare `vram_gb`, TOTAL across shards;
the per-device need is `vram_gb / shape`; `fraction_for_gb` converts at the
metal, at build time, and finally has a caller. `Partition.memory` stays a
fraction because both substrates take one. And the metal MEASURES its card at
registration instead of declaring it, because GB against a lying `vram_gb` is
worse than a fraction — it is wrong confidently.

**Since answered (Q5, 2026-09-01): the desk supervises its metals.** The one
restart that is automatic is a METAL's — a Modal preempt takes every host on
it, and the runs must recontinue with no human in the loop. The desk is the
supervisor because it already holds everything the loop needs: the metal's
build recipe (journaled on the `metal` row since ADR 0002 and replayed by
`from_journal`), the archived deliveries (`placements()`, #72), and the move
verb (`reroute`, #72). The loop is reap → knock → re-register → reroute: the
reaper concludes the metal silent, knocks its plane address (on a lazy venue
the knock IS the boot), the reborn container registers itself at bring-up
(Q5a) and the row updates in place (Q5), and every run whose placement was on
that metal is rerouted — re-placed onto whatever fits, the reborn metal
included, and redelivered, which is resume. A HOST dying alone (a resident's
OOM, ADR 0002 Q7) is not auto-restarted: decarve, delist, and a human
resubmit — Samarth's boundary, stated. Q5b–Q5d were delegated and resolved as
recommended: the reaper knocks through the plane address (a `boot_for`
resolver where a knock does not boot), the shift's duties move into the
metal's own process, the desk's recipe row is canon and rides every carve, and
reap composes with reroute, parking what nothing fits and retrying parked runs
on every `metal` registration.

### Touched / untouched

- **Touched** — `spec/specs.py`: delete `GpuSet` and `PoolMember.n`; drop
  `GpuGroup.sharing` and `gpus`; `fraction: float | None` →
  `vram_gb: float | None` on both member types. The spec is where demand
  vocabulary lives, so a unit change is this region's by definition.
- **Touched** — `spec/validate.py`: delete `check_sleep_groups_have_one_learner`
  (arity now says it), `check_sleep_implies_lag_zero` → keyed on member count,
  `check_fractions_fit` (see Q7).
- **Touched** — `runner/desk.py`: `Demand.memory` (fraction) → `Demand.vram_gb`
  (total) and `Demand.sharing` deleted; `placement_units` reverts to one unit per
  group; `provision_unit` computes per-device GB; `MetalService.residual` /
  `choose_devices` book in GB; `Metal` gains a measuring constructor
  (`MetalService.measure()`, Q6).
- **Touched** — `runner/desk.py`, the supervision loop (Q5–Q5d):
  `register_metal` UPDATES a known name at the same address (fresh `metal`
  event; `from_journal` already replays last-write-wins) and reaps that metal's
  listings by probe with zero retries; a different address stays a refusal.
  The desk grows a `boot_for` resolver beside `host_for`/`metal_for`; `reap`
  knocks a silent metal, then reroutes every run whose latest placement was on
  its reaped hosts (`reroute(run_id, avoiding, park=True)`), and a `metal`
  registration retries every parked run. The carve request carries the desk's
  `builds` row and `MetalService.build` builds from it; a re-registration with
  a different recipe updates the row.
- **Touched** — `runner/campaign.py`: `demands_of` stops reading `sharing` and
  passes `vram_gb` through. This is the one place specs become demands, so the
  unit crossing belongs here or nowhere.
- **Touched** — `runner/host.py`: `Partition.memory` docstring states it is
  DERIVED from a GB demand and is the substrate's unit, not the spec's.
- **Touched** — `deploy/*.py`: every `gpu_config` rewritten; every
  `OwnedMetal(...)` / `register_metal(...)` takes measured facts; registration
  moves into bring-up (Q5a) and the shift's stats tasks and commit tick move
  into the metal's own process (Q5b); the "not re-registered" catch goes.
- **Touched** — `tests/test_canonical.py`: the expected canonical literal.

- **Untouched** — `tests/test_resume.py`. It computes `run_id` from the report
  and hardcodes none, so resume-equivalence is a property test and survives the
  identity break intact. This is the load-bearing entry in this list.
- **Untouched** — `runner/loop.py`, `runner/daemons/`, `training/`. The daemons
  receive `Engine` objects and cannot tell local from remote; nothing downstream
  of placement reads a fraction or a group.
- **Untouched** — `runner/arbiter.py`. An alternating group still maps to an
  exclusive arbiter group; only the word that selects it changes.
- **Untouched** — `observe/`. `Partition.row()` still ships a fraction, so the
  hosts view, the device cards and `fleet_data` render unchanged.
- **Untouched** — `runner/remote.py`, `RemotePool`, `HostService`. The wire
  carries addresses and verbs, never a fraction.
- **Untouched** — `spec/canonical.py`. The bytes change because the tree changes;
  the hashing rule does not.

### Promises / non-promises

- **Promises.** The same experiment has the same `run_id` on an L4 fleet and an
  H100 fleet. A metal's residual accounts for every member's declared memory
  (no `max()` collapse). A member's declared size reaches the substrate that
  enforces it, or the build refuses. `sharing` and `GpuSet` do not appear in the
  canonical bytes. A per-rank slice larger than one device raises the acquire
  rung by name instead of clamping. A metal that dies and comes back at the
  same address is re-registered with its measured facts and its corpses
  reaped, with no human step; every run placed on it is rerouted or parked,
  and a parked run is retried on the next registration — all proven on fakes.
  The fakes suite is green.
- **Non-promises.** It does not prove the split gsm shape on metal — the
  two-carve variant is exactly what parked before, and it is UNPROVEN until a
  run lands. It does not ship default factories or a default `transport_for`, so
  same-container address resolution stays a deploy's to get right; that is the
  live cause of the park and it deserves its own commit. It does not pin devices
  (Q4). It does not make `total / shape` exact — per-rank overhead (CUDA context,
  NCCL buffers, unsharded activations) does not divide, so the estimate is
  optimistic as `tp` grows and the number is a reservation you pad. It says
  nothing about interconnect topology, multi-node collectives, or links between
  partitions — that is its own later ADR. It does not observe a real Modal
  preempt: the reap → knock → re-register → reroute loop is UNPROVEN on Modal
  until one happens or is forced. It does not delete the Modal store itself —
  that is an operator's step, named in the Outcome, never run by an agent.

### Interfaces

`fraction_for_gb(gb, metal)` becomes the ONE crossing between the two units and
moves onto the path: the spec and the desk speak GB, `MetalService.build` calls
it once, and everything below `Partition` speaks fractions. The desk still
DEDUCES and the metal still ENFORCES (#68) — only the unit on that wire changes,
so `carve`'s request grows `vram_gb` and loses `memory`, and `residual()` returns
GB free per device. The gate keeps `check_members_match_their_shape`
(`tp`/`fsdp` are still build facts attested against live metal) and loses the
fraction-sum check. `observe/` sees no change: `Partition.row()` still reports a
fraction, because that is what the partition owns.

The supervision loop meets the wire it already has: the `metal` verb accepts a
known name at the same address (update) and refuses it at another (collision);
`carve` requests grow `builds`; `reap`'s verdicts grow `rerouted` and `parked`
per run beside `alive | recovered | reaped` per listing; the desk's resolvers
grow `boot_for(name)`. `observe/` sees the same `parked`/`provision` rows #72
already renders.

### Sketches

```python
# spec/specs.py — one group, one host; GB, total across shards
@dataclass(frozen=True)
class PoolMember:
    name: str
    base: str | None = None
    tp: int = 1
    vram_gb: float | None = None      # TOTAL across the tp shards; None = whole device

@dataclass(frozen=True)
class GpuGroup:                        # renamed per Q9
    """One group is one host. Several members ALTERNATE on its partition
    (exclusive: one resident live at a time, and therefore lag 0); one member
    is a dedicated host."""
    members: tuple[Member, ...]

# runner/desk.py — the books in GB, the crossing at build
@dataclass(frozen=True)
class Demand:
    pool: str | None
    capability: str
    base: str
    shape: int
    vram_gb: float                     # TOTAL; per-device need is vram_gb / shape
    group: int
    anchor: bool = False

class MetalService:
    @classmethod
    def measure(cls, name: str, **kw) -> "Metal":
        """The registered row read off the device, never typed (Q6)."""

    def residual(self) -> list[float]:  # GB free per device
    def choose_devices(self, count: int, gb: float) -> tuple[int, ...] | None:
```

## Questions

**Q1. Does `sharing="sleep"` keep its name as a member-count rule, or get a new
word?**
Recommendation: the word becomes **`alternate`** in prose, docstrings and
ARCHITECTURE.md — a multi-member group ALTERNATES its members on one partition.
"Sleep" named a vLLM mechanism (`enable_sleep_mode`), which is a build fact of
one engine, not a property of a group.
If the other branch: `sleep` stays and keeps pointing at a substrate feature
from a spec-level noun.

> **Samarth:** agree — "change the name to alternate"

**Q2. Run identity breaks. Compat path, or accept it?**
Recommendation: **accept the break.** Every existing `run_id` becomes
unreachable; `tests/test_canonical.py`'s literal is regenerated and nothing else
in the suite hardcodes an id.
If the other branch: a spec-version field in the canonical tree, which is
permanent complexity in the identity rule for a one-time migration.

> **Samarth:** agree — accept the break; delete the entire modal store, the data
> in it isn't useful. (What goes with the volume: run dirs and cas blobs, the
> desk's fleet journal `fleet/gsm.jsonl` — listings vanish and metal containers
> re-register on boot — and the once-fetched dataset rows cache.)

**Q3. Is `vram_gb` per shard or total across shards?**
Recommendation: **total.** Memory demand is a property of the model and the
batch, not of the sharding, so the number must be invariant under changing `tp`
— which is exactly what I5 says a topology knob may not affect. Per-device need
is `vram_gb / shape`.
If the other branch: per-shard needs no division, but every `tp` change forces
re-doing arithmetic on a knob that is supposed to be semantically free.

> **Samarth:** agree — total across all shards

**Q4. Device pinning: whose obligation?** `MetalService.build` hands each factory
a `Partition`, and `partition.devices` is honored by nobody for engines —
`VllmEngine.__init__` takes no device list, so a partition on devices `(2,3)`
still boots vLLM on the first visible devices. The books are honest about *how
much* and silent about *which*.
Recommendation: **(a) state the obligation, pass `devices` explicitly, leave
multi-device metals with split partitions UNPROVEN.** Moving it behind
`MetalService` does not work: the mechanism is `CUDA_VISIBLE_DEVICES`, which is
process-global and read once at CUDA init, and every host on a metal shares one
container. Every venue today is a 1-device metal, so nothing is broken now.
If the other branch **(b) one process per partition** — correct, and a rewrite of
where the books live: `self.hosts` / `self.pending` / `self.services` are
process-local, which is the same ceiling that makes a Metal unable to span
containers. That is its own ADR, not a clause in this one.
*Since drafted:* ADR 0002 ("a resident is a process") takes branch (b) for
engines AND learners, because the torch memory cap is per process too — so
this question is answered there, and here it reduces to: does this ADR
state the obligation in the meantime, or defer the whole of "which" to 0002?
*Closed:* ADR 0002 is Implemented (CONTEXT #74) and landed branch (b) — the pin,
the cap, and the device-count assertion at hello. Nothing remains here.

> **Samarth:** (no answer needed — closed by ADR 0002)

**Q5. A metal container restarts — who wins, the journal or the device, and what
IS a re-registration?** Today `register_metal` refuses a taken name
(`desk.py:264`); every venue catches the refusal and prints "not re-registered"
(`gsm_a100.py:297`, `dsl_a100.py:285`, `fleet_a100.py:313`,
`gsm_sweep_a100.py:304`), and `up` then finds the name in `status()` and reports
the plane held — on the journal's row. Three facts the code adds to the question
as first drafted:

- *The address cannot tell a restart from a second `up`.* The plane address is
  minted from the scheme (`f"{scheme}://metal"`, `gsm_a100.py:294`), so it is
  identical across container generations AND across a double `up` on the same
  live container — the case the "not re-registered" print handles benignly
  today. "Update when the address matches" needs a second discriminator, or it
  must be safe to apply in both cases.
- *The row is the desk's VIEW, not its books.* Under this ADR's own interface
  placement the crossing lives at the metal: the desk deduces from the live
  `residual()` in GB and the metal's build converts against its own (Q6:
  measured) `Metal`, which is fresh at every bring-up. `desk.metal[name]` is
  read by `status()` (`desk.py:706`) and by nothing on the placement path —
  `provision_unit` reads `metal_remotes` only (`desk.py:386`). What a stale row
  corrupts is the view — `status()`, the observer's fleet page, the boot
  instructions — plus any desk-side acquire-rung pre-check this ADR adds
  (`vram_gb / shape` against one device). That check therefore stays at the
  metal, where the card is known.
- *What a restart actually breaks is the generation.* `MetalService.hosts`,
  `pending` and `carves` are process-local: a reborn container is bare with its
  counter reset, while the desk still lists every host carved on it,
  `metal=name`. The code knows this failure and handles it LAZILY —
  `provision_unit` reaps a corpse when a fresh carve re-mints its exact name
  (`test_a_recycled_metals_same_name_carve_reaps_the_corpse`), and `reap`
  concludes silent listings on a 15-minute schedule. A re-registration is an
  EARLIER and STRONGER signal of the same event — the metal says "up, bare" —
  and today it is thrown away.

The fleet journal is outside run identity and outside every run dir
(`stores/base.py:488`), so nothing here bears on resume-equivalence.

Recommendation: **a re-registration of a known name AT THE SAME ADDRESS is the
container generation turning over. The row is overwritten with the frame's
measured facts and journaled as a fresh `metal` event — `from_journal` already
assigns per event (`desk.py:246`), so replay is last-write-wins with no change —
and the desk immediately RECONCILES that metal's listings by probe: the reaper's
own conclusion scoped to one metal, with zero retries, because the metal itself
just said it is up, so a silent host on it is dead, not rebooting. Each corpse
is delisted with `reason="metal re-registered"`; no decarve, the fraction freed
itself when the container did.** Probing, rather than trusting the frame's
snapshot of its books, is what makes a double `up` a no-op (living hosts answer
and stay listed) and what closes the race where the desk carved onto the
newborn before its registration frame landed (the newborn answers too). A
same-name registration from a DIFFERENT address stays a loud refusal: that is
two deploys colliding on a name, not a restart. The "not re-registered"
try/except leaves every venue.
If the other branch: registration stays write-once and a card change requires an
explicit deregister — one more human step in a loop that already pays for GPU
loyalty, and the failure mode when someone forgets is a wrong number in
`status()` rather than an error; the orphaned listings stay the scheduled
reaper's for up to 15 minutes.

> **Samarth:** for auto-restart, there are two cases we need to consider: if a
> host dies, or if the metal dies. i think personally, the only case we should
> actually care about auto-restarting is if the metal dies, because there could
> be a chance that's due to a modal pre-empt (and we should auto-recontinue). in
> that case, the desk needs to be the one to respawn the metal upon preempt. so
> the desk needs to have the build instructions. i think this is fine because
> now, we dont have factory lambdas anymore, so everything should be able to
> live on the desk

*Folded:* the answer widens Q5 from "what is a re-registration" to "the desk
supervises its metals" (the Decision's new paragraph). Three facts make that
cheaper than it sounds. The desk already holds the build instructions: since
ADR 0002 the recipe is journaled on the `metal` row and replayed
(`desk.py:262`, `:280`), so the precondition is met. The desk already has the
move: `reroute(run_id, avoiding, park)` replays an archived delivery onto a
fresh placement — resume, nothing copied (#72). And on Modal a respawn is a
KNOCK: a call to a stopped-but-deployed container boots it, the reaper's own
`recovered` verdict (#68). What is missing is the composition, reap → knock →
re-register → reroute, and Q5's recommendation is the re-register step of it,
now load-bearing: the reborn metal announces itself at the same address, the
row updates, its corpses are reaped by probe. A host death without a metal
death stays manual, as answered. The rest is Q5b–Q5d.

**Q5a. Where does registration run — the shift, or bring-up?** Today it is the
first act of `metal_shift`, inside the `serve` input that `up` spawns once
(`gsm_a100.py:708`); `@modal.enter` (`gsm_a100.py:332`) builds the books and
registers nothing. A container reborn by any other knock — a `host` call, the
reaper's probe, a carve — comes up bare, un-registered, and holding its old row
at the desk; nothing in Q5 fires until a human re-runs `up`. (Modal may
re-deliver the spawned `serve` input to a new container; we do not rely on it.)
Recommendation: **registration moves into `bring_up`, beside the measurement Q6
puts there — the container announces itself the moment it exists, and
`metal_shift` keeps only the stats loop and the volume commit.** Register-at-
birth is exactly the phone-home contract `list_host` already holds hosts to.
If the other branch: registration stays the shift's first act, a knock-booted
metal is an unregistered stranger until the next `up`, and Q5's rule is
reachable only through the human loop.

> **Samarth:** see Q5's answer.

*Folded:* implied yes — a metal the desk respawned has no human to run `up`,
so it must announce itself at bring-up or the loop never closes. What remains
of the shift is Q5b.

**Q5b. Who knocks, and what becomes of the shift?** Today the plane address
resolves through `metal_for(address)`, and a call on it boots a stopped Modal
container; the shift (`metal_shift`: per-host stats tasks and the volume
commit tick) is an input `up` spawns once and nothing restarts.
Recommendation: **the reaper knocks through the plane address it already
holds — `describe()` on the silent metal's `RemoteMetal` — and a venue whose
knock does not boot supplies a `boot_for(name)` resolver beside
`host_for`/`metal_for`, the pattern the desk already uses for every venue
fact. The shift's residual duties move into the metal's own process, started
at bring-up (stats tasks on first carve, the commit tick on the metal's loop),
so nothing depends on a spawned `serve` surviving a preempt.** The desk's own
preempt is already covered: the reaper runs on a `modal.Period` schedule (#68),
and that scheduled call is the knock that boots the desk, which rebuilds
itself from the journal.
If the other branch: the desk learns a second venue vocabulary (shelling out
to `modal`, or spawning `serve` by handle), and a metal whose shift died
quietly stops committing its volume with nobody noticing.

> **Samarth:** i think you can resolve 5b 5c and 5d yourself.

*Resolved (agent, as recommended):* the reaper knocks through the plane
address it holds; `boot_for` for venues whose knock does not boot; the shift's
duties move into the metal's process.

**Q5c. Which copy of the recipe is canon on respawn?** Two exist: the deploy's
constants, re-declared at every bring-up (ADR 0002 Q4a), and the desk's
journaled row.
Recommendation: **the desk's row is canon and rides every carve request; the
metal builds from the request's recipe, and its own constants are only its
FIRST declaration. A re-registration at the same address carrying a different
recipe is a redeploy — a human's act — and UPDATES the row, journaled,
last-write-wins like the card (Q5).** This is what "everything lives on the
desk" means in code: `MetalService.build` reads `request["builds"]`, and
`describe()` reports what it last built from.
If the other branch: constants canon and the row descriptive — the desk cannot
rebuild a metal whose image changed underneath it, and two containers of one
metal name can build different residents for the same regime across a
redeploy with no record of the switch.

> **Samarth:** (delegated with Q5b)

*Resolved (agent, as recommended):* the desk's row is canon, rides every carve,
and a different recipe at re-registration updates it.

**Q5d. Reap → reroute: the auto-recontinue, its park, and its retry.** The
reaper today delists a silent host and stops; the runs on it are orphaned
until a campaign resubmits.
Recommendation: **the reaper's conclusion grows one step. After reaping a
metal's listings it knocks (Q5b), then for every run whose latest placement
was on them runs `reroute(run_id, avoiding=<dead host>, park=True)`: re-place
— a carve on any metal that fits, the reborn one included, since its
registration updated the row and its residual is asked live — and redeliver,
which is resume. Nothing fits → `parked` is journaled, as #72 already does,
and the desk RETRIES every parked run on each `metal` registration event —
the reborn metal's own registration is exactly that trigger, so a knock that
boots slowly still recontinues.** Crash midway: each reroute journals its new
placement, `from_journal` replays it, and a second attempt sees the new
binding in `placements()` with the old host delisted; `stop_anchored` probes
and at most one roster answers, so no run is adopted twice. Byte identity:
#72 already prices a move at one uncommitted update, and resume-equivalence
makes the recontinued run the same run.
If the other branch: reap parks always and a human resubmits — the preempt
costs a human wake-up, which is the case the answer to Q5 rules out.

> **Samarth:** (delegated with Q5b)

*Resolved (agent, as recommended):* reap → knock → reroute with park; parked
runs retried on every `metal` registration; crash-midway idempotence via the
journaled placement and `stop_anchored`.

**Q6. Where does measurement live?** CLAUDE.md requires the fakes suite to be
stdlib-only with torch lazy, and tests construct `Metal("node-a", "L4", 4)`
directly.
Recommendation: **`Metal` stays a plain frozen record; `MetalService.measure()`
is the classmethod that reads the device and constructs one**, used by real
venues only. Measurement is an act, not a field.
If the other branch: measuring inside `Metal.__post_init__` makes every test
construction import torch and breaks the stdlib-only rule.

> **Samarth:** (offered as a default in session, 2026-09-01; no objection —
> taken as agree: the measuring classmethod; `Metal` stays a plain record)

**Q7. Does `check_fractions_fit` convert to GB or get deleted?** It sums a
group's fractions and refuses past 1.0 (`validate.py:397`) — correct for
concurrent members, WRONG for alternating ones, which take turns and may each
legitimately want 0.85. dsl only dodges it by splitting into two groups.
Recommendation: **delete it.** Under one-group-one-host the sum rule is wrong for
the multi-member case, and for separate groups "do these fit on one metal" is
placement's question, answered against a real residual. `validate` holds a
`SiteSchema` and no metal, so it cannot answer it honestly.
If the other branch: keep a GB version and it can only compare a member against
itself, since validate never learns what card it will land on.

> **Samarth:** yea let's just delete the fraction sum check that's fine.

**Q8. Delete `PoolMember.n` with `GpuSet`?** It is read by nobody in `rlstack/`
or `deploy/` — the third inert field, alongside `GpuSet.n` and `GpuSet.nodes`.
Recommendation: **delete it.** Replica count is either a group repeated or a
placement decision; a field nothing reads is a promise the system does not keep.
If the other branch: it stays as documentation of an intent, in the canonical
bytes, hashing into every `run_id`.

> **Samarth:** (offered as a default in session, 2026-09-01; no objection —
> taken as agree: `PoolMember.n` deleted with `GpuSet`)

**Q9. Does `GpuGroup` keep its name now that one group is exactly one host?**
Recommendation: **rename to `HostSpec`**, and `GpuConfig.groups` → `hosts`. It
matches the `...Spec` suffix every other spec record wears, it makes the
alignment legible at the call site (`HostSpec(members=(...))` reads as "one
host"), and it is free — every deploy's `gpu_config` is being rewritten by this
ADR anyway. "Group" survives where it is still true: the arbiter's exclusive
group.
If the other branch: `GpuGroup` stays and the central claim of this ADR is true
in the code and invisible in the vocabulary — the same asymmetry, one layer up.

> **Samarth:** (offered as a default in session, 2026-09-01; no objection —
> taken as agree: `GpuGroup` → `HostSpec`, `GpuConfig.groups` → `hosts`)

**Q10. What does `vram_gb=None` mean?**
Recommendation: **a whole device per shard** — today's meaning of
`fraction=None`, preserved. It cannot be a GB number, because the spec does not
know the card; it stays a sentinel resolved at the metal.
If the other branch: make it required and every unsized deploy refuses at
validate — more honest, and it means no spec runs until it has been sized.

> **Samarth:** (offered as a default in session, 2026-09-01; no objection —
> taken as agree: `vram_gb=None` means a whole device per shard)

## Outcome

Landed 2026-09-01 on gsm-campaign in four commits, recorded as CONTEXT #75.

**What landed.** `spec/specs.py`: `GpuSet` and `PoolMember.n` deleted,
`GpuGroup` → `HostSpec(members)`, `GpuConfig.hosts`, `fraction` → `vram_gb`
(total across shards, None a whole device per shard), `sharing` gone — arity
alternates. `spec/validate.py`: `check_hosts_exist`,
`check_alternation_implies_zero_lag`, `check_post_pools_can_coreside` on
multi-member hosts; `check_sleep_groups_have_one_learner` and
`check_fractions_fit` deleted. `runner/desk.py`: `Demand.vram_gb` +
`per_device_gb`, one unit per HostSpec, `unit_gb`, `carve_request` with the
desk's `builds` row, `MetalService` books in GB (`gb_of` reads back),
`fraction_for_gb` called once in `build`, `per_device_gb(request)`,
`adopt_recipe`, `MetalService.measure`; the supervision loop —
`register_metal` update-in-place + `reconcile_metal`, `boot_for`, `reap` →
`recovers` / `conclude` / `strand` / `knock` / `retry_parked`, `parked()` the
queue, `finished()`, the per-loop `recontinue_lock`; `status()` carries the
metal's address. `runner/campaign.py`: `demands_of` passes GB through.
`runner/host.py`: the Partition docstring, `check_fit` as custody, `capacity`
gone. `runner/loop.py`: `attach_residents` groups by multi-member HostSpec,
remote pools free. `remote.py`'s class table, `rlstack/__init__` exports,
seven venues, the canonical literal.

**What the answers changed.** Q1 put "alternate" in every docstring and
retired "sleep" to vLLM's build fact. Q2's accepted break regenerated one
literal and touched no other test. Q3's "total" put `per_device_gb` on the
Demand and `unit_gb` at the desk. Q5's widening turned a write-once
registration into the re-register step of a loop the desk runs, and Q5d's
park-then-retry made the journal the queue — the crash-midway state is
journaled BEFORE any move. Q5c made the desk's recipe row canon on the carve.
Q6 kept `Metal` a plain record with `measure` beside it. Q7's deletion
removed the last memory arithmetic from the gate and, by the same reasoning,
from `Host.check_fit`. Q9/Q10 as recommended.

**Refinements, stated.** (1) The lag rule keys on a LEARNER alternating with
an engine, not on member count alone: two pools alternating on a host of
their own leave the learner training elsewhere, so a lag buffer is theirs to
choose. (2) `reap` strands only UNFINISHED runs (`finished`: ledger vs the
train plan's wave count, via store peeks) — the ADR's "every run whose latest
placement was on a reaped host" would re-adopt a 40-arm sweep's finished
tenants (an install and an add_bundle each) on every preempt. (3) The
"remote pool in a sleep group is refused" rule went with the one-learner
check that made it coherent: a remote pool attaches free, and alternation is
the serving host's own. (4) `reap`'s reply is three tables (`listings`,
`knocked`, `runs`) rather than one flat dict.

**Tests.** 876 green on fakes (from 868; +10 in `tests/test_desk.py`, the
gate and placement suites recoded), 110 torch-gated skips, ~7 s.
`tests/test_resume.py` untouched and green.

**Unproven on metal.** `MetalService.measure` on a real card (GiB from
`total_memory`); a real Modal preempt through reap → knock → re-register →
reroute; an async `@modal.enter` starting the announce task and the duties;
whether a spawned keepalive reschedules onto the reborn container; the split
gsm shape (two carves, the anchor over the wire into its own container —
exactly what parked before); the per-rank overhead `total / shape` does not
divide; the zombie adoption tasks a fake "container death" cancels but a
real one takes with the process.

**The operator step (Q2), never run by an agent.** Delete the Modal store
volume: run dirs and cas blobs, the desks' fleet journals (`fleet/*.jsonl` —
listings vanish and every metal re-registers itself at boot), and the
dataset rows cache (`measurements/gsm/rows-main.json`). Every pre-#75 run_id
is unreachable under the new canonical bytes.
