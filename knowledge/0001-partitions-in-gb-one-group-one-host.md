# ADR 0001 — The partition speaks GB, and one group is one host

| | |
|---|---|
| **Date** | 2026-09-01 |
| **Status** | Proposed |
| **Author** | Claude Opus 5 (session: gsm-campaign review) |
| **Touches** | `spec/`, `runner/` (desk, host, campaign), `deploy/`, `tests/` |
| **Invariants** | I3 (identity is computed), I5 (topology is semantics-neutral), I12 (a host is atomic and never reshaped) |
| **CONTEXT** | extends #43 (the fleet), #55 (the vocabulary rename), #68 (the metal plane); the entry number lands at implementation |

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
  `choose_devices` book in GB; `Metal` gains a measuring constructor.
- **Touched** — `runner/campaign.py`: `demands_of` stops reading `sharing` and
  passes `vram_gb` through. This is the one place specs become demands, so the
  unit crossing belongs here or nowhere.
- **Touched** — `runner/host.py`: `Partition.memory` docstring states it is
  DERIVED from a GB demand and is the substrate's unit, not the spec's.
- **Touched** — `deploy/*.py`: every `gpu_config` rewritten; every
  `OwnedMetal(...)` / `register_metal(...)` takes measured facts.
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
  rung by name instead of clamping. The fakes suite is green.
- **Non-promises.** It does not prove the split gsm shape on metal — the
  two-carve variant is exactly what parked before, and it is UNPROVEN until a
  run lands. It does not ship default factories or a default `transport_for`, so
  same-container address resolution stays a deploy's to get right; that is the
  live cause of the park and it deserves its own commit. It does not pin devices
  (Q4). It does not make `total / shape` exact — per-rank overhead (CUDA context,
  NCCL buffers, unsharded activations) does not divide, so the estimate is
  optimistic as `tp` grows and the number is a reservation you pad. It says
  nothing about interconnect topology, multi-node collectives, or links between
  partitions — that is ADR 0002.

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

> **Samarth:**

**Q5. A metal container restarts onto a different card — who wins, the journal or
the device?** `register_metal` refuses a taken name (`desk.py:264`) and
`gsm_a100.py:297` catches the refusal and prints "not re-registered." Under
fractions that was survivable. Under GB it is silent corruption: the desk keeps
the OLD card's `vram_gb` forever while the container runs on a new one, and with
`GPUS = [...]` a restart onto a different kind is routine.
Recommendation: **re-registration UPDATES the row when the address matches**,
journaled as a fresh `metal` event, and `from_journal` replays last-write-wins. A
name+address pair identifies one container's standing claim; its measured facts
are whatever it last reported.
If the other branch: registration stays write-once and a card change requires an
explicit deregister — one more human step in a loop that already pays for GPU
loyalty, and the failure mode when someone forgets is a wrong number rather than
an error.

> **Samarth:**

**Q6. Where does measurement live?** CLAUDE.md requires the fakes suite to be
stdlib-only with torch lazy, and tests construct `Metal("node-a", "L4", 4)`
directly.
Recommendation: **`Metal` stays a plain frozen record; `MetalService.measure()`
is the classmethod that reads the device and constructs one**, used by real
venues only. Measurement is an act, not a field.
If the other branch: measuring inside `Metal.__post_init__` makes every test
construction import torch and breaks the stdlib-only rule.

> **Samarth:**

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

> **Samarth:**

**Q8. Delete `PoolMember.n` with `GpuSet`?** It is read by nobody in `rlstack/`
or `deploy/` — the third inert field, alongside `GpuSet.n` and `GpuSet.nodes`.
Recommendation: **delete it.** Replica count is either a group repeated or a
placement decision; a field nothing reads is a promise the system does not keep.
If the other branch: it stays as documentation of an intent, in the canonical
bytes, hashing into every `run_id`.

> **Samarth:**

**Q9. Does `GpuGroup` keep its name now that one group is exactly one host?**
Recommendation: **rename to `HostSpec`**, and `GpuConfig.groups` → `hosts`. It
matches the `...Spec` suffix every other spec record wears, it makes the
alignment legible at the call site (`HostSpec(members=(...))` reads as "one
host"), and it is free — every deploy's `gpu_config` is being rewritten by this
ADR anyway. "Group" survives where it is still true: the arbiter's exclusive
group.
If the other branch: `GpuGroup` stays and the central claim of this ADR is true
in the code and invisible in the vocabulary — the same asymmetry, one layer up.

> **Samarth:**

**Q10. What does `vram_gb=None` mean?**
Recommendation: **a whole device per shard** — today's meaning of
`fraction=None`, preserved. It cannot be a GB number, because the spec does not
know the card; it stays a sentinel resolved at the metal.
If the other branch: make it required and every unsized deploy refuses at
validate — more honest, and it means no spec runs until it has been sized.

> **Samarth:**

## Outcome

Filled at implementation.
