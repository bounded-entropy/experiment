# ADR 0002 — A resident is a process

| | |
|---|---|
| **Date** | 2026-09-01 |
| **Status** | Proposed |
| **Author** | Claude Fable 5.1 (session: gsm-campaign, the ADR 0001 review) |
| **Touches** | `runner/` (host, desk's MetalService, remote, interfaces, a new residents module), `runner/learners/`, `runner/fakes.py`, `deploy/`, `observe/` (one view), `tests/` |
| **Invariants** | I8 (multi-tenancy on both sides), I12 (a host is atomic and never reshaped), I5 (topology is semantics-neutral), I7 (the substrate is certified) |
| **CONTEXT** | extends #43 (the host), #44 (row routing), #45 (the rank chorus), #51/#52 (sub-GPU hosts, the sleep seam), #53 (bounded rank teardown), #68 (the metal plane), #69 (the blind desk); answers ADR 0001's Q4 with its branch (b); the entry number lands at implementation |

## Original prompt

> can we create an ADR that actually moves vLLM to its own process? i feel like
> this made sense all along (and allows for live monitorability checking too of
> which pools are living which is nice). is there a parallel for torch learners?

> actually before you create an ADR, let

> let's talk through some things:
>
> questions:
>
> 1) is it possible to make learners their own process too?
>
> 2) if we made learners their own process, this means that the trainer daemons
> associated with a learner would have to be in its process. i would like if
> torch learners had a complete parallel with vLLM (continuous batching, etc.),
> that would be nice. but i understand that we cant have the daemon model really
> because the gradients need to stay in the same process to update the relevant
> weights (and the trainer should own that part). so i guess maybe we shouldn't
> make learners their own process? i feel like we should definitely at least
> solve the memory problem with learners in this pass though (like theres no
> torch cap on memory or smth)

> just to be clear, for the requests to the learner, will they have the loss
> stated in the request, etc.? and will the learner look up the loss against its
> own registry? i think this is the clearest boundary, because i feel as if the
> learner should not own any of the stuff relevant to the experiment.
>
> the one issue with this, is that the learner definitely has to emit some sort
> of bundle to the right place, because it has to process whatever its doing in
> place right? so does the form of a request from the trainer to the learner
> actually give it instructions to emit the bundle to a certain place? what is
> the shape of this request

> everything you said makes a lot of sense to me. let's implement

## Context / problem

Three defects, one root: every knob a substrate offers for WHERE a resident
lives and HOW MUCH it may take is process-granular, and our partition is
smaller than a process. A metal container is one process; every host carved on
it is built by a thread of that process (`desk.py:942`, `MetalService.build`),
and every engine and learner those hosts wear shares that process's CUDA
context, allocator, and visible-device list.

**1. "Which" is unenforceable (ADR 0001, Q4).** A `Partition` carries device
indices and a fraction (`host.py:46`); the books choose both by first-fit
(`desk.py:911`). The fraction reaches vLLM as `gpu_memory_utilization`. The
indices reach nobody who can act on them: `VllmEngine.__init__` takes no device
(`vllm_engine.py:49`), its engine args name none (`:81`), and vLLM takes the
first N devices the process can see. The rank chorus assumes rank r lives on
`cuda:r` and says so — "a host is a partition, the container is that
partition, and its device indices are its own" (`ranks.py:21`) — which holds
only for a partition on devices 0..width-1. Only the single-device learner is
pinned, by the deploy (`gsm_a100.py:274`). Carve a partition onto devices (2,3)
of a 4-device metal and the books mark (2,3) while the engine boots on (0,1);
the next carve for (0,1) is accepted and lands on the same pair. Dormant today
because every registered metal is one device (`OwnedMetal(name, "A100-40GB",
1, 40.0)` in all four venues); live on the first 2-device metal.

**2. "How much" is unenforceable for learners.** The Partition docstring says
"torch's `set_per_process_memory_fraction` a cap; both honor this number"
(`host.py:50`). Nothing calls it — the only occurrence in the repo is that
docstring. It could not be called honestly: the cap bounds ONE PROCESS's caching
allocator on a device, so two learner-bearing partitions on one device inside
one metal process share one allocator and the only expressible cap is their
sum, which is not a partition. The engine's fraction IS honored today, but only
because vLLM v1 already runs its EngineCore in a child of the constructing
process (`vllm_engine.py:337`, the shutdown docstring) — a boundary vLLM owns
and we do not: no address, no pin, no liveness, no cap we set.

**3. Liveness is host-granular and inferred.** The desk knows a host is alive
by asking `host.status()` (`desk.py:177`). A resident that OOMed or wedged
inside a living container answers `status` fine — the roster is Python state —
and is discovered only when a verb into it fails, minutes later, from inside a
tenant's daemon. `sample_gpu` reads `nvidia-smi --query-gpu` (`host.py:603`):
device totals, unattributable across the partitions sharing a device. A
resident with its own pid is what `nvidia-smi` attributes memory to, and what a
supervisor can `poll()`.

**Two facts about the boundary itself.** First, the seam already exists and is
already crossed by processes: the Learner protocol is five verbs
(`interfaces.py:176`) whose arguments are plain data by construction — the
chorus's own record says its args "only ever carry what the Learner protocol
already passes … nothing here holds a live handle" (`ranks.py:57`) — and
`FsdpTorchLearner` broadcasts every `TokenBatch` to ranks 1..width-1 on every
microbatch (`fsdp_torch.py:66`). The Trainer never holds a tensor: its train
phase is flatten, pack, three verb calls, then blobs, bundle, ledger
(`trainer.py:157`). Loss and backward run inside `forward_backward` on the
learner's side (`torch_learner.py:136`); `emit` returns bytes and the Trainer
decides where they go. The engine side has its wire built: `RemotePool`
implements the full Engine protocol over a `Transport` (`remote.py:340`) and
`HostService` answers it (`remote.py:226`). There is no `RemoteLearner`.

Second, the seam is drawn one step too wide: `install` receives the whole
`ExperimentSpec` (`torch_learner.py:77`, `fsdp_torch.py:59`, `fakes.py:217`)
and reads exactly six things off it — the base, the loss name, each bank
entry's adapter type / init / trainable flag, the master seed (to derive an
init seed, `torch_learner.py:40`), and the optimizer settings. The chorus
broadcasts the full spec to every follower. A process cut along this seam must
carry a typed projection, or the learner keeps owning experiment facts.

**One more fact, about alternation.** `VllmEngine.sleep`/`wake`
(`vllm_engine.py:312`) are "THE evict verb an alternating host's arbiter hook
calls" — and on this branch nothing wires them: no `wake=`/`evict=` appears in
`rlstack/` or `deploy/`, and `Host._attach_regimes` (`host.py:218`) attaches
without hooks. Alternation here is admission-only. Across a process boundary
these verbs must become door verbs on the resident, which is also the moment to
wire them.

*Unmeasured:* the per-microbatch cost of a `TokenBatch` frame over a local
pipe, and the per-update cost of the emitted bytes. Both are bounded below by
what the chorus already pays over NCCL's object broadcast.

## Decision

**A RESIDENT is a PROCESS.** Every engine and every learner a host wears is a
supervised child of the metal's container process, born with
`CUDA_VISIBLE_DEVICES` equal to its partition's devices and a memory cap equal
to its partition's fraction (vLLM: `gpu_memory_utilization`, as today; torch:
`set_per_process_memory_fraction` on every visible device, before the base
loads), answering at an address over a local `Transport`. The Host stays in the
metal process as the DOOR — arbiter, roster, runner and daemons, journal — and
holds proxies: `RemotePool` over the local transport for each engine (the class
it already is) and a new `RemoteLearner` for its learner. The rank chorus
becomes internal to the learner resident: rank 0 IS the resident, its followers
are its children, and `CUDA_VISIBLE_DEVICES` makes "rank r on `cuda:r`" true by
construction. The Learner protocol's `install` takes a typed `Parameterization`
instead of an `ExperimentSpec`, and the learner package imports no spec class.
`MetalService`'s books hold process handles; `decarve` is the teardown ladder;
a dead resident is a dead host (I12). `describe()` and `status()` name each
resident with its pid and whether it answers. The Trainer does not move.

ARCHITECTURE.md's "Resident / daemon" entry says the Trainer sits beside its
Learner "because the autograd arc and the seal cannot cross a wire". Half of
that is a misattribution: the arc is the Learner's and the seal is the
Trainer's, and neither needs to cross — the five verbs' bytes do. The entry is
recoded at implementation.

Deliberately NOT in this ADR: the cross-tenant coalescer (#44's design — a
scheduler behind the learner's door, which this boundary makes a real door and
otherwise does not touch); a local tokenizer beside `RemotePool` (this makes it
due, not done); the memory UNIT (ADR 0001 — the cap takes whatever
`Partition.memory` is, a fraction today); interconnect topology.

### Touched / untouched

- **Touched** — `runner/interfaces.py`: `Parameterization` and `EntryInstall`
  records beside `TokenBatch`/`TrainStats`/`Emitted`; `Learner.install(tenant,
  parameterization)`. The protocols and the records that cross them live here.
- **Touched** — `runner/remote.py`: `RemoteLearner` (implements `Learner` over a
  `Transport`), `LearnerService` (serves the five verbs), `EngineService` (the
  engine-verb half of today's `HostService.serve`/`answer`, admission-free —
  admission stays in `HostService`, which now delegates through the proxy), the
  JSON codecs for `TokenBatch`/`TrainStats`/`Emitted`/`Parameterization` (bytes
  as base64, the way `encode_bundle` already does at `:99`), and a multiplexed
  local transport. This file IS the wire.
- **Touched** — `runner/residents.py` (new): `ResidentBirth` (what a child is
  told), `Resident` (spawn, the local transport, `alive()`, `stop()`), the two
  child mains (engine, learner), and the teardown ladder lifted from
  `ranks.py`. Rule 8: `runner/` is the substrate — "who may occupy the metal";
  stdlib `multiprocessing` at module scope, torch only inside the child mains.
- **Touched** — `runner/host.py`: `Host` takes proxies; `_attach_regimes` wires
  `evict`/`wake` to the resident door (Q8); `attest_regimes` reads build facts
  off the proxies (`RemotePool.tp`, `RemoteLearner.fsdp`, both from the hello
  frame); `status()` grows `residents`; the `host-up` event carries their
  labels and pids; the Partition docstring's claim becomes true.
- **Touched** — `runner/desk.py`, `MetalService` only: `build` spawns one
  resident per regime instead of calling factories in a thread; the books hold
  `Resident` handles; `decarve` runs the ladder; a resident's exit decarves its
  host (Q7); `describe()` reports residents. `Metal`, `Desk`, `carve`'s
  booking rule and the journal are untouched.
- **Touched** — `runner/learners/torch_learner.py`, `fsdp_torch.py`: `install`
  reads a `Parameterization`; `_init_seed` moves to the runner (`loop.py`);
  the `ExperimentSpec` import goes; `_follow_rank` caps its own device.
- **Touched** — `runner/learners/ranks.py`: the ladder is imported from
  `residents.py` rather than owned here; the header docstring's "the container
  is that partition" becomes a statement of fact.
- **Touched** — `runner/loop.py`: builds the `Parameterization` off the spec at
  Phase 1 (`:170`) — the one place a spec becomes an install — and derives init
  seeds there.
- **Touched** — `runner/fakes.py`: `FakeLearner.install` signature; both fakes
  are already stdlib and picklable by module reference, so they run inside a
  child unchanged otherwise.
- **Touched** — `deploy/*.py`: factories become BUILDERS a child can import by
  qualified name (Q4); `release=` disappears (decarve is process teardown);
  `@modal.exit` tears residents down through the ladder; the Modal-facing
  routers (`host`/`host_ask`/`metal`/`metal_ask`) are unchanged.
- **Touched** — `observe/views.py`: `hosts_data`/`render_hosts` show the
  resident rows the `host-up` event now carries. `observe/web/` renders them
  on the hosts page — one row per resident, no new page.
- **Touched** — `ARCHITECTURE.md`: Resident, Rank chorus, Resident / daemon,
  Host, MetalService entries recoded to the shape above.
- **Touched** — `tests/`: `test_host`, `test_desk`, `test_remote` (the
  learner's byte-identity test in the shape of `test_a_remote_main_pool_is_
  byte_identical_to_local`), `test_ranks`, `test_architecture` (one new edge:
  `runner/learners/` imports nothing from `rlstack.spec`).

- **Untouched** — `runner/daemons/` (`trainer.py`, `scorer.py`, the generator
  and evaluator): they hold an `Engine` and a `Learner` by protocol and the
  proxies satisfy it; the Trainer's train phase is verb calls on plain data.
  This is the load-bearing entry: if a daemon needs to change, the boundary
  was drawn in the wrong place.
- **Untouched** — `runner/engines/vllm_engine.py`: device selection is the
  process's, the fraction is already the constructor's, sleep/wake/shutdown
  already exist. Claimed here; checked at implementation.
- **Untouched** — `runner/arbiter.py`: residents are keyed by object identity,
  and the proxy is the object; hooks are async callables, and the door verbs
  are async callables.
- **Untouched** — `spec/`, `spec/canonical.py`: machinery names are not hashed
  (#55); no field of the canonical tree moves. `run_id` is unchanged for every
  existing run.
- **Untouched** — `data/stores/`: no new store key, no new journal file. The
  `host-up` event gains a field in a host journal, outside every run dir.
- **Untouched** — `runner/remote.py`'s `RemotePool`, `RemoteHost`,
  `RemoteMetal`, `RemoteDesk`, `LocalTransport`: `RemotePool` gains no code
  and one more caller (the Host itself); the rest never see a resident.
- **Untouched** — `runner/desk.py`'s `Desk`: listings stay host-granular; the
  desk never learns the word resident (see non-promises).
- **Untouched** — `runner/residency.py` (`BundleResidency`): lives inside the
  engine resident, exactly as it lives inside the engine object today.
- **Untouched** — `policy/`, `training/`, `inference/`, `rlstack_engine/`: the
  two worlds and the plugin do not know what a process is.
- **Untouched** — `tests/test_resume.py`: it computes run ids from reports and
  hardcodes none; resume-equivalence is a property test and must stay green
  without edits.

### Promises / non-promises

- **Promises.** A resident's process sees exactly its partition's devices: the
  child's hello frame reports its visible device count and the metal refuses
  the build when it differs from `len(partition.devices)`. A learner resident's
  allocator is capped at `partition.memory` on every visible device before the
  base loads; an allocation past it is an OOM in THAT process, killing that
  resident and never a neighbor. A run whose engine and learner are process
  residents is byte-identical, on fakes, to the same run in-process. The
  learner package imports no spec class, pinned by `test_architecture`. A dead
  resident is a dead host: the metal decarves it, frees its booking, and the
  desk's next probe reaps the listing. `status()` and `describe()` name each
  resident with pid and liveness, and `nvidia-smi` attributes memory to it.
  Teardown is bounded: the ladder's budget per resident, compounded once for a
  learner's chorus. Every venue keeps working on 1-device metals. The suite is
  green.
- **Non-promises.** It does not build the coalescer. It does not add a local
  tokenizer — `engine.tokenize` per trajectory becomes a process round-trip,
  measured in the Outcome and fixed in its own commit if the number says so. It
  does not change the memory unit. It does not prove a multi-device metal with
  split partitions on metal until a 2+-device venue runs: the pin is asserted
  by device count, the topology is not. It does not teach the desk about
  residents — listings stay host-granular, and a host with a dead resident is
  reaped as a dead host. It does not restart a dead resident in place (Q7). It
  does not change vLLM's own internal process structure. It does not make the
  learner's verbs awaitable (Q6).

### Interfaces

The Learner protocol keeps five verbs; `install` takes a `Parameterization`.
Two new wire pairs mirror the one that exists: `RemoteLearner`/`LearnerService`
for the learner, `RemotePool`/`EngineService` for the engine — `RemotePool`
unchanged, `EngineService` being the engine-verb half of today's `HostService`
with admission left behind in `HostService`, which now admits and then forwards
through the proxy. Admission is unchanged: it happens once, at the serving
host's arbiter, and traffic is counted at that seam as before. The resident
door adds verbs on no protocol: `hello` (birth facts — kind, base, `tp` or
`fsdp`, `sleeps`, devices seen, cap applied, pid), `sleep`/`wake` (engine
residents that report `sleeps`), `stop`. `MetalService.describe()["hosts"][h]`
gains `residents`; `Host.status()` gains `residents`; the `host-up` journal
event carries `residents: [{label, pid}]`. `observe/` sees one more row per
host. Nothing on the fleet journal's `list`/`metal`/`provision` rows changes.

### Sketches

```python
# runner/interfaces.py — the install crossing, typed; the learner imports no spec class
@dataclass(frozen=True)
class EntryInstall:
    name: str
    adapter_type: str                    # an ADAPTER_TYPES key, resolved on the learner's side
    init: Mapping[str, Any]              # "seed" already derived by the runner; never the master seed
    trainable: bool
    sites: tuple[SiteMeta, ...]          # resolved off the schema by the runner, as today

@dataclass(frozen=True)
class OptimSettings:                     # today's OptimSpec fields, projected
    name: str
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    overrides: Mapping[str, Mapping[str, object]]

@dataclass(frozen=True)
class Parameterization:
    base: str
    loss: str                            # a LOSSES key, resolved on the learner's side
    entries: tuple[EntryInstall, ...]    # install order
    optim: OptimSettings

class Learner(Protocol):
    fsdp: int
    def install(self, tenant: str, parameterization: Parameterization) -> None: ...
    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats: ...
    def optim_step(self, tenant: str) -> None: ...
    def emit(self, tenant: str) -> Emitted: ...
    def load(self, tenant: str, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None: ...


# runner/residents.py — one process wearing one regime of one host
@dataclass(frozen=True)
class ResidentBirth:
    label: str                           # "<host>:<regime>"
    partition: Partition
    regime: Regime
    builder: str                         # "package.module:function", importable in the image (Q4)
    venue: Mapping[str, Any]             # JSON-safe venue facts the builder takes (store locator, caches)

class Resident:
    @classmethod
    def spawn(cls, birth: ResidentBirth) -> "Resident":
        """Spawn (never fork), pin CUDA_VISIBLE_DEVICES, await hello, refuse on
        a device-count mismatch. Returns with the door open."""
    transport: Transport                 # multiplexed, local (Q5)
    hello: dict                          # the birth facts the child reported
    def alive(self) -> bool: ...
    def stop(self, *, grace_s: float, signal_grace_s: float) -> Teardown: ...   # ranks.py's ladder, lifted

def learner_main(birth: ResidentBirth, conn) -> None:
    """The child: cap every visible device at partition.memory BEFORE the base
    loads, lead the chorus at fsdp=regime.shape, serve LearnerService frames
    until stop, end the chorus, exit."""

def engine_main(birth: ResidentBirth, conn) -> None:
    """The child: build the engine at gpu_memory_utilization=partition.memory,
    tp=regime.shape, serve EngineService frames concurrently on its own loop
    until stop, shut the engine down, exit."""


# runner/remote.py — the learner's wire, the shape RemotePool already has
class RemoteLearner:
    def __init__(self, transport: Transport, *, fsdp: int) -> None: ...
    # the five verbs, each one frame; forward_backward blocks like today (Q6)

class LearnerService:
    def __init__(self, learner: Learner) -> None: ...
    def answer(self, verb: str, payload: dict) -> dict: ...


# runner/desk.py — MetalService.build, after
def build(self, name, regimes, devices, memory) -> Host:
    partition = Partition(self.metal.name, devices, memory, self.metal.gpu)
    residents = [Resident.spawn(ResidentBirth(f"{name}:{r.name}", partition, r,
                                              self.builder_for(r), self.venue))
                 for r in regimes]
    engines = [RemotePool(res.transport, base=res.hello["base"], tp=res.hello["tp"])
               for res in residents if res.hello["kind"] == "inference"]
    learner = next((RemoteLearner(res.transport, fsdp=res.hello["fsdp"])
                    for res in residents if res.hello["kind"] == "training"), None)
    return Host(name, engines=engines, learner=learner, residents=residents, ...)
```

## Questions

**Q1. Engines only, or engines and learners?** The prompt's second message
leans toward leaving learners in-process because "the gradients need to stay in
the same process to update the relevant weights".
Recommendation: **both, one mechanism.** The gradient never leaves the learner
today: loss and backward run inside `forward_backward`, and the Trainer calls
five verbs on plain data. The concern is real and already satisfied by the
protocol's shape. And the memory promise cannot be kept any other way — the
torch cap is per process, so a learner in the metal process is a learner whose
partition fraction is a docstring. Building the supervisor for engines alone
leaves the learner as the one resident the books cannot enforce.
If the other branch: engines get pinned and capped, the learner keeps the
metal process's whole allocator, and `Partition.memory` stays a claim for half
of every alternating host.

> **Samarth:**

**Q2. The learner's boundary: a typed `Parameterization` at install, loss bound
there and resolved in the learner's registry, and the learner imports no spec
class.** Today `install` receives an `ExperimentSpec` and reads six things off
it. The chorus broadcasts the whole spec.
Recommendation: **the projection above, built by `loop.py` at Phase 1 — the
one place a spec becomes an install — with the init seed derived there so the
master seed never crosses. The loss stays bound at install, not stated per
`forward_backward`: the learner is stateful per tenant regardless (moments,
accumulated grads), an experiment has one algorithm, and a per-request loss
would add the invariant that every microbatch of one update agrees. The acid
test is #69's, applied to `runner/learners/`: imports nothing from
`rlstack.spec`, pinned in `test_architecture`.** Both processes run one image,
and registry strings plus class sources already hash into `run_id` (#55), so
the learner-side lookup is attested, not assumed.
If the other branch: the spec keeps crossing and the learner keeps reading it —
the boundary exists in the process table and not in the code.

> **Samarth:**

**Q3. Emitted bytes on the wire, or the learner writes the cas and returns
addresses?** `emit` returns `{entry: bytes}` for adapters and moments; the
Trainer writes them under `adapters/<name>@v` and `optim/<name>@v`, compiles
the bundle, registers it, commits the ledger. The engine's pattern for bundle
payloads is a `cas_get` resolver handed in at build.
Recommendation: **bytes on the wire, base64 in the frame as `encode_bundle`
does.** The learner stays store-blind, which is the whole point of Q2; the
cost is one transfer per update — tens to a few hundred MB over a local pipe,
sub-second — measured in the Outcome and revisited only with a number.
If the other branch: the learner holds a store handle and a write path, and
"where the bytes go" is decided on both sides of the boundary.

> **Samarth:**

**Q4. How does the child learn to build its resident?** Today a venue hands
`MetalService` lambdas (`engine_factory=lambda regime, partition:
VllmEngine(...)`, `gsm_a100.py:268`) that close over a store, a scheme, and a
transport. A spawn-context child cannot receive a lambda, and importing a
deploy module inside a child executes its Modal app definition at import.
Recommendation: **builders by qualified name plus JSON-safe venue facts.** Each
venue exposes module-level `build_engine(regime, partition, venue)` and
`build_learner(regime, partition, venue)` in an importable module with no app
object in it (a `deploy/<venue>_builders.py`, or the existing module split so
its builders import clean); `ResidentBirth.builder` names one; `venue` carries
the store locator, the HF cache path, the engine's capacity knobs. The child
constructs its own `Store` from the locator — `cas_get` is then the child's
own. No callable ever crosses.
If the other branch: pickled callables — only module-level functions pickle by
reference anyway, so this is the same design with the name hidden inside
pickle's opcode stream and the app import as a side effect.

> **Samarth:**

**Q5. The engine resident's transport must multiplex.** An engine serves many
in-flight requests at once — a Generator's `sample_tokens` and a Scorer's
`score_tokens` overlap, and I8 says requests batch across tenants. A pipe that
carries one frame at a time would serialize sampling and destroy continuous
batching.
Recommendation: **a request-id-multiplexed local transport (a Unix socket or a
pipe pair, one reader task) and a child that dispatches frames concurrently on
its own asyncio loop — the loop vLLM's async engine already needs
(`vllm_engine.py:122`). The learner resident serves one frame at a time on the
same transport class: its verbs are sync and ordered, and multiplexing costs it
nothing.** The desk, HostService and every daemon see `Transport.call`/`ask`
exactly as before.
If the other branch: one frame in flight per resident, and the throughput
parity #45 proved over `RemotePool` is lost inside the container.

> **Samarth:**

**Q6. Learner verbs stay synchronous and blocking, or become awaitable?**
Today `forward_backward` is a sync call on the host's event loop, so the loop
blocks for the length of a gradient step; `status` still answers because
`answer` reads Python state without the loop. Across a process boundary the
verb COULD be awaited and the loop kept responsive.
Recommendation: **synchronous and blocking in this ADR — today's semantics
exactly, byte-safe by construction. The awaitable form is a one-line change
once the boundary exists and gets its own commit, with the interleaving it
permits (other daemons' work during a step) reasoned about on its own.**
If the other branch: the boundary and the scheduling change land together, and
a resume-equivalence difference has two suspects.

> **Samarth:**

**Q7. A resident dies — an OOM in the learner, a wedged engine core. What is
the host, and who notices?** The books are per host (I12: atomic, never
reshaped). Today a death inside a living container is discovered by the next
verb that fails, from inside a tenant.
Recommendation: **a dead resident is a dead host. `Resident.spawn` starts a
watcher (a thread on the child's `join`) that calls back into `MetalService`,
which runs the ladder on the host's other residents, frees the booking, and
leaves the listing for the desk's probe to reap (`Listing.alive()` now fails
because `service_for` no longer answers). A verb in flight when the child dies
raises in the transport, the daemon fails, the run fails — as today; the
watcher's decarve and the verb's failure are both idempotent
(`decarve` on a missing host returns `decarved: False`). The tenant's state is
in the store; its restart is a re-submit that re-carves, which is resume.**
If the other branch (restart in place): the metal rebuilds the resident on the
same partition, re-installs every tenant, re-adds every bundle, and journals
none of it — a second lifecycle hidden from the desk and the observer.

> **Samarth:**

**Q8. Alternation crosses the door, and gets wired.** The arbiter's `evict`/
`wake` hooks are async callables; the engine's `sleep`/`wake` exist and are
wired by nothing on this branch.
Recommendation: **`Host._attach_regimes` wires `evict=resident.sleep` and
`wake=resident.wake` for every engine resident whose hello reports
`sleeps: true`, over the door. Learners attach with no hooks: a learner cannot
hand memory back today, and an alternating host with a learner relies on the
engine side's sleep — exactly today's #52 mechanism, now reachable.** The
switch still happens only at zero in-flight work, so no frame meets a
half-woken engine.
If the other branch: alternation stays admission-only, and a multi-member
group under ADR 0001 sizes for both residents resident at once.

> **Samarth:**

**Q9. The teardown ladder moves.** `Teardown`, `stop`, `end_the_children`,
`escalate` (`ranks.py:79`, `:227`, `:279`, `:294`) are the repo's one bounded
way to end a process that holds a GPU. A resident needs the same ladder; a
learner resident needs it twice (its own chorus inside, itself outside).
Recommendation: **lift the ladder into `runner/residents.py`; `ranks.py`
imports it. A learner resident's `stop` is: `stop` frame (the child ends its
chorus with its own ladder and exits) → SIGTERM → SIGKILL, each rung joining
what it signalled; the outer budget is the inner budget plus one grace.**
If the other branch: two ladders, and #53's lesson — a lost report from an
unjoined child — is re-learnable in the copy that drifts.

> **Samarth:**

**Q10. Frames are JSON, and byte identity is the test.** The `Transport`
contract says frames are JSON-safe, and `LocalTransport` enforces it with a
round-trip (`remote.py:319`). The chorus ships `TokenBatch` by pickle today.
Recommendation: **JSON frames, honoring the contract: tuples become lists and
back, floats survive Python's repr round-trip losslessly, bytes are base64.
`TokenBatch.token_extras`/`doc_turn_extras` are recording-channel values that
already journal as JSON, so they are JSON-safe by construction — and a value
that is not will now fail loudly at the wire instead of quietly in a pickle.
The pin is one test in `test_remote`'s shape: a run with process residents is
byte-identical to the in-process run on fakes; `test_resume.py` stays green
untouched.**
If the other branch: pickle over the pipe — faster, outside the contract, and
a leak of non-JSON state into a batch goes unnoticed until it reaches a real
wire.

> **Samarth:**

**Q11. Real child processes in the fakes suite.** The suite is stdlib-only and
~5 s. `FakeEngine` and `FakeLearner` are stdlib and picklable by module
reference. A spawn-context child costs 100–300 ms.
Recommendation: **real processes in a handful of tests — the byte-identity
test (Q10), the death-decarves-the-host test (Q7), the device-count refusal,
the ladder — and the in-process `LocalTransport` everywhere else.** The suite
grows by a second or two and the boundary is exercised for real on every run.
If the other branch: an in-process "resident" double, and the first real
process bug is found on a GPU.

> **Samarth:**

**Q12. What the observer sees.** Residents are internal to a host; the desk's
listings stay host-granular.
Recommendation: **`Host.status()` and `MetalService.describe()` gain
`residents: [{label, kind, pid, alive, devices, memory}]`; the `host-up`
journal event carries `residents: [{label, pid}]` so the hosts view can show
them without a probe; the UI's hosts page adds one row per resident. No new
journal event, no new page, nothing in a run dir.**
If the other branch: `describe()` only — visible to the desk, invisible to the
observer, and the "which pools are living" the prompt asked for is a wire call
away instead of on the page.

> **Samarth:**

## Outcome

Filled at implementation.
