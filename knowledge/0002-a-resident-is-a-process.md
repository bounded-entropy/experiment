# ADR 0002 — A resident is a process

| | |
|---|---|
| **Date** | 2026-09-01 |
| **Status** | Implemented (2026-09-01; CONTEXT #74). **Amended 2026-09-04** (Samarth): the learner is a routable resident exactly like an engine, and the Trainer need not share its host or its process — see the Amendment at the end and ADR 0006 Part A |
| **Author** | Claude Fable 5.1 (session: gsm-campaign, the ADR 0001 review) |
| **Touches** | `runner/` (host, desk's MetalService, remote, interfaces, a new residents module), `runner/learners/`, `runner/fakes.py`, `data/stores/` (one opener), `deploy/`, `observe/` (one view, one move), `tests/` |
| **Invariants** | I8 (multi-tenancy on both sides), I12 (a host is atomic and never reshaped), I5 (topology is semantics-neutral), I7 (the substrate is certified) |
| **CONTEXT** | extends #43 (the host), #44 (row routing), #45 (the rank chorus), #51/#52 (sub-GPU hosts, the sleep seam), #53 (bounded rank teardown), #68 (the metal plane), #69 (the blind desk); answers ADR 0001's Q4 with its branch (b); recorded as **#74** |

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

And, answering the twelve questions of this ADR in session:

> Q1) engines and learners.
> Q2) yes, the loss should stay bound at install, because otherwise momentum,
> etc. doesnt make much sense.
> Q3) emitted bytes. eventually, the other mode we will support is NCCL
> directly between two residents (dont do anything too drastic that would
> prevent this from happening in the future)
> Q4) isn't building not a property of deploy or a particular venue, but rather
> a universal property? like, given a specific partition, it's deterministic
> how to build on there, right?
> Q5) sure that works
> Q6) sure let's reason about that later.
> Q7) yes, a restart should go through the entire loop of resubmitting and
> recarving so the metalservice (and by proxy, the desk and observer) knows
> whats up
> Q8) does this imply that a learner can't sleep? why?
> Q9) yes
> Q10) what are frames?
> Q11) sure
> Q12) sure

And the three that stayed open:

> Q4a: just ensure that its stored in a way where runs can be auto-restartable.
> as long as that's satisfied, you can do what you want
>
> Q8a: sure, that's fine
>
> Q10: lets just do JSON.
>
> i think we're ready, please implement

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

**A third fact, about how a resident is built.** Every venue hands
`MetalService` lambdas (`gsm_a100.py:268`): `VllmEngine(regime.base,
tp=regime.shape, gpu_memory_utilization=partition.memory, max_model_len=4096,
max_bundles=32, max_rank=16, max_members=…, cas_get=store.cas_get, serves=…)`
and `TorchLearner(device=f"cuda:{partition.devices[0]}")`. Read them closely
and they are two things interleaved: what follows from (regime, partition) —
the class, the base, the width, the device, the fraction — and what does not:
the capacity knobs (`max_model_len`, `max_bundles`, `max_rank`, `max_members`,
`serves`, sleep mode; a learner's dtype, grad clip, activation checkpointing)
and where the store is. The first kind is universal and lives in a deploy by
accident; the second is a typed record nobody has written, so it travels as
keyword arguments inside a closure that cannot cross a process.

**A fourth, about alternation.** `VllmEngine.sleep`/`wake`
(`vllm_engine.py:312`) are "THE evict verb an alternating host's arbiter hook
calls" — and on this branch nothing wires them: no `wake=`/`evict=` appears in
`rlstack/` or `deploy/`, and `Host._attach_regimes` (`host.py:218`) attaches
without hooks. Alternation here is admission-only: an "alternating" host holds
both residents' footprints at once and only serializes their compute. The
Learner has no sleep verb at all, so even a wired host would keep a second
copy of the base — the learner's frozen one, beside the engine's — resident
through every sampling phase. ADR 0001's Q7 lets alternating members each
declare most of a partition; that is honest only if both hand it back.

*Unmeasured:* the per-microbatch cost of a `TokenBatch` frame over a local
pipe, and the per-update cost of the emitted bytes. Both are bounded below by
what the chorus already pays over NCCL's object broadcast.

## Decision

**A RESIDENT is a PROCESS.** Every engine and every learner a host wears is a
supervised child of the metal's container process, born with
`CUDA_VISIBLE_DEVICES` equal to its partition's devices and a memory cap equal
to its partition's fraction (vLLM: `gpu_memory_utilization`, as today; torch:
`set_per_process_memory_fraction` on every visible device, before the base
loads), answering at an address over a local multiplexed `Transport`. The Host
stays in the metal process as the DOOR — arbiter, roster, runner and daemons,
journal — and holds proxies: `RemotePool` over the local transport for each
engine (the class it already is) and a new `RemoteLearner` for its learner. The
rank chorus becomes internal to the learner resident: rank 0 IS the resident,
its followers are its children, and `CUDA_VISIBLE_DEVICES` makes "rank r on
`cuda:r`" true by construction. The Learner protocol's `install` takes a typed
`Parameterization` instead of an `ExperimentSpec`, the loss bound there by
registry key (Q2: the optimizer's moments belong to one loss), and the learner
package imports no spec class. `MetalService`'s books hold process handles;
`decarve` is the teardown ladder; a dead resident is a dead host (I12), and its
tenants come back only through the whole loop — resubmit, recarve — so the
metal, the desk and the observer agree on what happened (Q7). `describe()` and
`status()` name each resident with its pid and whether it answers. The Trainer
does not move. *(Amended 2026-09-04: it may. The learner's door is reachable
over any Transport, a training demand yields a route like a pool's, and the
Trainer runs wherever the run is anchored — see the Amendment below.)*

**Building a resident is rlstack's, not a venue's (Q4).** `runner/residents.py`
owns `build_engine(regime, partition, build, store)` and `build_learner(regime,
partition, build)`: the class, base, width, device and fraction follow from
(regime, partition) — `tp = shape`; `fsdp = shape`, so a sharded regime leads
a chorus and an unsharded one is a plain `TorchLearner` on `cuda:0`, which
after the pin IS the partition's first device — and the capacity knobs arrive
as typed `EngineBuild` / `LearnerBuild` records. The venue contributes exactly
two JSON-safe values at bring-up: those records and a `StoreAddress` the child
opens its own store from. No callable crosses to a child, and the same
(regime, partition, build) builds the same resident on every venue. The
recipe lives on the metal and is RE-DECLARED at every bring-up from the
deploy's own constants, so a restarted container carves the same residents
with no human step; it is also journaled — on the `metal` registration row
and on every `host-up` event — so a rebuilt desk and the observer can say
what recipe a host was built from (Q4a: runs stay auto-restartable through
resubmit + recarve, and the record of how is durable).

**The door speaks the same verbs to both kinds (Q8).** `sleep`/`wake` are door
verbs for engines AND learners. An engine's are vLLM's level-1 sleep, as today.
A learner's are new: `sleep` moves the frozen base, every tenant's params and
moments to host RAM and returns the allocator's cache; `wake` restores. A
resident's `hello` says whether it `sleeps`, and the Host wires `evict`/`wake`
hooks for every resident that does — which un-orphans the sleep seam on this
branch and makes an alternating host hold ONE base copy at a time. Whether the
learner half lands in this pass is Q8a.

**What is kept open for the NCCL mode (Q3).** Samarth named the next mode:
payloads moving between two residents over NCCL, with no process in between.
This ADR is that mode's precondition rather than its obstacle — each resident
owns its own CUDA context in its own process, which is what a process group
needs — and it keeps three things deliberately loose so the mode lands as a
codec and a verb, not a rewrite: (a) frames are the CONTROL channel and payload
bytes are encoded by ONE codec (`encode_emitted`, beside `encode_bundle`), so
by-reference carriage later replaces the codec while the verbs stay; (b) a
resident has a stable identity (label, pid, door) and the metal process holds
every handle on a host, so it can broker a rendezvous (an address and a port in
a later `hello`) without a new registry; (c) nothing routes payloads THROUGH
the metal process by construction — today's `emit` bytes pass through the
Trainer because the Trainer owns the store and the bundle, not because the
transport requires it. A learner-to-engine `transfer` verb is a later ADR.

ARCHITECTURE.md's "Resident / daemon" entry says the Trainer sits beside its
Learner "because the autograd arc and the seal cannot cross a wire". Half of
that is a misattribution: the arc is the Learner's and the seal is the
Trainer's, and neither needs to cross — the five verbs' bytes do. The entry is
recoded at implementation.

Deliberately NOT in this ADR: the cross-tenant coalescer (#44's design — a
scheduler behind the learner's door, which this boundary makes a real door and
otherwise does not touch); a local tokenizer beside `RemotePool` (this makes it
due, not done); the memory UNIT (ADR 0001 — the cap takes whatever
`Partition.memory` is, a fraction today); awaitable learner verbs (Q6: sync
now, reasoned about later); the NCCL transfer verb; interconnect topology.

### Touched / untouched

- **Touched** — `runner/interfaces.py`: `Parameterization` and `EntryInstall`
  records beside `TokenBatch`/`TrainStats`/`Emitted`; `Learner.install(tenant,
  parameterization)`. The protocols and the records that cross them live here.
- **Touched** — `runner/remote.py`: `RemoteLearner` (implements `Learner` over a
  `Transport`), `LearnerService` (serves the five verbs), `EngineService` (the
  engine-verb half of today's `HostService.serve`/`answer`, admission-free —
  admission stays in `HostService`, which now delegates through the proxy), the
  JSON codecs for `TokenBatch`/`TrainStats`/`Emitted`/`Parameterization` (bytes
  as base64, the way `encode_bundle` already does at `:99`), and the
  multiplexed local transport (Q5). This file IS the wire.
- **Touched** — `runner/residents.py` (new): `EngineBuild`/`LearnerBuild`,
  `ResidentBirth` (what a child is told), the universal `build_engine`/
  `build_learner`, `Resident` (spawn, the local transport, `hello`, `alive()`,
  `stop()`), the two child mains, and the teardown ladder lifted from
  `ranks.py` (Q9). Rule 8: `runner/` is the substrate — "who may occupy the
  metal"; stdlib `multiprocessing` at module scope, torch only inside the child
  mains and the builders (rule 7, lazy).
- **Touched** — `data/stores/`: `StoreAddress` (backend name, root, locator)
  and `open_store(address)` — the one place backends are known by name, which
  is what "one file per backend" already implies. `observe/locate.py`'s
  `store_for` becomes a caller of it. A filing move: a child must open the
  store the metal opened, and the observer's resolver is the wrong region to
  import from `runner/`.
- **Touched** — `runner/host.py`: `Host` takes proxies; `_attach_regimes` wires
  `evict`/`wake` to the door of every resident whose hello says `sleeps` (Q8);
  `attest_regimes` reads build facts off the proxies (`RemotePool.tp`,
  `RemoteLearner.fsdp`, both from hello); `status()` grows `residents`; the
  `host-up` event carries their labels and pids; the Partition docstring's
  claim becomes true.
- **Touched** — `runner/desk.py`, `MetalService` only: takes `builds` and a
  `StoreAddress` instead of factories; `build` spawns one resident per regime
  instead of calling lambdas in a thread; the books hold `Resident` handles;
  `decarve` runs the ladder; a resident's exit decarves its host (Q7);
  `describe()` reports residents and the metal's build recipe. `Metal`, `Desk`,
  `carve`'s booking rule and the journal are untouched.
- **Touched** — `runner/learners/torch_learner.py`, `fsdp_torch.py`: `install`
  reads a `Parameterization`; `_init_seed` moves to the runner (`loop.py`);
  the `ExperimentSpec` import goes; `_follow_rank` caps its own device;
  `TorchLearner.sleep`/`wake` (Q8a decides whether now).
- **Touched** — `runner/learners/ranks.py`: the ladder is imported from
  `residents.py` rather than owned here; the header docstring's "the container
  is that partition" becomes a statement of fact.
- **Touched** — `runner/loop.py`: builds the `Parameterization` off the spec at
  Phase 1 (`:170`) — the one place a spec becomes an install — and derives init
  seeds there.
- **Touched** — `runner/fakes.py`: `FakeLearner.install` signature; both fakes
  are already stdlib and picklable by module reference, so they run inside a
  child unchanged otherwise; a fake `EngineBuild`/`LearnerBuild` is the empty
  record.
- **Touched** — `deploy/*.py`: the lambdas go; each venue passes
  `EngineBuild(...)`, `LearnerBuild(...)` and a `StoreAddress` to
  `MetalService` at bring-up; `release=` disappears (decarve is process
  teardown); `@modal.exit` tears residents down through the ladder; the
  Modal-facing routers (`host`/`host_ask`/`metal`/`metal_ask`) are unchanged.
- **Touched** — `observe/views.py`: `hosts_data`/`render_hosts` show the
  resident rows the `host-up` event now carries. `observe/web/` renders them
  on the hosts page — one row per resident, no new page.
- **Touched** — `ARCHITECTURE.md`: Resident, Rank chorus, Resident / daemon,
  Host, MetalService entries recoded to the shape above; `EngineBuild`/
  `LearnerBuild`/`StoreAddress` defined once.
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
  already exist, and `ServingBuild` (`policy/adapters/rollout.py:31`) stays the
  lowering-facing record it is — `EngineBuild` is what a venue declares,
  `ServingBuild` is what the built engine tells its lowerings. Claimed here;
  checked at implementation.
- **Untouched** — `runner/arbiter.py`: residents are keyed by object identity,
  and the proxy is the object; hooks are async callables, and the door verbs
  are async callables.
- **Untouched** — `spec/`, `spec/canonical.py`: machinery names are not hashed
  (#55); no field of the canonical tree moves; the capacity knobs stay OUT of
  the spec (Q4a) exactly as they are out of it today. `run_id` is unchanged for
  every existing run.
- **Untouched** — `data/stores/base.py` and the backends' bytes: no new store
  key, no new journal file. The `host-up` event gains a field in a host
  journal, outside every run dir.
- **Untouched** — `runner/remote.py`'s `RemotePool`, `RemoteHost`,
  `RemoteMetal`, `RemoteDesk`, `LocalTransport`: `RemotePool` gains no code
  and one more caller (the Host itself); the rest never see a resident.
- **Untouched** — `runner/desk.py`'s `Desk`: listings stay host-granular; the
  desk never learns the word resident, nor a build recipe (see Q4a).
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
  learner package imports no spec class, pinned by `test_architecture`. No
  callable crosses to a child; the same (regime, partition, build) builds the
  same resident on every venue, by rlstack's own code. A dead resident is a
  dead host: the metal decarves it, frees its booking, and the desk's next
  probe reaps the listing; nothing is restarted in place. `status()` and
  `describe()` name each resident with pid and liveness, and `nvidia-smi`
  attributes memory to it. Every resident whose hello says `sleeps` is wired
  into its host's alternation. Teardown is bounded: the ladder's budget per
  resident, compounded once for a learner's chorus. Every venue keeps working
  on 1-device metals. The suite is green.
- **Non-promises.** It does not build the coalescer. It does not add a local
  tokenizer — `engine.tokenize` per trajectory becomes a process round-trip,
  measured in the Outcome and fixed in its own commit if the number says so. It
  does not change the memory unit. It does not prove a multi-device metal with
  split partitions on metal until a 2+-device venue runs: the pin is asserted
  by device count, the topology is not. It does not teach the desk about
  residents or build recipes — listings and capability matching stay as they
  are. It does not restart a dead resident in place. It does not change vLLM's
  own internal process structure. It does not make the learner's verbs
  awaitable. It does not build the NCCL transfer verb, only refrains from
  blocking it. Learner sleep at `fsdp > 1` is UNPROVEN whatever Q8a decides:
  `fully_shard`'s DTensor parameters (`fsdp_torch.py:128`) make offload a
  separate proof, and a sharded learner reports `sleeps: false` until it lands.
  *Retired by CONTEXT #82 (sharded sleep via `reset_sharded_param`): the
  relabel is `FSDPModule._apply`'s own, `sleeps` became a probed build fact,
  and the metal proof is `stress_fleet.py::learner_sleep`, written and unrun.*

### Interfaces

**A frame** (Q10) is the repo's word for one message on a `Transport`
(`remote.py:213`): a verb name plus a JSON-safe payload dict, and the JSON-safe
reply dict that comes back — `call(verb, payload)` on the admitted, async path
and `ask(verb, payload)` on the admission-free, sync one. `RemotePool.add_bundle`
sends the frame `("add_bundle", {"base", "tp", "bundle": {"bundle_id",
"policy_version", "payloads": {name: base64}, "adapter_types"}})` and gets `{}`
back; `LocalTransport` round-trips every frame through `json.dumps`/`loads` so
a same-process wire is held to the same contract as a real one. Frames are the
CONTROL channel. A future NCCL data channel between residents runs beside them.

The Learner protocol keeps five verbs; `install` takes a `Parameterization`.
Two new wire pairs mirror the one that exists: `RemoteLearner`/`LearnerService`
for the learner, `RemotePool`/`EngineService` for the engine — `RemotePool`
unchanged, `EngineService` being the engine-verb half of today's `HostService`
with admission left behind in `HostService`, which now admits and then forwards
through the proxy. Admission is unchanged: it happens once, at the serving
host's arbiter, and traffic is counted at that seam as before. The resident
door adds verbs on no protocol: `hello` (birth facts — kind, base, `tp` or
`fsdp`, `sleeps`, devices seen, cap applied, pid), `sleep`/`wake`, `stop`.
`MetalService.describe()` gains `builds` (the metal's recipe) and, per host,
`residents`; `Host.status()` gains `residents`; the `host-up` journal event
carries `residents: [{label, pid}]`. `observe/` sees one more row per host.
Nothing on the fleet journal's `list`/`metal`/`provision` rows changes.

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


# data/stores/ — how a child opens the store the metal opened
@dataclass(frozen=True)
class StoreAddress:
    backend: str                         # "local" | "modal_volume" — one file per backend, named here
    root: str                            # the mount or directory
    locator: str                         # the store's own locator, as it describes itself

def open_store(address: StoreAddress) -> Store: ...


# runner/residents.py — what is universal, and what a venue declares
@dataclass(frozen=True)
class EngineBuild:                       # the capacity knobs a venue declares; nothing placement-shaped
    max_model_len: int
    max_bundles: int
    max_rank: int
    max_members: int
    serves: tuple[str, ...]              # the adapter types this build pays for
    enable_sleep_mode: bool
    enforce_eager: bool = True

@dataclass(frozen=True)
class LearnerBuild:
    dtype: str                           # "bfloat16"
    grad_clip: float
    checkpoint_activations: bool

@dataclass(frozen=True)
class ResidentBirth:
    label: str                           # "<host>:<regime>"
    partition: Partition
    regime: Regime
    build: EngineBuild | LearnerBuild
    store: StoreAddress

def build_engine(regime: Regime, partition: Partition, build: EngineBuild, store: Store) -> Engine:
    """UNIVERSAL: VllmEngine(regime.base, tp=regime.shape,
    gpu_memory_utilization=partition.memory, cas_get=store.cas_get, **build)."""

def build_learner(regime: Regime, partition: Partition, build: LearnerBuild) -> Learner:
    """UNIVERSAL: shape 1 -> TorchLearner(device="cuda:0", ...);
    shape n -> lead_fsdp_learner(width=n, ...). cuda:0 IS the partition's first
    device once CUDA_VISIBLE_DEVICES is pinned."""

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
    loads, build_learner, serve LearnerService frames until stop, end the
    chorus, exit."""

def engine_main(birth: ResidentBirth, conn) -> None:
    """The child: open_store(birth.store), build_engine, serve EngineService
    frames concurrently on its own loop until stop, shut the engine down, exit."""


# runner/learners/torch_learner.py — the learner's half of alternation (Q8a)
class TorchLearner:
    async def sleep(self) -> None:
        """Hand the device back: base, every tenant's params and moments to
        pinned host RAM; empty the allocator's cache. Idempotent."""
    async def wake(self) -> None:
        """The inverse. Placement is settled before any verb runs, so no
        forward meets a half-woken learner — the arbiter switches at zero
        in-flight work, as for engines."""


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
                                              self.builds.for_regime(r), self.store_address))
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

> **Samarth:** engines and learners.

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

> **Samarth:** yes, the loss should stay bound at install, because otherwise
> momentum, etc. doesnt make much sense.

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

> **Samarth:** emitted bytes. eventually, the other mode we will support is
> NCCL directly between two residents (dont do anything too drastic that would
> prevent this from happening in the future)

*Folded:* the Decision's "kept open for the NCCL mode" paragraph — one payload
codec, stable resident identities the metal can broker, no payload routed
through the metal process by construction. The transfer verb is a later ADR.

**Q4. How does the child learn to build its resident?** As first drafted:
builders by qualified name plus a JSON bag of venue facts, on the premise that
building was the venue's. Samarth's answer rejects the premise, and the code
agrees with him — see the Context's third fact: every deploy lambda is the
universal part (class, base, width, device, fraction — all functions of
(regime, partition)) interleaved with a typed record nobody had written (the
capacity knobs) and a store to read from.
Recommendation, refolded: **building is rlstack's. `runner/residents.py` owns
`build_engine`/`build_learner`; `EngineBuild`/`LearnerBuild` are typed records;
the venue declares them and a `StoreAddress` at bring-up and nothing else. The
child opens its own store through `open_store`. No name-by-string, no bag, no
callable.** Given (regime, partition, build) the resident is determined, on
every venue; the build record is the one input that is not derivable from the
partition, which is why it is a record and not an argument list.
If the other branch: builders stay in `deploy/`, reached by name, and the same
regime on two venues can build two different residents for reasons no record
states.

> **Samarth:** isn't building not a property of deploy or a particular venue,
> but rather a universal property? like, given a specific partition, it's
> deterministic how to build on there, right?

*Folded:* yes, with one refinement — deterministic given the partition, the
regime, AND a build record. The record holds what the partition cannot tell
you: how long a context, how many bundles, which adapter types the build pays
for, whether it sleeps; a learner's dtype and clip. Those are not derivable and
are not in the spec either, which raises Q4a.

**Q4a. Where does the build record live?** It is not derivable from the
partition (Q4), and it is not in the spec today — `PoolMember` is (name, base,
tp, n, fraction) and `LearnerMember` is (fsdp, fraction) (`specs.py:161`).
Recommendation: **on the metal, at bring-up: `MetalService(metal, builds=
Builds(engine=EngineBuild(...), learner=LearnerBuild(...)), store=
StoreAddress(...))`, reported in `describe()` and stamped into every hello.
One recipe per metal; every resident it carves is built from it. The desk
stays blind to it — capability matching is (base, shape), as today.** This
keeps the knobs out of identity (a capacity knob in the canonical tree would
make a `max_model_len` bump a different `run_id`, the thing I5 forbids), and it
is honest about what they already are: deploy constants, now typed and visible.
It also leaves a known asymmetry in place rather than hiding it: two metals
with different `max_model_len` both "serve (base, tp)", and a join cannot tell
them apart. That is true today and is a placement question for its own entry.
If the other branch (per carve, from the desk): the carve request carries a
build record, so the DEMAND side must know engine capacity — either the spec
grows the knobs (into identity) or the campaign layer invents them (a third
place, unrecorded).

> **Samarth:** just ensure that its stored in a way where runs can be
> auto-restartable. as long as that's satisfied, you can do what you want

*Folded:* the metal re-declares its recipe from the deploy's constants at
every bring-up, so the recarve a restart goes through (Q7) rebuilds the same
residents unattended; the recipe is journaled on the `metal` row and every
`host-up` event so the record of how is durable.

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

> **Samarth:** sure that works

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

> **Samarth:** sure let's reason about that later.

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

> **Samarth:** yes, a restart should go through the entire loop of resubmitting
> and recarving so the metalservice (and by proxy, the desk and observer) knows
> whats up

**Q8. Alternation crosses the door, and gets wired.** The arbiter's `evict`/
`wake` hooks are async callables; the engine's `sleep`/`wake` exist and are
wired by nothing on this branch. As first drafted this question said a learner
"cannot hand memory back today" and wired hooks for engines only.
Recommendation, refolded: **`sleep`/`wake` are door verbs for BOTH kinds, and
`Host._attach_regimes` wires `evict`/`wake` for every resident whose hello
reports `sleeps: true`, engine or learner.** The switch still happens only at
zero in-flight work, so no frame meets a half-woken resident.
If the other branch: alternation stays admission-only, and a multi-member
group under ADR 0001 sizes for both residents resident at once.

> **Samarth:** does this imply that a learner can't sleep? why?

*Folded:* no — the draft mistook an absence for an inability. A learner holds
the frozen base, each tenant's adapter params and Adam moments, accumulated
grads between `forward_backward` and `optim_step`, and the allocator's cached
blocks; activations exist only inside a step. Every one of those can move to
host RAM and back, which is exactly what vLLM's level-1 sleep does for the
engine (weights to host RAM, KV cache discarded). Nothing in torch prevents it;
the repo simply never wrote the verb, because alternation on this branch is
admission-only and nobody needed it. It matters more than it looks: on an
alternating host the learner's frozen base is a SECOND COPY of the weights the
engine holds, resident through every sampling phase; ADR 0001's Q7 lets
alternating members each want most of the partition, which is honest only if
both hand it back; and vLLM checks free device memory at build, so an engine
sized for the whole partition refuses to boot beside a learner that never
sleeps. Cost: the base crosses PCIe each way — a 14B bf16 base is ~28 GB, one
to two seconds per switch on an A100 — the same order as the engine's own
sleep. What is NOT simple is `fsdp > 1`: `fully_shard` parameters are DTensors
and their offload is its own proof (non-promises). Hence Q8a.

**Q8a. Does learner sleep land in this pass?**
Recommendation: **yes, for `fsdp = 1`: `TorchLearner.sleep`/`wake` as
sketched, `hello` reports `sleeps: true` for an unsharded learner and `false`
for a sharded one, and the hook wiring is the same code path as the engine's.
The cap makes the absence visible — a capped learner and a capped engine on
one partition, each sized for it, cannot both be awake — so shipping the door
verb without the learner's half would leave every alternating host oversized
by one base copy, which ADR 0001's sizing then has to lie about.** A sharded
learner keeps today's behavior (no hooks, co-resident) and says so in its
hello.
If the other branch: the door verbs land symmetric and the learner's `sleep`
is a stub that reports `sleeps: false`; alternating hosts keep two base copies
resident; ADR 0001's Q7 needs a sum rule after all for the learner-bearing
case.

> **Samarth:** sure, that's fine

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

> **Samarth:** yes

**Q10. Frames are JSON, and byte identity is the test.** A frame is defined
under Interfaces: one verb's JSON-safe payload dict and its JSON-safe reply,
the unit every `Transport` in the repo carries; `LocalTransport` enforces the
contract with a `json.dumps`/`loads` round-trip (`remote.py:319`). The chorus
ships `TokenBatch` by pickle today, outside that contract, because it never
had a Transport.
Recommendation: **JSON frames for the resident wire, honoring the contract:
tuples become lists and back, floats survive Python's repr round-trip
losslessly, bytes are base64. `TokenBatch.token_extras`/`doc_turn_extras` are
recording-channel values that already seal as canonical jsonl
(`waves/<update>.jsonl.gz`, `stores/base.py:12`), so they are JSON-safe by
construction — and a value that is not will now fail loudly at the wire
instead of quietly in a pickle. The pin is one test in `test_remote`'s shape:
a run with process residents is byte-identical to the in-process run on
fakes; `test_resume.py` stays green untouched.** Frames are the control
channel; the NCCL data channel Q3 names runs beside them and is not a frame.
If the other branch: pickle over the pipe — faster, outside the contract, and
a leak of non-JSON state into a batch goes unnoticed until it reaches a real
wire.

> **Samarth:** what are frames?

> **Samarth (with the definition above):** lets just do JSON.

**Q11. Real child processes in the fakes suite.** The suite is stdlib-only and
~5 s. `FakeEngine` and `FakeLearner` are stdlib and picklable by module
reference. A spawn-context child costs 100–300 ms.
Recommendation: **real processes in a handful of tests — the byte-identity
test (Q10), the death-decarves-the-host test (Q7), the device-count refusal,
the ladder — and the in-process `LocalTransport` everywhere else.** The suite
grows by a second or two and the boundary is exercised for real on every run.
If the other branch: an in-process "resident" double, and the first real
process bug is found on a GPU.

> **Samarth:** sure

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

> **Samarth:** sure

## Outcome

Landed 2026-09-01 on gsm-campaign, recorded as CONTEXT #74.

**What landed.** `runner/residents.py` (Builds / EngineBuild / LearnerBuild /
FakeEngineBuild / FakeLearnerBuild, ResidentBirth, the universal
`build_engine` / `build_learner`, pin / cap / devices_seen, the Door and its
frames over a request-id multiplexed pipe, `Resident.spawn` / `in_process` /
`watch` / `stop`, the ladder lifted from ranks.py); `Parameterization` /
`EntryInstall` / `OptimSettings` in interfaces.py with `install(tenant,
parameterization)`; `loop.parameterization_of` + `init_seed`; `RemoteLearner`,
`LearnerService`, `EngineService` and the four codecs in remote.py; `Host(
residents=)` with door-wired hooks and resident rows; `MetalService(builds=,
spawn=)` with `decarve`-as-ladder, `resident_exited`, `shutdown`, `describe`
carrying the recipe; `Desk.register_metal(builds=)` journaled and replayed;
`StoreAddress` / `Store.address()` / `open_store`; `TorchLearner.sleep` /
`wake` / `shutdown`, `FsdpTorchLearner.sleeps = width == 1`, follower caps;
fakes with `sleeps` / `naps` / `down`; the observer's resident rows; four
venues converted.

**What the answers changed.** Q4 turned name-by-string builders in `deploy/`
into rlstack-owned universal builders plus typed recipes — the deploy lambdas
were the universal part interleaved with an unwritten record. Q8 made the
door symmetric and added the learner's sleep (fsdp=1). Q4a put the recipe on
the metal, re-declared at bring-up and journaled on `metal` and `host-up`
rows. Q3 pinned one payload codec and kept resident identities stable for
the NCCL mode. Q6 stayed synchronous, as recommended.

**Tests.** 868 green on fakes (from 855; +13 in `tests/test_residents.py`),
110 torch-gated skips, ~7 s. `test_resume.py` untouched and green. One new
architecture edge: `runner/learners/` imports nothing from `rlstack.spec`.

**One thing found by the tests.** The watcher waits on the child's sentinel
rather than joining it: two threads reaping one child left the loser reading
ECHILD as "alive" through every rung of the ladder (deaf, wedged, even lost
after SIGKILL), observed on fakes and fixed before landing.

**Unproven on metal.** The pin on a 2+-device metal (asserted by device count
in the hello, never yet exercised), the torch cap, learner sleep's move set
(adapter-type tensors outside `parameters()` are a stated gap), the follower
cap, `open_store` on a Modal mount, teardown inside Modal's grace, and every
round-trip cost — the TokenBatch frame per microbatch, the emitted bytes per
update, `tokenize` per trajectory (the local tokenizer beside RemotePool is
now due). Learner sleep at `fsdp > 1` is not built; a chorus reports
`sleeps: false`.

**Since landed.** ADR 0001 (CONTEXT #75) made the desk's journaled recipe row
the CANON that rides every carve request (`MetalService.adopt_recipe`); the
deploy's constants re-declared at bring-up are the metal's FIRST declaration,
and a re-registration carrying a different recipe updates the desk's row.

**Deviations, stated.** `observe/locate.store_for` was not moved: the child's
address is derived from the Store object (`Store.address()`), not parsed from
a locator, so the observer's resolver had nothing to gain. The pipe transport
lives in `residents.py` beside the door it serves, not in `remote.py`; the
protocol-level pieces (services, codecs, `RemoteLearner`) are in `remote.py`
as decided.

## Amendment (2026-09-04) — the learner is fully remote, like the engine

> **Samarth:** ok yes. let's actually make the learner fully remote just like
> the engine, and remove the constraint that the trainer must live on the same
> process.

**What changes.** This ADR made the learner a process behind a door and kept
the Host's proxy to it LOCAL: `RemoteLearner` was built only by the metal that
spawned the resident (`desk.py:1529`), `HostService` forwarded no learner verb
to a foreign caller, `check_fit` demanded an owned learner, and the campaign
layer anchored every delivery on the learner's host because "the learner is
never remote" (#43, #69). The amendment removes that rule: the six learner
verbs (`uninstall` joined the protocol at implementation) cross the host door exactly as `sample_tokens` and `score_tokens` do,
admitted at the SERVING host's arbiter per frame; a `LearnerMember` resolves to
a `RemoteLearner` from a route address exactly as a `PoolMember` resolves to a
`RemotePool`; a remote learner attaches to the runner's arbiter as a
zero-footprint free resident, its alternation living at the far host; the
learner demand yields a route, so the anchor of a delivery becomes a CHOICE
(the learner's host by default when the spec declares one, `main` otherwise —
ADR 0006 Q2, refolded). A run's Trainer therefore lives wherever the run is
anchored, and any experiment may join a standing learner from any host.

**What this ADR's reasoning already said.** The "cannot cross a wire" line was
named a misattribution above: the autograd arc is the Learner's, the seal is
the Trainer's, and only the verbs' bytes cross. The rank chorus stays internal
to the learner resident (rank 0 answers the door and announces to its
followers), so a frame from another host is the same frame. The store is the
volume; the Trainer writes it from wherever it sits. Nothing in this ADR's
Decision is contradicted except the sentence "The Trainer does not move", left
standing with its note, because the reasoning is the artifact.

**Where the shape lives.** ADR 0006 Part A carries the decision's seams,
promises and the implementer's stated sub-decisions (admission host-side, an
explicit `uninstall` verb at the end of a tenancy, sync frames kept per Q6,
custody journaled at the learner's host). The CONTEXT entry that records the
landing is ADR 0006's: **#79**, 2026-09-04.
