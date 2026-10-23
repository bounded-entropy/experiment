"""The seams the runner drives: Engine (inference metal) and Learner (training
metal), plus the events and results that cross them.

An Engine speaks TOKENS — a stream of TokenEvents ending in a FinishEvent —
and every request pins its bundle at submission, so registering a new bundle
never disturbs generation in flight. A Learner mirrors that on the other side:
installation is additive and every verb pins a tenant, so nothing about one
tenant's traffic, install or load disturbs another's (I8). `tp` and `fsdp` are
BUILD facts the submit gate attests. Everything real slides in behind these
protocols — fakes, vLLM, the torch learners, a remote pool — and the runner
never knows which it has.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rlstack.client import GradientComparison
from rlstack.data.flatten import TokenBatch
from rlstack.data.trajectory import Message
from rlstack.policy.adapters.base import Directive, Mechanism, Reachability
from rlstack.policy.compile import Bundle
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.meters import TrafficMeter
from rlstack.spec.specs import SamplingSpec


# ---------------------------------------------------------------------------
# the token stream
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenEvent:
    """One generated token. `extras` is the per-token recording channel: a
    adapter type's engine plugin deposits sampling-time facts here (e.g. the
    adapter
    index drawn for this token), and they seal into Turn.token_extras."""

    token_id: int
    logprob: float
    text_delta: str
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinishEvent:
    """Terminal event of a stream: why generation stopped, plus per-request
    sampling facts (e.g. a drawn latent) that seal into Turn.turn_extras."""

    finish: str                 # "stop" | "eos" | "length"
    stop_hit: str | None = None
    turn_extras: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# training-side results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainStats:
    """One forward_backward's rails. logprob_gap is |trainer − behavior| mean —
    the silent-off-policy alarm; the fake learner reports zeros, the torch
    learner makes these real, the loop's handling never changes.

    `provided` is the bank's PROVIDED tensors, summarized to one float each
    under their declared names — a declaration-keyed column family exactly like
    postdata, not a bag: every key is some adapter type's `provides` entry, so
    the flow graph already named it and dictionary.json already describes it.
    The Trainer means them across an update's microbatches into the ledger."""

    loss: float
    mean_ratio: float
    logprob_gap: float
    grad_norm: float
    tokens: int
    provided: Mapping[str, float] = field(default_factory=dict)
    components: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Emitted:
    """What the learner hands the store after an optim_step: delta payloads
    plus optimizer moments, keyed by bank name."""

    adapters: Mapping[str, bytes]
    optim: Mapping[str, bytes]


@dataclass(frozen=True)
class EntryInstall:
    """One bank entry as the learner needs it: the adapter type BY REGISTRY
    KEY (resolved on the learner's side, against the same image), its init
    with the per-entry seed ALREADY DERIVED (the master seed never crosses),
    whether it trains, and the sites the runner resolved off the schema."""

    name: str
    adapter_type: str
    init: Mapping[str, Any]
    trainable: bool
    sites: tuple[SiteMeta, ...]
    route: str | None = None


@dataclass(frozen=True)
class OptimSettings:
    """The optimizer as the learner needs it: OptimSpec's fields, projected.
    Overrides are keyed by entry ("pi") or entry.group ("pi.mapper")."""

    name: str
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    overrides: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class Parameterization:
    """What `install` builds, projected off the spec BY THE RUNNER: the base,
    the loss by registry key, the bank entries in install order, and the
    optimizer settings. Nothing else of the experiment reaches a learner —
    a learner owns tensors, not experiments — and the loss is bound HERE
    rather than per forward, because the moments it accumulates belong to
    one loss (ADR 0002, Q2). The one place a spec becomes one of these is
    `loop.parameterization_of`."""

    base: str
    loss: str
    entries: tuple[EntryInstall, ...]
    optim: OptimSettings


# ---------------------------------------------------------------------------
# the protocols
# ---------------------------------------------------------------------------

class Engine(Protocol):
    """Inference metal: one resident engine serving every tenant behind it.

    MULTI-TENANCY INVARIANT (I8): bundle registration is ADDITIVE — any number
    of tenants' bundles coexist, each request pins its own at submission, and
    requests batch together regardless of whose bundle they carry, so every
    serving Mechanism must be per-request selectable and no registration ever
    disturbs anyone else's traffic.

    REACHABILITY INVARIANT: whether a mechanism reaches a site is a property of
    this BUILD — kernel coverage, fusion maps, installed plugins — so the
    engine self-reports it (`reachability`) and Phase 0 holds every served
    adapter against the answer. An engine must refuse nothing it reported and
    report nothing it cannot serve.
    """

    base: str | None
    """The model this metal serves — checked against each pool's declared
    base at submit (pool-base-mismatch). None is the fake-metal wildcard."""

    tp: int
    """Tensor-parallel width of this BUILD — how many devices one forward
    spans. Sharding is a build fact like reachability (#43): a pool declaring
    tp=4 binds only onto an engine built tp=4 (pool-shape-mismatch), and
    switching shards means handing different metal, never editing a spec."""

    meter: TrafficMeter
    """Where this engine counts the traffic it serves: prompt tokens at
    submission, decoded tokens in its own token loop, time to first token.
    One meter per HOST — a host wires its own into every engine it owns
    (Host.wire_meter) and drains it once per stats tick. An engine nobody
    adopted counts into the private meter it was born with, which nothing
    drains and nothing reads."""

    def sample_tokens(
        self,
        messages: Sequence[Message],
        sampling: SamplingSpec,
        stop: tuple[str, ...],
        bundle_id: str,
        seed: int,
        directives: Sequence[Directive] = (),
    ) -> AsyncIterator[TokenEvent | FinishEvent]:
        """Stream tokens for one request, pinned to `bundle_id`, ending with a
        FinishEvent. Registering new bundles never affects requests in flight.
        `directives` are the caller's per-request instructions to the bundle's
        adapter types (ADR 0004); what they did is in the FinishEvent's
        turn_extras."""
        ...

    async def score_tokens(
        self,
        messages: Sequence[Message],
        token_ids: Sequence[int],
        bundle_id: str,
        directives: Sequence[Directive] = (),
    ) -> tuple[float, ...]:
        """Logprob of each given token continuing `messages`, pinned to
        `bundle_id` — ONE prefill pass over context + tokens, no decode loop,
        deterministic (no seed). The scoring half of the pool surface: judges
        sample, teachers score."""
        ...

    def add_bundle(self, bundle: Bundle) -> None:
        """Register a compiled bundle (the add_lora analog)."""
        ...

    def prepare_transfer(self, manifest: Mapping, transfer) -> dict:
        """Prepare an immutable tensor publication at this resident's epoch."""
        ...

    def finish_transfer(self, source: str, token: str) -> dict:
        """Acknowledge installation of the exact received bundle."""
        ...

    def abort_transfer(self, source: str, token: str) -> None:
        """Discard partial publication; preserve existing immutable bundles."""
        ...

    def add_library(self, name: str, payload: bytes) -> None:
        """Hold one NAMED PAYLOAD for routes to read (ADR 0019): the bytes
        `lora_torch.emit` writes for one LoRA set, read from the store by a
        ROLE and handed over here — an engine never opens a store.

        Idempotent (a name is write-once, so the same name with different
        bytes is refused), safe under traffic, and LRU-bounded by the build's
        `max_library`. Eviction is not a loss and not announced: a caller that
        routes under `lib:<name>` hands the bytes again before it samples, and
        a route naming a library adapter this engine does not hold is refused
        at the request."""
        ...

    def holds_library(self, name: str, digest: str) -> bool:
        """Is `name` held right now, with the bytes whose sha256 is `digest`?
        What a proxy asks before sending a payload the engine likely holds."""
        ...

    def knows_bundle(self, bundle_id: str) -> bool:
        """Is this bundle resident on this engine right now?

        Residency is not durable: a bounded pool evicts, a restarted container
        starts empty. Neither is a failure, because a committed version is
        re-derivable from the store — so a caller ASKS, and rebuilds on a miss
        (runner/restore.py). The ledger says what a version is; this says only
        whether we happen to be holding it.
        """
        ...

    def reachability(self, sites: Sequence[SiteMeta]) -> Reachability:
        """This build's inventory: for each site, the mechanisms that reach it
        (Mechanism.NONE when nothing does). Self-reported, per build."""
        ...

    def tokenize(self, text: str) -> tuple[int, ...]:
        """The engine's tokenizer — flatten uses it for injected spans only."""
        ...

    def first_contact(self) -> dict:
        """WHAT THIS BUILD ACTUALLY TOOK, measured once it has served
        something (ADR 0008, F5) — empty until then.

        An engine reports the GB its weights occupy and how many tokens its
        KV cache holds; a learner reports its peak allocated GB per rank
        after its first forward. The host journals the answer BESIDE the GB
        its partition was declared at, because every OOM on this fleet was a
        declared number nobody had measured: 48 then 64 GB per card met a
        46 GiB resident and one 8 GiB fp32 block, three times in one evening.

        Empty is not a refusal — it means "nothing has happened here yet",
        which is what the host keeps asking until it stops being true."""
        ...


class Learner(Protocol):
    """Training metal: ONE frozen base, many tenants' deltas + optimizers.

    MULTI-TENANCY INVARIANT (I8, the trainer-side mirror of the Engine's):
    installation is ADDITIVE — every tenant's adapter set stays wired on the
    one loaded base — and every verb PINS a tenant (the run_id), so nothing
    about one tenant's install, step, or load disturbs another's state. The
    realization is additive install + ROW ROUTING: each row of a batched
    forward carries the slot whose deltas apply to it. A verb pins one tenant,
    so today every forward is the one-slot case; cross-tenant coalescing is a
    scheduling upgrade behind this same surface, not a kernel change.
    """

    fsdp: int
    """FSDP shard width of this BUILD — across how many ranks the base's
    parameters shard. A build fact (#43), attested at submit against the
    LearnerMember's declared fsdp (learner-shape-mismatch); 1 = unsharded."""

    dp: int
    """Data-parallel groups of this BUILD (ADR 0022): `accumulate` splits an
    update's microbatches across `dp` groups of `fsdp` ranks each, and sums
    their gradients in microbatch order. A build fact attested like fsdp;
    1 = every microbatch on one group."""

    def install(self, tenant: str, parameterization: Parameterization) -> None:
        """Phase 1: build `tenant`'s trainable parameterization for every bank
        entry. Additive across tenants; rebuilding a tenant resets its state.
        The record is the whole of what a learner may know about the
        experiment (ADR 0002)."""
        ...

    def tokenize(self, tenant: str, text: str) -> tuple[int, ...]:
        """Tokenize injected text with the installed base, without special tokens.

        Generated token IDs remain recorded data; only injected spans need this
        CPU operation. Training never needs an inference resident to flatten.
        """
        ...

    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats:
        """One microbatch under `tenant`'s adapters: trainer-kernel forward,
        loss, backward. Grads accumulate on that tenant's params."""
        ...

    def accumulate(self, tenant: str,
                   batches: Sequence[TokenBatch]) -> tuple[TrainStats, ...]:
        """EVERY MICROBATCH OF ONE UPDATE, forward and backward (ADR 0022):
        gradients accumulate on `tenant`'s params as if `forward_backward`
        had run on each batch IN ORDER — the same bytes at any `dp` — and
        one TrainStats comes back per batch, in order. The verb the Trainer
        issues once per update, so a dp learner can split the batches across
        its groups; at dp=1 it IS that loop."""
        ...

    def forward(self, tenant: str, batch: TokenBatch) -> tuple[float, ...]:
        """The no-grad twin of `forward_backward` (ADR 0019): one float per
        document, its mean NLL over its loss_mask tokens, under `tenant`'s
        adapters and each row's own route. Nothing accumulates."""
        ...

    def gradient_cosines(
        self, tenant: str, entry: str,
        reference_batches: Sequence[TokenBatch],
        candidate_batches: Sequence[Sequence[TokenBatch]],
    ) -> tuple[GradientComparison, ...]:
        """Measure ordinary supervised gradients without changing learner state.

        Each group uses equal document weight after its own selected-token
        mean NLL. Reference gradients are computed once; candidate gradients
        use the same entry parameters and never touch persistent `.grad`
        buffers, optimizer moments or parameter versions. Initially unsharded
        (fsdp=dp=1); distributed builds refuse this operation explicitly.
        """
        ...

    def optim_step(self, tenant: str,
                   lr_scales: Mapping[str, float] | None = None) -> None:
        """Apply `tenant`'s accumulated gradients; its deltas advance one version.

        `lr_scales` multiplies each NAMED param group's BASE learning rate for
        THIS step only — keyed by group name (`"memory:03"`), by `entry.group`
        where two entries share a group name, or by entry for all of its
        groups (`lr_scale_of` is the rule). A schedule is therefore the
        caller's pure function of the step; nothing compounds. None is the
        step this verb always took."""
        ...

    def load_set(self, tenant: str, entry: str, route: str,
                 payload: bytes | None) -> None:
        """Start ONE set of bank entry `entry` over (ADR 0019): from a NAMED
        PAYLOAD (one LoRA set as `lora_torch.emit` writes it), or from the
        set's own deterministic init when None. Either way that set's
        optimizer moments are reset. A route `lib:<name>` installs a FROZEN
        library set outside every optimizer group and outside `emit`."""
        ...

    def drop_set(self, tenant: str, entry: str, route: str) -> None:
        """Forget one LIBRARY set (`lib:<name>`) of `entry`. Library sets
        only; one nobody holds is already dropped."""
        ...

    def emit_set(self, tenant: str, entry: str, route: str) -> bytes:
        """ONE set of `entry` as a named payload — what `load_set`, the store's
        `write_named` and the engine's `add_library` all read."""
        ...

    def emit(self, tenant: str) -> Emitted:
        """Serialize `tenant`'s deltas + optimizer moments for store and bundle."""
        ...

    def emit_initial(self, tenant: str, versions: Mapping[str, int]) -> Emitted:
        """Bind logical versions after restore/warm-start, then export checkpoint zero."""
        ...

    def emit_adapters(self, tenant: str, versions: Mapping[str, int] | None = None) -> Mapping[str, bytes]:
        """Adapter bytes only; never serialize optimizer state for publication."""
        ...

    def emit_optim(self, tenant: str) -> Mapping[str, bytes]:
        """Optimizer bytes, requested only at a durable checkpoint."""
        ...

    def export_state(self, tenant: str, versions: Mapping[str, int], names: Sequence[str]):
        """Immutable tensor states, retained in the GPU-owning resident."""
        ...

    def policy_pin(self, tenant, versions, names, adapter_types) -> Bundle:
        """Content identity for replay-only updates, without exporting weights."""
        ...

    def prepare_transfer(self, tenant, versions, names, adapter_types) -> dict:
        """Capture state on the resident, returning only its control manifest."""
        ...

    def send_transfer(self, token: str, endpoint: Mapping, timeout_s: float) -> dict:
        """Move a captured snapshot over the peer data channel."""
        ...

    def release_transfer(self, token: str) -> None:
        """Drop a publication snapshot after acknowledgement or failure."""
        ...

    def load(self, tenant: str, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None:
        """Restore `tenant`'s deltas (and moments, unless None → fresh)."""
        ...

    def first_contact(self) -> dict:
        """WHAT THIS BUILD ACTUALLY TOOK after its first forward (ADR 0008,
        F5): `peak_gb` per rank, read off the allocator, empty before the
        first forward. See Engine.first_contact — same verb, same reason, and
        the learner is where the reason came from."""
        ...

    def uninstall(self, tenant: str) -> None:
        """Install's inverse: unwire `tenant` and forget its state — the end
        of a tenancy, said out loud (ADR 0006 Part A).

        A learner outlives the runs on it, and since a run may be anchored on
        another host it can no longer be assumed that the process holding the
        tenant dies with it: the host that ran the run uninstalls at the end,
        done or failed, so a standing learner does not accumulate the dead.
        IDEMPOTENT — a tenant that was never installed is already uninstalled,
        because "gone" is the goal state, not an error. Resume is unaffected:
        re-attaching installs again and restores from the store's blobs, as
        every attach already does."""
        ...


def lr_scale_of(lr_scales: Mapping[str, float], entry: str, group: str) -> float:
    """The scale ONE named param group takes from an `optim_step`'s
    `lr_scales`: `entry.group` wins, then the bare group name, then the entry
    (every group of it), else 1.0 — the overrides' grammar, specific over
    general. The one rule every Learner applies."""
    if group and f"{entry}.{group}" in lr_scales:
        return float(lr_scales[f"{entry}.{group}"])
    if group and group in lr_scales:
        return float(lr_scales[group])
    return float(lr_scales.get(entry, 1.0))


def check_lr_scales(lr_scales: Mapping[str, float],
                    groups: Mapping[str, Sequence[str]]) -> None:
    """Every key of `lr_scales` addresses a group this tenant steps (`groups`:
    trainable entry -> its group names). A key that reaches nothing is a
    schedule silently not applied, so it is refused."""
    known = set(groups)
    for entry, names in groups.items():
        known.update(name for name in names if name)
        known.update(f"{entry}.{name}" for name in names if name)
    unknown = sorted(set(lr_scales) - known)
    if unknown:
        raise ValueError(f"lr_scales names no param group: {unknown} "
                         f"(this tenant steps {sorted(known)})")
