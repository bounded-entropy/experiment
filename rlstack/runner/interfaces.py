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

from rlstack.data.flatten import TokenBatch
from rlstack.data.trajectory import Message
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.compile import Bundle
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.meters import TrafficMeter
from rlstack.spec.specs import ExperimentSpec, SamplingSpec


# ---------------------------------------------------------------------------
# the token stream
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenEvent:
    """One generated token. `extras` is the per-token recording channel: a
    kind's engine plugin deposits sampling-time facts here (e.g. the adapter
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
    learner makes these real, the loop's handling never changes."""

    loss: float
    mean_ratio: float
    logprob_gap: float
    grad_norm: float
    tokens: int


@dataclass(frozen=True)
class Emitted:
    """What the learner hands the store after an optim_step: delta payloads
    plus optimizer moments, keyed by bank name."""

    adapters: Mapping[str, bytes]
    optim: Mapping[str, bytes]


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
    ) -> AsyncIterator[TokenEvent | FinishEvent]:
        """Stream tokens for one request, pinned to `bundle_id`, ending with a
        FinishEvent. Registering new bundles never affects requests in flight."""
        ...

    async def score_tokens(
        self,
        messages: Sequence[Message],
        token_ids: Sequence[int],
        bundle_id: str,
    ) -> tuple[float, ...]:
        """Logprob of each given token continuing `messages`, pinned to
        `bundle_id` — ONE prefill pass over context + tokens, no decode loop,
        deterministic (no seed). The scoring half of the pool surface: judges
        sample, teachers score."""
        ...

    def add_bundle(self, bundle: Bundle) -> None:
        """Register a compiled bundle (the add_lora analog)."""
        ...

    def reachability(self, sites: Sequence[SiteMeta]) -> Mapping[str, Mechanism]:
        """This build's inventory: for each site, the mechanism that reaches it
        (Mechanism.NONE when nothing does). Self-reported, per build."""
        ...

    def tokenize(self, text: str) -> tuple[int, ...]:
        """The engine's tokenizer — flatten uses it for injected spans only."""
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

    def install(
        self,
        tenant: str,
        spec: ExperimentSpec,
        resolved_sites: Mapping[str, tuple[SiteMeta, ...]],
    ) -> None:
        """Phase 1: build `tenant`'s trainable parameterization for every bank
        entry. Additive across tenants; rebuilding a tenant resets its state."""
        ...

    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats:
        """One microbatch under `tenant`'s adapters: trainer-kernel forward,
        loss, backward. Grads accumulate on that tenant's params."""
        ...

    def optim_step(self, tenant: str) -> None:
        """Apply `tenant`'s accumulated gradients; its deltas advance one version."""
        ...

    def emit(self, tenant: str) -> Emitted:
        """Serialize `tenant`'s deltas + optimizer moments for store and bundle."""
        ...

    def load(self, tenant: str, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None:
        """Restore `tenant`'s deltas (and moments, unless None → fresh)."""
        ...
