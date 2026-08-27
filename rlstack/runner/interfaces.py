"""The seams the runner drives: Engine (inference metal) and Learner (training
metal), plus the event and result types that cross them.

The Engine speaks TOKENS — its native unit — as a stream of TokenEvents ending
in a FinishEvent; a request's bundle is pinned at submission, so registering a
new bundle never disturbs generation in flight. The EnginePoolClient
(runner/traffic.py) assembles the stream into a Turn, the membrane's record unit.

Everything real slides in behind these protocols: FakeEngine/FakeLearner for
dry runs and tests (runner/fakes.py), vLLM and the torch learner in Phase B2/3,
the resident daemon in Phase C. The runner never knows which it has.
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
    """Inference metal. One resident engine (pool) behind one interface.

    MULTI-TENANCY INVARIANT (multi-LoRA generalized to multi-adapter): bundle
    registration is ADDITIVE — any number of experiments' bundles coexist,
    every request pins its own bundle at submission, and requests batch
    together regardless of whose bundle they carry. Every serving Mechanism
    must therefore be per-request selectable. One resident engine serves many
    experiments in parallel; nothing about a registration disturbs anyone
    else's traffic.

    REACHABILITY INVARIANT: whether a mechanism reaches a site is a property
    of this BUILD — kernel coverage, fusion maps, installed plugins — so the
    engine self-reports it (`reachability`) and the runner checks every served
    adapter against the answer at Phase 0. An engine must refuse nothing it
    reported and report nothing it cannot serve; the parity certificate is the
    numerical proof, per build fingerprint.
    """

    base: str | None
    """The model this metal serves — checked against each pool's declared
    base at submit (pool-base-mismatch). None is the fake-metal wildcard."""

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
    """Training metal. Owns ONE frozen base, many tenants' deltas + optimizers.

    MULTI-TENANCY INVARIANT (the trainer-side mirror of the Engine's):
    installation is ADDITIVE — any number of experiments' adapter sets coexist
    on one loaded base — and every verb PINS a tenant (the run_id). One
    resident learner serves many experiments; nothing about one tenant's
    install, step, or load disturbs another's state. The v0 realization is
    swap-install (the active tenant's adapters are wired into the module tree,
    switching costs module rebinds, never weight copies); batched multi-tenant
    forwards are a kernel upgrade behind the same surface.
    """

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
