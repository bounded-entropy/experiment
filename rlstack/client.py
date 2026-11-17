"""The pool interface: neutral ground both worlds may type against.

A consumer-defined Protocol, like registry.py: environments (inference world)
sample during rollouts and postprocessors (training world) sample or score
after the seal, neither world importing the other. Nothing is limited to the
policy pool — `pool(name)` reaches any pool the experiment's Topology
declares, so judges, teachers and hinting pipelines address the whole
inference side. Every client for one episode shares one seed sequence, which
is what keeps multi-pool traffic deterministic. EnginePoolClient
(runner/traffic.py) is the implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rlstack.data.trajectory import Message, Turn
from rlstack.policy.adapters.base import Directive
from rlstack.spec.specs import AdapterSpec, ExperimentSpec, SamplingSpec


class PoolClient(Protocol):
    """One client against one pool, plus a door to the others.

    `sample` returns a complete Turn — the engine's own token ids, behavior
    logprobs, recorded extras, finish reason — pinned to the pool's current
    bundle, with a seed derived per call. `pool(name)` returns a sibling
    client for another declared pool, sharing this episode's seed sequence.

    `directives` on either verb are the caller's per-request instructions to
    the bank's adapter types (ADR 0004, Q2) — typed records each adapter type
    declares, at most one per adapter type per request. What they made the
    rollout do is recorded at the seal, so passing one never asks replay to
    remember anything.

    `max_tokens` on `sample` is a per-call CEILING under the run's sampling
    budget (the call gets the smaller of the two): an environment bounding
    one kind of call — a tool's answer — without changing the run's sampling.

    `sampling` replaces the run defaults for this call; a per-call max_tokens
    ceiling still applies. Experiment functions can explicitly mix greedy
    evaluation calls and stochastic exploration in one episode.

    `affinity` on either verb is the call's ROUTING KEY among the pool's
    replicas (ADR 0022) — OpenAI's `prompt_cache_key`, in this repo's words:
    calls sharing a key go to the replica that already holds their prefix.
    None is the episode's id, so an environment that never names one keeps
    one episode on one replica; an environment branching a conversation may
    name a key per branch. Routing never reaches the sealed bytes.
    """

    async def sample(self, messages: Sequence[Message],
                     stop: tuple[str, ...] = (), *,
                     directives: Sequence[Directive] = (),
                     max_tokens: int | None = None,
                     sampling: SamplingSpec | None = None,
                     affinity: str | None = None) -> Turn: ...

    async def score(self, messages: Sequence[Message],
                    token_ids: Sequence[int], *,
                    directives: Sequence[Directive] = (),
                    affinity: str | None = None) -> tuple[float, ...]:
        """Logprobs of ALREADY-CHOSEN tokens continuing `messages`, under this
        pool's serving stack: one prefill pass, no decode, no randomness.
        The teacher/hinted channel (I9) — a postprocessor scores sealed tokens
        under privileged conditioning and emits a token_level column. Judges
        sample; teachers score."""
        ...

    def pool(self, name: str) -> "PoolClient": ...


@dataclass(frozen=True)
class GradientRow:
    """One independently conditioned document, with exact scored-token positions.

    `token_ids` already includes its complete student context. The binary mask
    is token-aligned; prompt tokens and each document's first token are zero.
    The measurement client preserves these IDs and this mask without tokenization.
    """

    row_id: str
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]


@dataclass(frozen=True)
class GradientDocumentStats:
    """Selected-token loss numerators and denominator for one measured document.

    NLL is always the hard-token diagnostic. KL measurements additionally
    report full-distribution KL(T||S), teacher entropy and cross-entropy.

    A nonfinite numerator is None, accompanied by an invalid comparison; it
    never becomes a serialized NaN or a fabricated zero loss.
    """

    nll_sum: float | None
    scored_tokens: int
    kl_sum: float | None = None
    teacher_entropy_sum: float | None = None
    student_cross_entropy_sum: float | None = None
    objective: str = "sft"


@dataclass(frozen=True)
class GradientComparison:
    """Read-only loss-gradient alignment on identical student coordinates.

    References use token-mean NLL; candidates use NLL or detached-teacher KL.
    Each group averages its document means with equal document weight.
    Invalid comparisons have no cosine. Only scalars and document sufficient
    statistics cross the wire; no gradient or optimizer state is exposed.
    """

    cosine: float | None
    dot: float | None
    reference_norm: float | None
    candidate_norm: float | None
    valid: bool
    reason: str | None
    reference_documents: tuple[GradientDocumentStats, ...]
    candidate_documents: tuple[GradientDocumentStats, ...]
    reference_seconds: float = 0.0
    candidate_seconds: float = 0.0
    parameter_count: int = 0
    parameter_schema_sha256: str = ""
    min_norm: float = 1e-12
    reference_cache_hit: bool = False
    candidate_cache_hit: bool = False


@dataclass(frozen=True)
class TrainingResult:
    """A completed or deliberately stopped ordinary child run."""

    run_id: str
    run_ref: str
    completed: int
    versions: Mapping[str, int]
    stopped: bool = False


@dataclass(frozen=True)
class TrainingSnapshot:
    """Checkpointed adapter and optimizer state, keyed by bank entry."""

    adapters: Mapping[str, bytes]
    optim: Mapping[str, bytes]
    versions: Mapping[str, int]


class Training(Protocol):
    """Compose normal sealed-data runs and read-only learner measurements.

    There is no optimizer loop in this client. A child spec goes through the
    same setup, Trainer, ledger and checkpoint recovery as a top-level run.
    """

    base: str
    seed: int
    microbatch_tokens: int

    async def run(self, spec: ExperimentSpec, *, checkpoint_every: int,
                  save_state: bool = True) -> TrainingResult: ...

    async def snapshot(self, run_ref: str) -> TrainingSnapshot: ...

    async def evaluate(self, adapter: AdapterSpec, payload: bytes | None,
                       rows: Sequence[Mapping[str, Any]]) -> tuple[float, ...]: ...

    async def gradient_cosines(
            self, start_payload: bytes | None,
            reference_rows: Sequence[GradientRow],
            candidate_rows: Sequence[Sequence[GradientRow]], *,
            adapter: AdapterSpec, teacher_prefix_ids: tuple[int, ...] | None = None
    ) -> tuple[GradientComparison, ...]: ...

    async def tokenize(self, text: str) -> tuple[int, ...]: ...

    async def rows(self, uri: str) -> list[dict[str, Any]]: ...

    async def put(self, payload: bytes) -> str: ...

    async def publish(self, name: str, payload: bytes, meta: Mapping[str, Any]) -> None: ...

    async def probes(self, run_ref: str, update: int) -> dict: ...


class Names(Protocol):
    """Named adapters, read-only."""

    async def read_named(self, name: str) -> bytes | None: ...


class TrainingPoolClient(PoolClient, Protocol):
    """A training processor's pool client and explicit composition doors."""

    training: Training
    names: Names
