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


# ---------------------------------------------------------------------------
# fits from a postprocessor (ADR 0019)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    """One leg of a fit: `rows` ("cas://<sha>", a jsonl of trajectory rows)
    for `epochs` passes in batches of `batch`, at peak learning rate `lr`
    under `decay` ("linear": lr falls to 0 across the stage — HF Trainer's
    default — or "constant"). Declared here because it is what a processor
    hands `client.fit.forks`; the loop that runs one is runner/fit.py."""

    rows: str
    epochs: int
    batch: int
    lr: float
    decay: str = "linear"


@dataclass(frozen=True)
class GradientRow:
    """One independently conditioned document, with exact scored-token positions.

    `token_ids` already includes its complete student context. The binary mask
    is token-aligned; prompt tokens and each document's first token are zero.
    The fitting client preserves these IDs and this mask without tokenization.
    """

    row_id: str
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]


@dataclass(frozen=True)
class GradientDocumentStats:
    """Selected-token NLL numerator and denominator for one measured document.

    A nonfinite numerator is None, accompanied by an invalid comparison; it
    never becomes a serialized NaN or a fabricated zero loss.
    """

    nll_sum: float | None
    scored_tokens: int


@dataclass(frozen=True)
class GradientComparison:
    """Read-only loss-gradient alignment on identical student coordinates.

    References and each candidate are means of per-document token-mean NLLs.
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


class Fits(Protocol):
    """`client.fit`: small LoRA fits on the run's OWN learner, under a
    throwaway tenant, for a processor that declares `fits = True`."""

    async def forks(self, start_payload: bytes | None,
                    fork_rows: Sequence[Sequence[Mapping[str, Any]]],
                    probe_rows: Sequence[Mapping[str, Any]],
                    stage: Stage, *, every_probe: bool = False) -> list:
        """Fork `start_payload` (None = a fresh set) once per entry of
        `fork_rows`, train fork k on `fork_rows[k]` (trajectory rows) under
        `stage`'s epochs, batch, lr and decay — its `rows` is ignored — and
        answer each fork's per-document mean NLL over `probe_rows` AFTER
        training, in fork order. With `every_probe`, each fork's every probe
        point instead (before its first step, then after its stage)."""
        ...

    async def gradient_cosines(
        self, start_payload: bytes | None,
        reference_rows: Sequence[GradientRow],
        candidate_rows: Sequence[Sequence[GradientRow]],
    ) -> tuple[GradientComparison, ...]:
        """Compare supervised gradients at one fixed temporary student state.

        References are independent documents, averaged after each selected-token
        mean NLL. Each candidate is its own document group. Nothing is fitted,
        no optimizer state conditions the reward, and input tokens/masks remain
        exact. Results retain candidate order.
        """
        ...

    async def tokenize(self, text: str) -> tuple[int, ...]:
        """`text` as the learner's base tokenizes it, without special tokens:
        a processor has no tokenizer of its own, and text it builds at run
        time (a dream) becomes a trajectory row's tokens only through this."""
        ...

    async def rows(self, uri: str) -> list[dict[str, Any]]:
        """The trajectory rows of a `cas://<sha>` jsonl in the run's store —
        background and probe rows a task's meta names by uri."""
        ...


class Names(Protocol):
    """`client.names`: named adapters, read-only."""

    async def read_named(self, name: str) -> bytes | None:
        """The named payload under this run's subdir, or None while it does
        not exist."""
        ...


class FittingPoolClient(PoolClient, Protocol):
    """What a `fits = True` processor's `client` is: the pools, plus `fit`
    and `names`."""

    fit: Fits
    names: Names
