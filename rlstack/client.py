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
    """

    async def sample(self, messages: Sequence[Message],
                     stop: tuple[str, ...] = (), *,
                     directives: Sequence[Directive] = ()) -> Turn: ...

    async def score(self, messages: Sequence[Message],
                    token_ids: Sequence[int], *,
                    directives: Sequence[Directive] = ()) -> tuple[float, ...]:
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


class Fits(Protocol):
    """`client.fit`: small LoRA fits on the run's OWN learner, under a
    throwaway tenant, for a processor that declares `fits = True`."""

    async def forks(self, start_payload: bytes | None,
                    fork_rows: Sequence[Sequence[Mapping[str, Any]]],
                    probe_rows: Sequence[Mapping[str, Any]],
                    stage: Stage) -> list[tuple[float, ...]]:
        """Fork `start_payload` (None = a fresh set) once per entry of
        `fork_rows`, train fork k on `fork_rows[k]` (trajectory rows) under
        `stage`'s epochs, batch, lr and decay — its `rows` is ignored — and
        answer each fork's per-document mean NLL over `probe_rows` AFTER
        training, in fork order."""
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
