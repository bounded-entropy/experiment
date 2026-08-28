"""The pool interface: neutral ground both worlds may type against.

A consumer-defined Protocol, like registry.py: environments (inference world)
sample during rollouts and postprocessors (training world) sample or score
after the seal, neither world importing the other. Nothing is limited to the
policy pool — `pool(name)` reaches any pool the experiment's GpuConfig
declares, so judges, teachers and hinting pipelines address the whole
inference side. Every client for one episode shares one seed sequence, which
is what keeps multi-pool traffic deterministic. EnginePoolClient
(runner/traffic.py) is the implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rlstack.data.trajectory import Message, Turn


class PoolClient(Protocol):
    """One client against one pool, plus a door to the others.

    `sample` returns a complete Turn — the engine's own token ids, behavior
    logprobs, recorded extras, finish reason — pinned to the pool's current
    bundle, with a seed derived per call. `pool(name)` returns a sibling
    client for another declared pool, sharing this episode's seed sequence.
    """

    async def sample(self, messages: Sequence[Message],
                     stop: tuple[str, ...] = ()) -> Turn: ...

    async def score(self, messages: Sequence[Message],
                    token_ids: Sequence[int]) -> tuple[float, ...]:
        """Logprobs of ALREADY-CHOSEN tokens continuing `messages`, under this
        pool's serving stack: one prefill pass, no decode, no randomness.
        The teacher/hinted channel (I9) — a postprocessor scores sealed tokens
        under privileged conditioning and emits a token_level column. Judges
        sample; teachers score."""
        ...

    def pool(self, name: str) -> "PoolClient": ...
