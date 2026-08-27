"""The inference interface: what anything that talks to pools sees.

Consumer-defined Protocols, deliberately neutral ground (like registry.py):
environments (inference world) sample during rollouts, and postprocessors
(training world) sample or SCORE after the seal (LLM judges, hinted/teacher
rescoring) — both type against these without either world importing the
other. The concrete implementation is EnginePoolClient (runner/traffic.py).

The pool principle: nothing is limited to the one policy pool. `pool(name)`
returns a client for any named engine pool in the experiment's GpuConfig —
hinting pipelines, judges, teachers all reach the whole inference side. Every
client for one episode shares one seed sequence, so multi-pool traffic stays
deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rlstack.data.trajectory import Message, Turn


class PoolClient(Protocol):
    """One awaitable against one engine pool, plus a door to the others.

    `sample` returns a complete Turn — the engine's token ids, behavior
    logprobs, recorded extras, finish reason — pinned to the pool's current
    bundle, with a seed derived per call. `pool(name)` returns a sibling
    client for another named pool, sharing this episode's seed sequence.
    """

    async def sample(self, messages: Sequence[Message],
                     stop: tuple[str, ...] = ()) -> Turn: ...

    async def score(self, messages: Sequence[Message],
                    token_ids: Sequence[int]) -> tuple[float, ...]:
        """Logprobs of ALREADY-CHOSEN tokens continuing `messages`, under this
        pool's serving stack — one prefill pass, no decode, no randomness.
        The teacher/hinted channel (I9): post processors score sealed tokens
        under privileged conditioning and emit token_level columns. Scoring
        traffic is prefill-shaped — the natural tenant of a future
        prefill-disaggregated pool."""
        ...

    def pool(self, name: str) -> "PoolClient": ...
