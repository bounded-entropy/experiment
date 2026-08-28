"""The PostProcessor contract: everything computed ABOUT sealed trajectories.

The sealed record holds what the policy did; a postprocessor computes what to
make of it — rewards, judge scores, advantages, teacher logprobs — as one
ORDERED pipeline (AlgoSpec.post / EvalSpec.post) running after the seal and
before the loss, one class per file under this folder. `process` sees ONE group
(the scope of a partial loss contribution), the columns earlier processors
produced for it, and a PoolClient; anything that needs a GPU is a
postprocessor's job (I9). The rules: `consumes` must be satisfied by an EARLIER
processor (checked at Phase 0), every column has exactly one owner, each
`produces` name comes back as len(group) floats in group order, and the
pipeline is deterministic given the seed tree, so resume recomputes identical
postdata. The result is stored per update beside the waves — columnar, aligned
to wave order — then broadcast per token into the TokenBatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.registry import POST, source_hash
from rlstack.spec.specs import SamplingSpec


class PostProcessor(ABC):
    """Subclass, declare produces/consumes/token_level/pools, implement
    `process`, register with @postprocessor. Subclasses must construct with no
    arguments — the decorator instantiates one shared instance.

    `pools` declares EVERY pool this processor sends traffic to, whether via
    `llm.pool(name)` or the default main-pinned client. Phase 0 holds it
    against the spec's declared pools ("main" exempt — the runner requires it
    unconditionally) and against sleep-sharing: the trainer admits exactly
    these residents around the pipeline, so an undeclared pool is sampled
    UNADMITTED, which under sleep colocation means a sleeping engine.
    `sampling` overrides the run's generation sampling for this processor's
    calls (a judge wants its own temperature and budget, not the policy's);
    None inherits. Both live in the class source, so they hash into run
    identity through code_hashes like the rest of the declaration."""

    produces: tuple[str, ...] = ()
    consumes: tuple[str, ...] = ()
    token_level: tuple[str, ...] = ()   # produced columns that are per-token:
                                        # one float per generated token per
                                        # trajectory, in sealed order
    pools: tuple[str, ...] = ()
    sampling: "SamplingSpec | None" = None

    @abstractmethod
    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      llm: PoolClient) -> Mapping[str, Sequence[float]]:
        """One vector per `produces` name; each len(group), in group order."""


@dataclass(frozen=True)
class PostDef:
    """A registered postprocessor: class, shared instance, its declaration."""

    name: str
    cls: type[PostProcessor]
    instance: PostProcessor
    produces: tuple[str, ...]
    consumes: tuple[str, ...]
    token_level: tuple[str, ...]
    pools: tuple[str, ...]
    source_hash: str


def postprocessor(name: str):
    def register(cls: type[PostProcessor]) -> type[PostProcessor]:
        instance = cls()
        POST.add(PostDef(name, cls, instance, tuple(instance.produces),
                         tuple(instance.consumes), tuple(instance.token_level),
                         tuple(instance.pools), source_hash(cls)))
        return cls
    return register
