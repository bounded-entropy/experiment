"""The PostProcessor contract: everything computed ABOUT sealed trajectories.

The sealed record holds what the policy did; this stage computes what to make
of it — rewards, judge scores, advantages, any per-group statistic — as one
declared pipeline (AlgoSpec.post / EvalSpec.post) that runs after the seal and
before the loss. Each processor is a CLASS in its own file under this folder:

    @postprocessor("my_score")
    class MyScore(PostProcessor):
        produces = ("score",)
        consumes = ()                       # columns from EARLIER in the pipeline
        async def process(self, group, data, llm) -> Mapping[str, Sequence[float]]:
            return {"score": [ ... one float per trajectory ... ]}

The contract every subclass inherits: `process` sees ONE group (the scope of a
partial loss contribution), the columns earlier processors produced for that
group, and a SampleClient (an LLM judge is just a processor that samples —
`llm.pool(name)` reaches any engine pool). It returns one vector per declared
`produces` name, len(group) floats each, in group order.

Pipelines are ordered: `consumes` must be satisfied by earlier processors
(checked at Phase 0), columns have one owner, and the resulting postdata is
stored per update beside the rollouts — columnar, aligned to wave order — then
broadcast per token into the TokenBatch, where a loss `requires` the columns it
uses (grpo requires "advantage"). Deterministic given the seed tree: resume
recomputes identical postdata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rlstack.client import SampleClient
from rlstack.data.trajectory import Group
from rlstack.registry import POST, source_hash


class PostProcessor(ABC):
    """Subclass, declare produces/consumes, implement `process`, register with
    @postprocessor. Subclasses must construct with no arguments — the decorator
    instantiates one shared instance."""

    produces: tuple[str, ...] = ()
    consumes: tuple[str, ...] = ()

    @abstractmethod
    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      llm: SampleClient) -> Mapping[str, Sequence[float]]:
        """One vector per `produces` name; each len(group), in group order."""


@dataclass(frozen=True)
class PostDef:
    """A registered postprocessor: class, shared instance, its declaration."""

    name: str
    cls: type[PostProcessor]
    instance: PostProcessor
    produces: tuple[str, ...]
    consumes: tuple[str, ...]
    source_hash: str


def postprocessor(name: str):
    def register(cls: type[PostProcessor]) -> type[PostProcessor]:
        instance = cls()
        POST.add(PostDef(name, cls, instance, tuple(instance.produces),
                         tuple(instance.consumes), source_hash(cls)))
        return cls
    return register
