"""The Reward contract: score one rollout as a vector of named scalars.

A reward is registered as a CLASS in its own file under this folder:

    @reward("my_reward")
    class MyReward(Reward):
        components = ("score", "length_penalty")
        async def score(self, rollout, llm) -> tuple[float, ...]:
            return (..., ...)

The invariant every subclass inherits: `score` takes the finished (unsealed)
Rollout and returns one float per name in `components`, in order — a possibly
degenerate vector (most rewards return one scalar). The runner zips names with
values and checks the length, so a body that disagrees with its declaration
fails loudly. Rewards run per-episode BEFORE the seal and may sample (judges
are rewards, never losses); advantages declare which components they consume,
and Phase 0 joins the two (I4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rlstack.inference.environments.base import SampleClient
from rlstack.inference.rollout import Rollout
from rlstack.registry import REWARDS, source_hash


class Reward(ABC):
    """Subclass, declare `components`, implement `score`, register with @reward.

    Subclasses must construct with no arguments — the decorator instantiates
    one shared instance.
    """

    components: tuple[str, ...] = ()

    @abstractmethod
    async def score(self, rollout: Rollout, llm: SampleClient) -> tuple[float, ...]:
        """One float per declared component, in declaration order."""


@dataclass(frozen=True)
class RewardDef:
    """A registered reward: the class, one shared instance, its declaration."""

    name: str
    cls: type[Reward]
    instance: Reward
    components: tuple[str, ...]
    source_hash: str


def reward(name: str):
    def register(cls: type[Reward]) -> type[Reward]:
        instance = cls()
        REWARDS.add(RewardDef(name, cls, instance,
                              tuple(instance.components), source_hash(cls)))
        return cls
    return register
