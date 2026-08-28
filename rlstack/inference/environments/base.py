"""The Environment contract: drive one episode against its pools, return the
Rollout.

An environment is registered as a CLASS in its own file under this folder:

    @environment("my_env")
    class MyEnv(Environment):
        async def run(self, client: PoolClient, task: Task) -> Rollout:
            ...

The rule every subclass inherits: `run` returns a Rollout — mutable,
inference-world, unsealed. The RUNNER seals; environments never do. Sampling is
free here, through the policy pool or any other declared pool via
`client.pool(name)`, and thousands of episodes run as concurrent coroutines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rlstack.client import PoolClient
from rlstack.data.trajectory import Task
from rlstack.inference.rollout import Rollout
from rlstack.registry import ENVS, source_hash


class Environment(ABC):
    """One episode. Subclass, implement `run`, register with @environment.

    Subclasses must construct with no arguments — the decorator instantiates
    one shared instance.
    """

    @abstractmethod
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        """Drive sample calls until the episode is finished; return the Rollout
        unsealed — the seal is the runner's, at run_episode."""


@dataclass(frozen=True)
class EnvironmentDef:
    """A registered environment: the class plus one shared instance."""

    name: str
    cls: type[Environment]
    instance: Environment
    source_hash: str


def environment(name: str):
    def register(cls: type[Environment]) -> type[Environment]:
        ENVS.add(EnvironmentDef(name, cls, cls(), source_hash(cls)))
        return cls
    return register
