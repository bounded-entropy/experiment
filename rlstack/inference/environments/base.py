"""The Environment contract: drive the engine through one episode, return the
Rollout.

An environment is registered as a CLASS in its own file under this folder:

    @environment("my_env")
    class MyEnv(Environment):
        async def run(self, llm: SampleClient, task: Task) -> Rollout:
            ...

The invariant every subclass inherits: `run` returns a Rollout — the mutable,
inference-world episode. The RUNNER seals it (after rewards); environments
never do. Environments may sample freely (that is the stage rule routing them
here), and thousands run as concurrent coroutines against the resident engine.

An environment is not limited to the one policy pool: the SampleClient
(rlstack/client.py) reaches every named engine pool via `llm.pool(name)` —
hinting pipelines, helper models, anything the episode needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rlstack.client import SampleClient
from rlstack.data.trajectory import Task
from rlstack.inference.rollout import Rollout
from rlstack.registry import ENVS, source_hash


class Environment(ABC):
    """One episode. Subclass, implement `run`, register with @environment.

    Subclasses must construct with no arguments — the decorator instantiates
    one shared instance.
    """

    @abstractmethod
    async def run(self, llm: SampleClient, task: Task) -> Rollout:
        """Drive sample calls until the episode is finished; return the Rollout
        (unsealed — sealing is the runner's job, after rewards)."""


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
