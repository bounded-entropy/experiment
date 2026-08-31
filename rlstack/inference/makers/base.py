"""The TaskMaker contract: mint a NEW task from an already-sealed trajectory.

A maker is the content half of a Derive leaf (data/plan.py): the plan pins
WHERE a derived episode sits and WHAT it derives from; the maker pins what
the derived task SAYS. It is a PURE, synchronous function of the sealed
bytes — no client, no sampling, no store, no randomness — because everything
a reflect prompt needs is already sealed (the original prompt, the attempt,
the metadata), and anything that needs a completion is the ENVIRONMENT's job
inside the derived episode. Purity is what makes a resumed run re-mint the
same task to the byte.

    @task_maker("my_maker")
    class MyMaker(TaskMaker):
        def make(self, source: Trajectory) -> Task:
            ...

Registered like an environment, declared like one (GenSpec.makers), hashed
like one (code_hashes): editing a maker's body is a different experiment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rlstack.data.trajectory import Task, Trajectory
from rlstack.registry import MAKERS, source_hash


class TaskMaker(ABC):
    """One derivation. Subclass, implement `make`, register with @task_maker.
    Subclasses must construct with no arguments — the decorator instantiates
    one shared instance."""

    @abstractmethod
    def make(self, source: Trajectory) -> Task:
        """The derived task, pure and total: every sealed trajectory this
        maker can be pointed at yields a task or raises loudly."""


@dataclass(frozen=True)
class MakerDef:
    """A registered task maker: the class plus one shared instance."""

    name: str
    cls: type[TaskMaker]
    instance: TaskMaker
    source_hash: str


def task_maker(name: str):
    def register(cls: type[TaskMaker]) -> type[TaskMaker]:
        MAKERS.add(MakerDef(name, cls, cls(), source_hash(cls)))
        return cls
    return register
