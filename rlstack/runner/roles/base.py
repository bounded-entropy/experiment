"""The shape every runner follows.

A runner owns one GPU responsibility, and its loop is always the same four
beats: await its CONDITION (a predicate over the store — never a call into
another runner), ADMIT the residents its work occupies, do the work, write the
store and notify. Override a runner's condition method to change when it wants
the metal; the arbiter stays dumb about which runner is asking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rlstack.data.stores.base import RunHandle
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.signals import RunSignals


class StopRequest:
    """A DELIBERATE stop, asked of one run's runners (ADR 0014, Part C).

    Set by the host's `stop(drain=True)`; read by the Trainer at every update
    boundary, where it checkpoints the update it last committed and returns,
    so a stop loses nothing. In-process state only: the host that adopted
    the run is the one that stops it, and nothing about a stop is a byte of
    the run directory.
    """

    def __init__(self) -> None:
        self.requested = False
        self.reason = ""
        # set by the Trainer once its drain checkpoint is on the store: the
        # signal the loop ends every other runner on
        self.drained = False

    def request(self, reason: str = "") -> None:
        self.requested = True
        self.reason = reason


class Runner(ABC):
    def __init__(self, signals: RunSignals, arbiter: Arbiter,
                 run: RunHandle) -> None:
        self.signals = signals
        self.arbiter = arbiter
        self.run = run

    def committed(self) -> int:
        """The trainer's progress as the store tells it: the ledger tail."""
        tail = self.run.ledger_tail()
        return int(tail["update"]) if tail else 0

    @abstractmethod
    async def run_forever(self) -> None:
        """The runner loop; returns when this runner's work for the run is
        done."""
