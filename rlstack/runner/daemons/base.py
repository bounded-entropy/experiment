"""The daemon shape every role follows.

A role owns one GPU responsibility. Its loop is always the same four beats:
await its CONDITION (a predicate over the store — never a call into another
role), hold its LEASE for the metal, do the work, write the store and notify.
Subclass a role and override its condition method to change when it wants the
mutex; the lease itself stays dumb.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rlstack.data.stores.base import RunHandle
from rlstack.runner.lease import Lease
from rlstack.runner.signals import RunSignals


class Daemon(ABC):
    def __init__(self, signals: RunSignals, lease: Lease, run: RunHandle) -> None:
        self.signals = signals
        self.lease = lease
        self.run = run

    def committed(self) -> int:
        """The trainer's progress as the store tells it: the ledger tail."""
        tail = self.run.ledger_tail()
        return int(tail["update"]) if tail else 0

    @abstractmethod
    async def run_forever(self) -> None:
        """The daemon loop; returns when this role's work for the run is done."""
