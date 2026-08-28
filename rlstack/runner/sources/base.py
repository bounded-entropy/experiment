"""The feed contract: the trainer ALWAYS reads waves from its own run.

A feed's one job is to make update u's rows exist in THIS run's waves/ and hand
them back — or say "not yet". Live data is written by the Generator and simply
read here; storage-backed data is copied in on first request. Either way every
run is self-contained and the trainer's await condition is uniformly
"feed.obtain(u) returned rows".
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class WaveFeed(ABC):
    @abstractmethod
    def obtain(self, update: int) -> list[dict] | None:
        """Update u's trajectory rows, guaranteed present in the run's
        waves/ — or None when they don't exist yet (live only)."""
