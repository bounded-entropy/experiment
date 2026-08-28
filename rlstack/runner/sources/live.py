"""Live: the Generator writes waves/<u>; this feed just reads them.

The behavior policy is whatever bundle the generator sampled under — recorded
per turn, opportunistic within the lag buffer. "Not yet" is the normal answer
here, and the trainer's await condition absorbs it.
"""

from __future__ import annotations

from rlstack.data.stores.base import RunHandle
from rlstack.runner.sources.base import WaveFeed


class LiveFeed(WaveFeed):
    def __init__(self, run: RunHandle) -> None:
        self._run = run

    def obtain(self, update: int) -> list[dict] | None:
        try:
            return self._run.read_wave(update)
        except FileNotFoundError:
            return None
