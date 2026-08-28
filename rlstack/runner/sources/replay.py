"""Replay: another run's sealed waves, copied in update for update.

Off-policy consumption through the exact contract live uses. The behavior
policy is whatever the parent recorded — bundle ids, logprobs and draws are all
in the sealed rows — which is what a loss's importance correction runs against.
Group structure is the parent's, verbatim.
"""

from __future__ import annotations

from rlstack.data.stores.base import RunHandle, Store
from rlstack.runner.sources.base import WaveFeed


class ReplayFeed(WaveFeed):
    def __init__(self, store: Store, source: str, run: RunHandle) -> None:
        # "store://<run_id>[/...]" — the first path segment names the run
        self._parent_id = source[len("store://"):].split("/")[0]
        self._parent = store.open_run(self._parent_id)   # raises if absent
        self._run = run

    def obtain(self, update: int) -> list[dict] | None:
        try:
            return self._run.read_wave(update)       # already copied
        except FileNotFoundError:
            pass
        try:
            rows = self._parent.read_wave(update)
        except FileNotFoundError:
            raise ValueError(
                f"parent run {self._parent_id!r} has no sealed wave for "
                f"update {update} — its run stopped earlier; lower n_updates"
            ) from None
        self._run.write_wave(update, rows)
        return rows
