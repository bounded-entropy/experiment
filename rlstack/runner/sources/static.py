"""Static dataset: a fixed, content-addressed file of sealed trajectories.

The SFT / offline-distillation shape: cas://<sha> resolves to jsonl of
trajectory rows (as written by trajectory_to_row). Each trajectory is its own
singleton group — no cross-trajectory context — and update u takes the u-th
slice of rollouts_per_wave rows, cycling deterministically. Rows are written
into the run's rollouts/ on first request: every run is self-contained."""

from __future__ import annotations

import json

from rlstack.data.stores.base import RunHandle, Store
from rlstack.runner.sources.base import WaveFeed


class StaticFeed(WaveFeed):
    def __init__(self, store: Store, source: str, run: RunHandle,
                 rollouts_per_wave: int) -> None:
        raw = store.cas_get(source).decode("utf-8")
        self._rows = [json.loads(line) for line in raw.splitlines() if line]
        if not self._rows:
            raise ValueError(f"static dataset {source!r} is empty")
        if rollouts_per_wave > len(self._rows):
            raise ValueError(
                f"rollouts_per_wave={rollouts_per_wave} exceeds the dataset "
                f"({len(self._rows)} trajectories)")
        self._per_wave = rollouts_per_wave
        self._run = run

    def obtain(self, update: int) -> list[dict] | None:
        try:
            return self._run.read_rollouts(update)       # already sliced in
        except FileNotFoundError:
            pass
        start = ((update - 1) * self._per_wave) % len(self._rows)
        rows = []
        for i in range(self._per_wave):
            row = dict(self._rows[(start + i) % len(self._rows)])
            row["group"] = f"row-{(start + i) % len(self._rows):06d}"
            rows.append(row)
        self._run.write_rollouts(update, rows)
        return rows
