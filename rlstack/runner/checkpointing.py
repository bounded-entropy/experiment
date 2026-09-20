"""Checkpointing: a PLACEMENT-TIME declaration of what a crash may cost and
how the policy travels (ADR 0014).

A run's COMMIT is the ledger line — one per update, cheap, what the Generator
pins from and what the observer reads. A run's CHECKPOINT is the durable
point: the blobs for the current version plus one `checkpoints.jsonl` line.
Attach rewinds to the last checkpoint, so a crash costs at most `every`
updates of compute, and that trade is stated here, deliberately, with no
default anywhere — every submission door requires one of these.

Nothing in this record enters identity (I5): the same spec under any cadence
is the same run, and a resume may change its cadence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Delivery = Literal["wire", "store"]

DELIVERIES: tuple[str, ...] = ("wire", "store")


@dataclass(frozen=True)
class Checkpointing:
    """How often the run is durable, and how its policy reaches its pools.

    `every` — blobs and a checkpoint line at every update u with
    `u % every == 0`, ALWAYS at the extent's last update, and ALWAYS on a
    drained stop. `every=1` is the old behavior to the byte, plus one file.

    `delivery` — `wire`: the Trainer pushes each committed bundle to every
    pool that serves the policy, over that pool's transport, before the
    ledger line; a consumer that misses an uncheckpointed version parks the
    run. `store`: the Trainer touches no engine and every consumer faults in
    from the store — which only works when every version has a blob, so
    `store` is bound to `every == 1` here, by name. Ignored by a run with no
    serving pool: there is nothing to deliver to.
    """

    every: int
    delivery: Delivery = "wire"

    def __post_init__(self) -> None:
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError(f"Checkpointing.every must be an int >= 1, got {self.every!r}")
        if self.delivery not in DELIVERIES:
            raise ValueError(f"Checkpointing.delivery must be one of {DELIVERIES}, got {self.delivery!r}")
        if self.delivery == "store" and self.every != 1:
            raise ValueError(
                "Checkpointing(delivery='store') requires every=1: a consumer faulting "
                "in from the store needs a blob at every version, and only every=1 "
                f"writes one (got every={self.every})")

    def due(self, update: int, extent: int) -> bool:
        """Is update `update` (1-based) a checkpoint, in a run `extent` updates long?"""
        return update % self.every == 0 or update >= extent

    def row(self) -> dict[str, Any]:
        """The wire form: what rides the frame beside `subdir` and `resume`."""
        return {"every": self.every, "delivery": self.delivery}

    @classmethod
    def from_row(cls, row: Mapping[str, Any] | None) -> "Checkpointing":
        """The frame's declaration, or a loud refusal — no default (ADR 0014)."""
        if not isinstance(row, Mapping) or "every" not in row:
            raise ValueError(
                "a submission must declare its Checkpointing({'every': k, "
                "'delivery': 'wire'|'store'}): the cadence is a deliberate "
                "decision and there is no default")
        return cls(every=int(row["every"]), delivery=str(row.get("delivery", "wire")))


EVERY_UPDATE = Checkpointing(every=1)
"""The old discipline, named: a blob at every version. Tests and the
file-system-only path use it; nothing else defaults to it."""
