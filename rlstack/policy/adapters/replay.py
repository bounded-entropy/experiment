"""The batched replay seam: which adapter state each ROW of a trainer forward
carries.

The engine batches requests that pin different bundles and indexes its LoRA
slot banks per token (punica); the trainer batches documents into ONE padded
forward and indexes the installed deltas PER ROW — same flavor, same reason
(I8: multi-tenancy on both sides of the bridge). This module is the trainer's
half of that: one RowPlan per loaded base, routed by the learner for the
duration of one forward, read by every replay lowering wired into the tree.

The plan rides ON THE MODEL because the model is the one handle
Adapter.install_replay receives (adapters/base.py) — the routing has to reach
every lowering without widening that five-member surface. torch is imported at
module scope, so this file loads from the kinds' compute halves and the
learner, never from the package root (STYLE rule 7).
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

ROW_PLAN = "_rlstack_row_plan"


@dataclass(frozen=True)
class ReplayRows:
    """One padded forward's routing table.

    A SLOT is one tenant's installed deltas, addressed the way a site can ask
    for them: {site path -> the params object holding that site's delta}.
    `slots` is the slot order and `index` is [rows] long — row r applies slot
    index[r]. A verb pins one tenant, so today every microbatch is the
    degenerate one-slot case; a coalesced microbatch is the same record with
    more slots and a mixed index.
    """

    slots: tuple[Mapping[str, Any], ...]
    index: torch.Tensor                 # [rows], long

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("a replay forward routes to at least one slot")
        if int(self.index.max()) >= len(self.slots):
            raise ValueError(
                f"row plan reaches slot {int(self.index.max())} of "
                f"{len(self.slots)} — rows carry installed slots only")

    def uniform(self) -> Mapping[str, Any] | None:
        """The one slot every row carries, or None when the rows disagree.

        THE parity anchor: with one slot a site applies the plain (x A^T) B^T
        it applied under swap-install, over the whole batch in one GEMM pair —
        so a single-tenant microbatch is bit-identical to the pre-batching
        trainer, and the per-row path is entered only when it is needed.
        """
        return self.slots[0] if len(self.slots) == 1 else None


class RowPlan:
    """The model's routing table: EMPTY between forwards, set for exactly one.

    A replay lowering that runs unrouted has no way to know whose delta
    applies, so it raises instead of guessing — an unrouted replay forward is
    a wiring bug, never a fallback to "whoever went last".
    """

    def __init__(self) -> None:
        self._rows: ReplayRows | None = None

    @contextmanager
    def route(self, rows: ReplayRows):
        """Pin `rows` for the duration of one forward."""
        if self._rows is not None:
            raise RuntimeError("this base is already routed — replay forwards "
                               "on one model do not nest")
        self._rows = rows
        try:
            yield rows
        finally:
            self._rows = None

    @property
    def rows(self) -> ReplayRows:
        if self._rows is None:
            raise RuntimeError(
                "this replay forward carries no row plan: the learner routes "
                "one per forward (policy.adapters.replay.row_plan)")
        return self._rows


def row_plan(model: Any) -> RowPlan:
    """The model's one row plan, created by whoever asks first.

    Both sides ask here — the learner to route a forward, a kind's
    install_replay to hand the plan to the sites it wires — so the table is
    per loaded base and exactly one function knows where it lives.
    """
    plan = getattr(model, ROW_PLAN, None)
    if plan is None:
        plan = RowPlan()
        setattr(model, ROW_PLAN, plan)
    return plan
