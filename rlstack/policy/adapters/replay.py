"""The replay routing seam: which slot each ROW of a trainer forward carries.

A slot is one tenant's installed deltas; a RowPlan is routed for exactly one
padded forward and RAISES when a lowering runs unrouted, because an unrouted
replay forward is a wiring bug and never a fallback to whoever went last. The
trainer-side twin of the engine's per-token adapter index (I8).

The plan rides ON THE MODEL because the model is the one handle install_replay
receives, so routing reaches every wired lowering without widening the adapter
type contract. torch is imported at module scope: this file loads from the
adapter types' compute halves and the learner, never from the package root.
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

    A SLOT is one tenant's installed deltas, addressed the way a site asks for
    them: {site path -> the params object holding that site's delta}. `slots` is
    the slot order and `index` is [rows] long — row r applies slot index[r].
    Every Learner verb pins one tenant, so today each microbatch is the
    degenerate one-slot case; a coalesced microbatch is the same record with
    more slots and a mixed index.

    `facts` is the other half of the routing: row r's RECORDED turn extras, one
    mapping per turn of that row's document. Deltas are what a slot holds and
    facts are what its rows carry, so a lowering whose math depends on a
    sampling-time draw (a probabilistic latent) reads the draw from here rather
    than re-deriving something the rollout already decided (I6). The learner
    fills it ADAPTER-BLIND — it copies the batch's per-document mappings across
    and never looks inside — so a new recording adapter type needs no change
    here. None means the batch recorded nothing.
    """

    slots: tuple[Mapping[str, Any], ...]
    index: torch.Tensor                 # [rows], long
    facts: tuple[tuple[Mapping[str, Any], ...], ...] | None = None

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("a replay forward routes to at least one slot")
        if int(self.index.max()) >= len(self.slots):
            raise ValueError(
                f"row plan reaches slot {int(self.index.max())} of "
                f"{len(self.slots)} — rows carry installed slots only")
        if self.facts is not None and len(self.facts) != int(self.index.shape[0]):
            raise ValueError(
                f"row plan carries facts for {len(self.facts)} rows but routes "
                f"{int(self.index.shape[0])} — facts are addressed BY ROW")

    def uniform(self) -> Mapping[str, Any] | None:
        """The one slot every row carries, or None when the rows disagree.

        THE parity anchor: with one slot a site applies the plain (x A^T) B^T
        over the whole batch in one GEMM pair, so a single-tenant microbatch is
        bit-identical however many tenants share the learner, and the per-row
        path is entered only when it is actually needed.
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
        """Pin `rows` for one forward AND the backward that recomputes it.

        Checkpointed blocks run their forward a second time inside backward()
        (torch_learner.checkpoint_the_blocks), so a plan released at the end of
        the first pass would leave the recomputed one unrouted — the caller
        holds this open across loss.backward() for that reason.
        """
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


def leaf_module(model: Any, path: str) -> tuple[Any, str]:
    """The (parent module, attribute name) a site path addresses.

    A module-replacing replay lowering has to reach INTO the tree to swap a
    leaf, and every one of them addresses it the same way — the site's path is
    the base's own dotted `named_modules()` name (siteschema.SiteMeta.path), so
    the walk belongs here beside the routing rather than once per adapter type.
    """
    parent = model
    *walk, leaf = path.split(".")
    for step in walk:
        parent = getattr(parent, step)
    return parent, leaf


def row_plan(model: Any) -> RowPlan:
    """The model's one row plan, created by whoever asks first.

    Both sides ask here — the learner to route a forward, an adapter type's
    install_replay to hand the plan to the sites it wires — so the table is
    per loaded base and exactly one function knows where it lives.
    """
    plan = getattr(model, ROW_PLAN, None)
    if plan is None:
        plan = RowPlan()
        setattr(model, ROW_PLAN, plan)
    return plan
