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


class SiteWrapper(torch.nn.Module):
    """The shared half of every module-replacing replay site: one wrapper per
    (family, path), a roster that makes install additive, and CHAIN mechanics
    so two FAMILIES at one path nest instead of colliding.

    A site path carries at most one delta PER TENANT (the bank rule), but a
    shared learner's tenants may bring different families to one path — a
    lora tenant and a plora tenant both at q_proj. Each family's wrapper
    applies only the rows whose routed state IS its own (the family filter is
    the subclass's forward), and passes every other row through to `inner` —
    which is the base Linear, or the OTHER family's wrapper, whichever
    join_site found standing there first.
    """

    def __init__(self, inner: torch.nn.Module, path: str, plan: RowPlan) -> None:
        super().__init__()
        self.inner = inner
        self.path = path
        self.plan = plan
        self.installed: list[Any] = []

    def add(self, state: Any) -> None:
        """Additive install: this state's delta becomes routable here."""
        if any(present is state for present in self.installed):
            raise RuntimeError(f"install at {self.path}: this state is already "
                               f"installed — install/uninstall out of balance")
        self.installed.append(state)

    def drop(self, state: Any) -> None:
        """install's inverse at one site; leave_site unwraps when empty."""
        kept = [present for present in self.installed if present is not state]
        if len(kept) == len(self.installed):
            raise RuntimeError(f"uninstall at {self.path}: this state was never "
                               f"installed — install/uninstall out of balance")
        self.installed = kept


def join_site(model: Any, path: str, wrapper: type,
              state: Any) -> "SiteWrapper":
    """Find this FAMILY's wrapper in the chain at `path` — joining it if it
    stands, wrapping the chain's head if it does not — and add the state.

    The chain is walked, not assumed: another family may already hold the
    path, and wrapping AROUND it is exactly right — each wrapper transparently
    passes rows that are not its own, so nesting order never changes a number.
    """
    parent, leaf = leaf_module(model, path)
    node = getattr(parent, leaf)
    probe = node
    while isinstance(probe, SiteWrapper):
        if isinstance(probe, wrapper):
            probe.add(state)
            return probe
        probe = probe.inner
    site = wrapper(node, path, row_plan(model))
    setattr(parent, leaf, site)
    site.add(state)
    return site


def leave_site(model: Any, path: str, wrapper: type, state: Any) -> None:
    """join_site's exact inverse: drop the state from this family's wrapper,
    and splice the wrapper OUT of the chain when its last state leaves —
    whether it is the chain's head (the module attribute) or nested inside
    another family's wrapper."""
    parent, leaf = leaf_module(model, path)
    node = getattr(parent, leaf)
    outer: SiteWrapper | None = None
    while isinstance(node, SiteWrapper) and not isinstance(node, wrapper):
        outer = node
        node = node.inner
    if not isinstance(node, wrapper):
        raise RuntimeError(
            f"uninstall at {path}: no {wrapper.__name__} in the chain — "
            f"install/uninstall out of balance")
    node.drop(state)
    if not node.installed:
        if outer is None:
            setattr(parent, leaf, node.inner)
        else:
            outer.inner = node.inner


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
