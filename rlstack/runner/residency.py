"""BundleResidency: what a pool is holding, and the rule for letting go.

An engine's registration is additive (I8) — but additive is not infinite, and
before detach existed it may as well have been: one bundle per tenant per
update, each with its own state, for the life of the process. The working set
is nothing like that. It is the newest committed version, whatever an eval
point pins, and the lag window; everything else is dead the moment a newer
version commits.

THE RULE, one bound and one immunity: hold at most `max_bundles` bundles that
actually carry state, evicting least-recently-used first, and NEVER evict one a
request is pinning — a request's policy is immune once submitted (I8). The
bound is therefore soft by exactly the in-flight set, which is the honest
trade: refusing to serve, or evicting under a live request, would both be worse
than briefly holding one bundle too many.

Eviction is safe because it is not a loss: every bundle here is a committed
policy version, and runner/restore.py brings any of them back from the store
with the content-addressed id as proof it is the same one.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping

# One bundle's state, adapter type -> whatever that type's attach() made.
Attached = Mapping[str, object]


class BundleResidency:
    def __init__(self, max_bundles: int,
                 detach: Callable[[Attached], None]) -> None:
        self.max_bundles = max_bundles
        self._detach = detach
        # bundle_id -> attached, in LEAST-RECENTLY-USED ORDER: a dict keeps
        # insertion order and `touch` reinserts, so the eviction candidate is
        # always the first key.
        self._held: dict[str, Attached] = {}
        self._inflight: dict[str, int] = {}     # bundle_id -> requests pinning

    def knows(self, bundle_id: str) -> bool:
        """Is this bundle resident right now? Not "was it ever registered" —
        residency is not durable, and the caller's next move on a miss is to
        restore it, not to fail."""
        return bundle_id in self._held

    def attached(self, bundle_id: str) -> Attached:
        """What each adapter type made resident for this bundle."""
        return self._held.get(bundle_id, {})

    def census(self) -> Mapping[str, int]:
        """How many resident bundles each adapter type holds state for."""
        counts: dict[str, int] = {}
        for state in self._held.values():
            for adapter_type in state:
                counts[adapter_type] = counts.get(adapter_type, 0) + 1
        return counts

    def hold(self, bundle_id: str, attached: Attached) -> None:
        """Take one bundle's state, then come back within the bound."""
        self._held[bundle_id] = attached
        self.evict_until_within_bound()

    def release(self, bundle_id: str) -> None:
        """Hand one bundle's residency back to its adapter types."""
        self._detach(self._held.pop(bundle_id))

    def touch(self, bundle_id: str) -> None:
        """Mark most-recently-used: reinsertion moves it to the end."""
        if bundle_id in self._held:
            self._held[bundle_id] = self._held.pop(bundle_id)

    @contextlib.contextmanager
    def pinned(self, bundle_id: str):
        """One request's hold: most-recently-used on entry, and immune from
        eviction until the last token is out (I8)."""
        self.touch(bundle_id)
        self._inflight[bundle_id] = self._inflight.get(bundle_id, 0) + 1
        try:
            yield
        finally:
            if self._inflight[bundle_id] > 1:
                self._inflight[bundle_id] -= 1
            else:
                self._inflight.pop(bundle_id)

    def evict_until_within_bound(self) -> None:
        """Drop least-recently-used bundles until the bound holds.

        TWO EXEMPTIONS, and both are about not throwing away what is in use.
        The MOST recently used bundle is never a candidate — it is the working
        set by definition, and evicting it would mean restoring it on the very
        next request. And a bundle carrying no state is neither counted nor
        evicted: a bare-base pool registers one, holding it costs nothing, and
        dropping it would take a servable id away for no memory back.

        A pinned bundle is skipped rather than stopping the sweep, so an old
        version held by a lagging request does not protect the merely stale
        versions behind it.
        """
        carrying = [bundle_id for bundle_id, state in self._held.items() if state]
        for bundle_id in carrying[:-1]:         # never the most recently used
            if len(carrying) <= self.max_bundles:
                return
            if self._inflight.get(bundle_id):
                continue                        # pinned: immune (I8)
            self.release(bundle_id)
            carrying.remove(bundle_id)
