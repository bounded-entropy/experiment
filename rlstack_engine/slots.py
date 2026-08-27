"""SlotTable: which bundle is resident in which bank slot.

The plugin analog of vLLM's LoRA slot management: banks are fixed-capacity
GPU tensors indexed by slot, so bundles must be assigned slots, found again,
and evicted. Acquisition is idempotent (add_bundle may be called twice for
the same bundle); eviction is explicit — the engine adapter decides policy,
this table only keeps the books.
"""

from __future__ import annotations


class SlotsFull(RuntimeError):
    """No free slot: capacity is a build fact; evict something first."""


class SlotTable:
    def __init__(self, max_slots: int) -> None:
        if max_slots < 1:
            raise ValueError(f"max_slots must be >= 1, got {max_slots}")
        self.max_slots = max_slots
        self._slot_of: dict[str, int] = {}
        self._free: list[int] = list(range(max_slots - 1, -1, -1))  # pop() -> 0 first

    def acquire(self, bundle_id: str) -> int:
        """The bundle's slot, assigning one if it is not resident."""
        if bundle_id in self._slot_of:
            return self._slot_of[bundle_id]
        if not self._free:
            raise SlotsFull(
                f"all {self.max_slots} slots resident; evict before loading "
                f"{bundle_id!r}")
        slot = self._free.pop()
        self._slot_of[bundle_id] = slot
        return slot

    def slot_of(self, bundle_id: str) -> int:
        """The resident bundle's slot; KeyError if it was never loaded."""
        return self._slot_of[bundle_id]

    def release(self, bundle_id: str) -> int:
        """Free the bundle's slot and return it (so banks can be cleared)."""
        slot = self._slot_of.pop(bundle_id)
        self._free.append(slot)
        return slot

    def resident(self) -> dict[str, int]:
        """{bundle_id: slot} for everything currently loaded."""
        return dict(self._slot_of)
