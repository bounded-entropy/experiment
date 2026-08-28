"""EnginePlugin: one serving mechanism the stock engine lacks.

A native lever is one the engine's own maintainers keep working — multi-tenant,
cache-correct, graph-captured, TP-sharded — across versions. A plugin must
RE-EARN each of those at a seam the engine never promised to keep stable, so
each earned property is one named method: probe (the seams exist on THIS build,
so a failure lands at boot and never mid-run), install (claim the seam),
load/evict (payloads into per-slot banks, multi-tenant from day one — banks
plus a per-token slot index, never a global singleton), cache_salt (a plugin
that changes hidden states MUST make bundle identity visible to the prefix
cache), and attend (the per-layer merge, written against BatchView only).

That a plugin's rollout lowering agrees numerically with its adapter type's
replay lowering is the parity certificate's job, keyed by build fingerprint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from rlstack import Mechanism

from rlstack_engine.batch_view import BatchView


@dataclass(frozen=True)
class EngineBuild:
    """One deployed engine build, as a plugin sees it.

    `fingerprint` keys the certificate cache; `symbols` is the build's
    inventory of internal seams, which probe checks required_symbols against.
    """

    fingerprint: str
    attention_backend: str
    symbols: frozenset[str]


class ProbeError(RuntimeError):
    """A required seam is missing on this build. Raised at boot, on purpose."""


class Seam(ABC):
    """Where a plugin attaches to the engine. A vLLM seam claims via
    register_backend; the fake seam records the claim for tests."""

    @abstractmethod
    def claim(self, plugin: "EnginePlugin") -> None:
        """Install `plugin` at this seam. Claiming twice is an error."""


class EnginePlugin(ABC):
    """Subclass per mechanism; one shared instance per engine process."""

    mechanism: ClassVar[Mechanism]
    consumes: ClassVar[tuple[str, ...]]              # adapter KINDS it serves
    required_symbols: ClassVar[frozenset[str]] = frozenset()

    def probe(self, build: EngineBuild) -> None:
        """Every required seam must exist on this build — the I7 boot probe."""
        missing = sorted(self.required_symbols - build.symbols)
        if missing:
            raise ProbeError(
                f"{type(self).__name__} needs seams this build lacks: "
                f"{', '.join(missing)} (build {build.fingerprint})")

    def install(self, seam: Seam) -> None:
        """Claim the seam. Call probe first; install assumes it passed."""
        seam.claim(self)

    @abstractmethod
    def load(self, slot: int, bundle_id: str, payloads: Mapping[str, bytes]) -> None:
        """Compile `payloads` (one per consumed bank entry, by entry name) into
        the banks at `slot`. Bundle-constant precompute happens here."""

    @abstractmethod
    def evict(self, slot: int) -> None:
        """Free `slot`'s banks. The SlotTable owns slot lifetimes."""

    def cache_salt(self, bundle_id: str) -> str | None:
        """Prefix-cache key contribution for a request pinned to `bundle_id`.
        Default: the bundle itself — correct for any plugin whose math changes
        hidden states. Return None ONLY from a numerics-neutral plugin."""
        return bundle_id

    @abstractmethod
    def attend(self, view: BatchView, q: object, out: object, lse: object) -> None:
        """Merge this mechanism's contribution into `out` in place, reading
        per-token slots from `view` — never from engine internals."""
