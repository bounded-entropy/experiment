"""dream_bank's rollout lowering: one punica adapter PER SET, selected per
request by a `Route` directive — plus the engine's LIBRARY adapters and
STACKED routes (ADR 0019).

Each bundle attaches N+1 LoRARequests, one per set. A route may also name a
frozen library adapter (`lib:<name>`, bytes the engine's Library holds) and
may stack two parts (`lib:<name>+dreamer`): a stack is served as ONE adapter
whose weights are the row/column concatenation of its parts
(`lora_torch.stack_fragments`), exact because every set is scaling 1. Library
adapters and stacks are materialized LAZILY, at the first request that routes
under them, and cached: a library adapter per name, a stack per (bundle id,
route). A stack dies with its bundle (`detach`) or with its library entry
(`forget_library`); the dreamer moving to a new version is a new bundle id, so
its stacks are simply built again under the new id.

THE SLOT ARITHMETIC. vLLM's `max_loras` is how many DISTINCT adapters one
batch may carry; it sizes punica's slot bank (every slot allocated at
`max_lora_rank`), and a request whose adapter finds no free slot WAITS in the
scheduler — it is never refused, and an adapter pushed out of vLLM's cache is
reloaded from its dir. So the budget is a throughput number: the distinct
adapters one wave keeps in flight. With B bundles being sampled, M memories
and L distinct library names in a wave that is

    B x (M + 1)     the sets
  + B x L           the stacks (one per bundle per name)
  + L               the library adapters alone (shared by every bundle)

and the build carries ONE number, `slots() = max_bundles x (max_members + 1)`,
so a desk prices library traffic through `max_members`:
`max_members >= M + L + ceil(L / max_bundles)`. A dreamer at r=32 stacked on
r=32 library adapters, 8 names a wave, 2 bundles resident, no memories:
`max_rank = 64`, `max_members = 12` (26 slots = 2 x 9 + 8), and `max_library
>= 8` (the default 32 holds four waves' names).

Materializing writes an adapter dir from inside `apply`, on the engine's loop:
tens of MB and well under a second for the q/v sets this was built for, once
per (bundle, route). vLLM and torch are imported at module scope — this file
loads only from DreamBank.rollout_lowering (STYLE rule 7).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import save as st_save
from vllm.lora.request import LoRARequest

from rlstack.policy.adapters import lora_torch
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.dream_bank import (
    BASE, DREAMER, ROUTE_RECORD, check_route, is_library, library_name,
    library_names_of, library_route, parts_of,
)
from rlstack.policy.adapters.dream_bank_torch import split_sets
from rlstack.policy.adapters.rollout import (
    Alignment, BuildDemands, Levers, Request, RolloutLowering, ServingBuild,
    check_rank_fits,
)
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES

def check_parts(route: str, memories: int) -> tuple[str, ...]:
    """The route's parts, once the ONE route grammar (`dream_bank.check_route`)
    has passed it: one part or a stack of exactly two distinct ones; a part is
    a set this entry holds or a library adapter, and `base` — no delta at all
    — stands alone."""
    check_route(route, memories)
    return parts_of(route)


@dataclass
class AttachedBank:
    """What one bundle's dream_bank state is on this engine: every set as its
    own adapter, by route."""

    bundle_id: str
    memories: int
    sets: dict[str, LoRARequest]


class DreamBankRollout(RolloutLowering):
    """The dreamer and every memory, each its own punica adapter; library
    adapters and stacks beside them, built when first routed under."""

    adapter_type = "dream_bank"
    mechanism = Mechanism.PUNICA
    claims = ("lora_request",)

    def __init__(self, build: ServingBuild) -> None:
        super().__init__(build)
        # `forget_library` and `detach` arrive on the door's threads while
        # `apply` runs on the engine's loop: one lock over both caches.
        self._lock = threading.Lock()
        self._library_adapters: dict[str, LoRARequest] = {}          # by name
        self._stacks: dict[tuple[str, str], LoRARequest] = {}        # by (bundle id, route)

    def demands(self) -> BuildDemands:
        """The build's one slot budget and its widest rank — see THE SLOT
        ARITHMETIC above for what a desk declares so library adapters and
        stacks have slots."""
        return BuildDemands(engine_args={
            "enable_lora": True,
            "max_loras": self.build.slots(),
            "max_lora_rank": self.build.max_rank})

    def reaches(self, meta: SiteMeta) -> bool:
        return meta.has_weight

    # ---- a bundle's sets ------------------------------------------------------

    def attach(self, bundle_id: str, payloads: Mapping[str, bytes]) -> AttachedBank:
        """Every set of every payload of this entry, as its own adapter dir
        and LoRARequest, keyed by route."""
        attached = AttachedBank(bundle_id=bundle_id, memories=0, sets={})
        for name in sorted(payloads):
            meta, fragments = split_sets(payloads[name])
            attached.memories = int(meta["memories"])
            for route, tensors in fragments.items():
                attached.sets[route] = self._adapter(
                    f"{bundle_id}/{route}",
                    self._bundle_dir(bundle_id) / route.replace(":", "_"),
                    st_save(tensors))
        return attached

    def detach(self, attached: AttachedBank) -> None:
        """The bundle's sets and every stack built on them."""
        with self._lock:
            for key in [key for key in self._stacks if key[0] == attached.bundle_id]:
                del self._stacks[key]
        shutil.rmtree(self._bundle_dir(attached.bundle_id), ignore_errors=True)

    # ---- one request ----------------------------------------------------------

    def apply(self, attached: AttachedBank, request: Request) -> Levers:
        """The adapter the route names, and the route recorded; `base` is no
        adapter at all. A library part is read from the engine's Library on
        EVERY request — that read is the refusal of a name nobody handed in,
        and what keeps a name in use recent."""
        directive = ADAPTER_TYPES.get(self.adapter_type).instance.directive_for(request)
        route = DREAMER if directive is None else directive.name
        parts = check_parts(route, attached.memories)
        if route == BASE:
            return Levers(turn_extras={ROUTE_RECORD: route})
        names = library_names_of(route)
        library = {name: self.build.library.get(name) for name in names}
        if route in attached.sets:
            adapter = attached.sets[route]
        elif len(parts) == 1:
            adapter = self.library_adapter(names[0], library[names[0]])
        else:
            adapter = self.stack(attached, route, parts, library)
        return Levers(kwargs={"lora_request": adapter},
                      turn_extras={ROUTE_RECORD: route}, library=names)

    def library_adapter(self, name: str, payload: bytes) -> LoRARequest:
        """`lib:<name>` alone: the named payload as its own adapter, one per
        name whatever bundle the request pins — a library adapter does not
        depend on the bundle."""
        with self._lock:
            if name not in self._library_adapters:
                self._library_adapters[name] = self._adapter(
                    f"lib/{name}", _named_dir(self.build.workdir / "library", name), payload)
            return self._library_adapters[name]

    def stack(self, attached: AttachedBank, route: str, parts: tuple[str, ...],
              library: Mapping[str, bytes]) -> LoRARequest:
        """`<part>+<part>`: the two parts' weights concatenated into ONE
        adapter, built once per (bundle id, route)."""
        key = (attached.bundle_id, route)
        with self._lock:
            if key not in self._stacks:
                first, second = (
                    library[library_name(part)] if is_library(part)
                    else _set_payload(attached.sets[part]) for part in parts)
                self._stacks[key] = self._adapter(
                    f"{attached.bundle_id}/{route}",
                    _named_dir(self._bundle_dir(attached.bundle_id) / "stacks", route),
                    lora_torch.stack_fragments(first, second))
            return self._stacks[key]

    def forget_library(self, name: str) -> None:
        """The Library evicted `name`: its own adapter and every stack that
        reads it go with it. A request in flight pinned the name, so nothing
        here is under a live request."""
        part = library_route(name)
        with self._lock:
            gone = [self._library_adapters.pop(name)] if name in self._library_adapters else []
            for key in [key for key in self._stacks if part in parts_of(key[1])]:
                gone.append(self._stacks.pop(key))
        for adapter in gone:
            shutil.rmtree(adapter.lora_path, ignore_errors=True)

    def align(self, attached: AttachedBank) -> Alignment:
        return Alignment(0)

    # ---- adapter dirs -----------------------------------------------------------

    def _bundle_dir(self, bundle_id: str) -> Path:
        return self.build.workdir / bundle_id.replace(":", "_")

    def _adapter(self, lora_name: str, adapter_dir: Path, payload: bytes) -> LoRARequest:
        """One named payload as an adapter dir and its LoRARequest — refused,
        and its dir removed, when it is wider than the build's kernels."""
        rank = lora_torch.write_adapter_dir(adapter_dir, self.build.base, {lora_name: payload})
        try:
            check_rank_fits(f"adapter {lora_name!r}", rank, self.build)
        except ValueError:
            shutil.rmtree(adapter_dir, ignore_errors=True)
            raise
        return LoRARequest(lora_name=lora_name, lora_int_id=self.build.next_lora_id(),
                           lora_path=str(adapter_dir))


def _named_dir(parent: Path, name: str) -> Path:
    """A dir for something named by a library name: the name may hold `/`, so
    the dir is a digest of it plus a readable tail."""
    digest = hashlib.sha256(name.encode()).hexdigest()[:12]
    return parent / f"{digest}-{re.sub(r'[^A-Za-z0-9._-]', '_', name)[-48:]}"


def _set_payload(adapter: LoRARequest) -> bytes:
    """A set's named payload, read back from the adapter dir attach wrote —
    a bundle's sets are not kept in memory twice."""
    return (Path(adapter.lora_path) / "adapter_model.safetensors").read_bytes()
