"""A trainer-only scalar head at a hidden boundary: it has no rollout lowering
at all, so a bundle pins its version like any other delta but never ships a
payload for it. What it contributes is `values`, a tensor the replay forward
provides: v(t) read off the trunk's CAUSAL state at every position — the
prefix value E[reward | prompt, completion <= t] once fit — for any loss
that wants per-position value estimates (none in the zoo today; the head
stands ready and its provided curve is watchable regardless).

The trunk state h_t already encodes the whole prefix, task included, so the
head itself is deliberately tiny: RMS-norm, one hidden layer, a
zero-initialized scalar — values are exactly zero at version 0, and the first
updates train only the critic (the loss's whitening turns zero credit into
zero surrogate).

init: d_model (the boundary's hidden width — a boundary site has no shape, so
the spec states it), hidden (the head's own width)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlstack.policy.adapters.base import AdapterType, adapter_type
from rlstack.policy.siteschema import SiteMeta

VALUES_PROVIDED = "values"     # [rows, padded_tokens], zero at padding


@adapter_type("value_head")
class ValueHead(AdapterType):
    serving = None
    provides = frozenset({VALUES_PROVIDED})

    def site_ok(self, meta: SiteMeta) -> bool:
        return meta.is_boundary and not meta.has_weight

    # compute half — torch loads lazily, from here only (rule 7)

    def params(self, sites: tuple[SiteMeta, ...], init: dict):
        from rlstack.policy.adapters import value_head_torch
        return value_head_torch.build(sites, init)

    def install_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import value_head_torch
        value_head_torch.install(model, params)

    def uninstall_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import value_head_torch
        value_head_torch.uninstall(model, params)

    def provide(self, params) -> Mapping[str, Any]:
        """Per-position values for the forward that just ran, off the captured
        boundary states — recomputed per microbatch, differentiable into the
        head's own parameters and nothing else (the capture is detached, so
        value training never leaks gradient into the policy's adapters)."""
        from rlstack.policy.adapters import value_head_torch
        return value_head_torch.provide(params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import value_head_torch
        return value_head_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import value_head_torch
        value_head_torch.load(params, payload)


def value_head(site: str, d_model: int, hidden: int = 128):
    """Sugar, beside lora()/plora(): the scalar head at one boundary site."""
    from rlstack.spec.specs import AdapterSpec
    return AdapterSpec(adapter_type="value_head", site=site,
                       init={"d_model": d_model, "hidden": hidden})
