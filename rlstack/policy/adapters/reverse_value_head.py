"""A REVERSE-autoregressive value head at a hidden boundary: trainer-only,
anti-causal, hindsight-informed by construction.

The classic critic estimates value from the PREFIX — it must guess how the
trajectory will end. This head reads the trunk's hidden states the other way:
the value at position t attends to positions t..T, the suffix — what the
policy actually went on to do — under its own learned REVERSE positional
encoding (distance to the end of the document, the natural coordinate of a
return-to-go). It is therefore not an unbiased baseline and does not pretend
to be one: reverse_ppo uses it as a return DECOMPOSITION (the credit at t is
the drop in suffix value from t to t+1, telescoping to the whole return), the
RUDDER idea with a reverse attention block in place of the forward LSTM.

Like value_head it has NO rollout lowering: a bundle pins its version like
any other delta but never ships a payload for it, and the engine never hears
its name. What it contributes is `reverse_values`, a [rows, padded_tokens]
tensor the replay forward provides — computed from DETACHED trunk states, so
its training never leaks gradient into the policy's own adapters.

init: d_model (the boundary's hidden width — a boundary site has no shape, so
the spec states it), heads (attention heads), max_len (the reverse position
table's reach; longer documents clamp)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlstack.policy.adapters.base import AdapterType, adapter_type
from rlstack.policy.siteschema import SiteMeta

VALUES_PROVIDED = "reverse_values"     # [rows, padded_tokens], zero at padding


@adapter_type("reverse_value_head")
class ReverseValueHead(AdapterType):
    serving = None
    provides = frozenset({VALUES_PROVIDED})

    def site_ok(self, meta: SiteMeta) -> bool:
        """A hidden boundary, not a weight: the head reads representations,
        it does not steer a matrix."""
        return meta.is_boundary and not meta.has_weight

    # compute half — torch loads lazily, from here only (rule 7)

    def params(self, sites: tuple[SiteMeta, ...], init: dict):
        from rlstack.policy.adapters import reverse_value_head_torch
        return reverse_value_head_torch.build(sites, init)

    def install_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import reverse_value_head_torch
        reverse_value_head_torch.install(model, params)

    def uninstall_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import reverse_value_head_torch
        reverse_value_head_torch.uninstall(model, params)

    def provide(self, params) -> Mapping[str, Any]:
        """The suffix values for the forward that just ran, off the captured
        boundary states — recomputed per microbatch, differentiable into the
        head's own parameters and nothing else (the capture is detached)."""
        from rlstack.policy.adapters import reverse_value_head_torch
        return reverse_value_head_torch.provide(params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import reverse_value_head_torch
        return reverse_value_head_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import reverse_value_head_torch
        reverse_value_head_torch.load(params, payload)


def reverse_value_head(site: str, d_model: int, heads: int = 4,
                       max_len: int = 2048):
    """Sugar, beside lora()/plora(): the head at one boundary site."""
    from rlstack.spec.specs import AdapterSpec
    return AdapterSpec(adapter_type="reverse_value_head", site=site,
                       init={"d_model": d_model, "heads": heads,
                             "max_len": max_len})
