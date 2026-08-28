"""BatchView: everything a plugin may know about one batch, and nothing else.

The ONE version-pinned shim. Plugins are written against this frozen record,
and the constructor that reads engine internals — which token belongs to which
request, which request pins which bundle — is the only code tracking the
engine's metadata layout across versions. When the engine moves a field, this
file changes and no plugin does.
"""

from __future__ import annotations

from dataclasses import dataclass


class BatchViewError(ValueError):
    """Misaligned batch view — column lengths must agree."""


@dataclass(frozen=True)
class BatchView:
    """Per-token facts for one layer's forward, aligned to the query rows.

    `token_slot[i]` is the bank slot of the bundle token i's request pinned —
    the multi-tenant gather index, the plugin's twin of punica's per-token
    adapter index. `is_decode[i]` distinguishes decode rows (attend to every
    side position) from prefill rows (causal against them).
    """

    token_slot: tuple[int, ...]
    layer_idx: int
    is_decode: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.token_slot) != len(self.is_decode):
            raise BatchViewError(
                f"token_slot has {len(self.token_slot)} rows but is_decode "
                f"has {len(self.is_decode)}")

    def __len__(self) -> int:
        return len(self.token_slot)

    @staticmethod
    def from_vllm(attn_metadata: object, layer_idx: int,
                  slot_of: dict[str, int]) -> "BatchView":
        """Build the view from vLLM attention metadata: the engine-side seam,
        unbuilt while no plugin's attend() is (see side_attention.py)."""
        raise NotImplementedError("B3: the vLLM metadata shim")
