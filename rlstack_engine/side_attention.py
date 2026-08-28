"""SideAttention: soft prompt + attention bias served jointly, by LSE merge.

The mechanism: the soft prompt's n positions are a causal PREFIX attending only
among themselves, so their K/V at every layer depend on nothing but the prompt
rows and are precomputed once per bundle at load(). Per layer the stock kernel
runs over the real tokens untouched (returning its LSE), each query's biased
attention over the n side positions is a dense matmul, and the two partials
combine exactly by log-sum-exp arithmetic. One plugin, two adapter types:
`consumes` names both and load() receives both entries' payloads, because a
mechanism compiles its adapters JOINTLY.

NOT PROVEN, AND THE PROBE SAYS SO. The lifecycle below — banks, slots, salt —
is real; attend() is not. On the pinned build the registration seam and the
merge kernel both exist (moved, not gone), but return_softmax_lse is NOT
plumbed through the dense FlashAttention path, so there is no seam that hands
the LSE back short of forking the kernel dispatch. required_symbols therefore
names the paths this build has plus the one it genuinely lacks, probe() fails
for the true reason, and every engine honestly reports NONE for SIDE_ATTENTION
— so a spec carrying an attn_bias is refused at Phase 0 rather than served
wrong. FlexAttention's score_mod is the seam that would unblock it. The replay
half IS built and proven (rlstack/policy/adapters/attn_bias_torch.py).
"""

from __future__ import annotations

from collections.abc import Mapping

from rlstack import Mechanism

from rlstack_engine.batch_view import BatchView
from rlstack_engine.plugin import EnginePlugin


class SideAttention(EnginePlugin):
    mechanism = Mechanism.SIDE_ATTENTION
    consumes = ("soft_prompt", "attn_bias")
    # The names a 0.28.0 build would have to answer to. They are deliberately
    # the CURRENT module paths, not #25's: probing for a symbol that moved
    # would fail for the wrong reason, and probing for one that was renamed
    # away would pass for the wrong one. The third is the one this build
    # genuinely lacks, and it is why probe() fails today.
    required_symbols = frozenset({
        "vllm.v1.attention.backends.registry.register_backend",
        "vllm.v1.attention.ops.merge_attn_states.merge_attn_states",
        "vllm.v1.attention.backends.flash_attn.dense_lse",
    })

    def __init__(self) -> None:
        # slot -> {bank entry name: payload}; a built attend() replaces the
        # values with per-layer side K/V (from the prompt rows) + bias banks.
        self._banks: dict[int, dict[str, bytes]] = {}
        self._bundle_at: dict[int, str] = {}

    # ---- bundle lifecycle ---------------------------------------------------

    def load(self, slot: int, bundle_id: str, payloads: Mapping[str, bytes]) -> None:
        if slot in self._banks:
            raise ValueError(
                f"slot {slot} already holds {self._bundle_at[slot]!r}; evict first")
        self._banks[slot] = dict(payloads)
        self._bundle_at[slot] = bundle_id

    def evict(self, slot: int) -> None:
        del self._banks[slot]
        del self._bundle_at[slot]

    def bank(self, slot: int) -> Mapping[str, bytes]:
        """The banks at `slot` — the one door to them; tests and a built
        attend() read through here, never the dict directly."""
        return self._banks[slot]

    # ---- forward ------------------------------------------------------------

    def attend(self, view: BatchView, q: object, out: object, lse: object) -> None:
        raise NotImplementedError(
            "dense side attention + LSE merge is unbuilt: vllm 0.28.0 does "
            "not plumb return_softmax_lse through the dense FlashAttention "
            "path, so the stock kernel's LSE never comes back and the two "
            "partials cannot be merged exactly. Every engine reports NONE for "
            "SIDE_ATTENTION and Phase 0 refuses the spec, so nothing reaches "
            "here on this build")
