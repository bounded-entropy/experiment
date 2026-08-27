"""SideAttention: soft prompt + attention bias served jointly, by LSE merge.

The mechanism: the soft prompt's n positions are a causal PREFIX, attending
only among themselves — so their K/V at every layer depend on nothing but the
prompt rows, and are precomputed ONCE per bundle at load(). Per layer, the
stock kernel runs over the real tokens untouched (returning its LSE); each
query's biased attention over the n side positions is a dense matmul; the two
partial results combine exactly via log-sum-exp arithmetic (vLLM's own
merge_attn_states kernel — the decomposition its cascade paths already use).

One plugin, two adapter kinds: `consumes` names both, and load() receives both
entries' payloads — E rows and bias — because a mechanism compiles its
adapters JOINTLY (the punica fragment-merge rule, stated as a rule).

B1 ships the lifecycle (banks, slots, salt) with payloads held as bytes;
B3 lands prefix K/V precompute and attend() behind probe().
"""

from __future__ import annotations

from collections.abc import Mapping

from rlstack import Mechanism

from rlstack_engine.batch_view import BatchView
from rlstack_engine.plugin import EnginePlugin


class SideAttention(EnginePlugin):
    mechanism = Mechanism.SIDE_ATTENTION
    consumes = ("soft_prompt", "attn_bias")
    required_symbols = frozenset({
        "flash_attn.return_softmax_lse",   # the backend can hand back the LSE
        "merge_attn_states",               # vLLM's LSE-combine kernel
    })

    def __init__(self) -> None:
        # slot -> {bank entry name: payload}; B3 replaces the values with
        # per-layer side K/V tensors (from the soft prompt rows) + bias banks.
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
        """The banks at `slot` (tests and the B3 attend read through this)."""
        return self._banks[slot]

    # ---- forward ------------------------------------------------------------

    def attend(self, view: BatchView, q: object, out: object, lse: object) -> None:
        raise NotImplementedError("B3: dense side attention + LSE merge")
