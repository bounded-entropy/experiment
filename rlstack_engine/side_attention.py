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

NOT PROVEN, AND THE PROBE SAYS SO (#46). The lifecycle below (banks, slots,
salt) is real; attend() is not, and on the PINNED vllm 0.28.0 build it cannot
be written as #25 designed it. What was checked in the image:

  - the registration seam MOVED and still exists: `register_backend` is at
    vllm.v1.attention.backends.registry, not vllm.attention.backends.registry,
    and AttentionBackendEnum even carries a CUSTOM member for third-party
    backends. load_general_plugins() still runs in BOTH the engine core and
    the worker, so ambient registration (#16) is intact.
  - merge_attn_states MOVED and still exists: vllm.v1.attention.ops.
    merge_attn_states (plus a triton variant and a _custom_ops binding).
  - THE BLOCKER: return_softmax_lse is NOT plumbed through the dense
    FlashAttention path. FlashAttentionImpl sets can_return_lse_for_decode =
    True, but v1/attention/backend.py reads it only when dcp_world_size > 1;
    every other user of return_softmax_lse on this build is an MLA backend or
    the context-parallel helper. A dense Qwen3 forward has no seam that hands
    the LSE back, so "stock kernel + tiny partition attention + exact merge"
    would mean forking 262 lines of FA-version-conditioned kernel dispatch.
    That is the hack this stops short of.

WHAT WOULD UNBLOCK IT, in order of preference:
  1. FLEX ATTENTION, which this build already has and which is a BETTER seam
     than the one #25 designed against: FlexAttentionMetadata carries a
     first-class `score_mod` field — torch's (score, b, h, q_idx, kv_idx) ->
     score hook, i.e. exactly an additive bias on the score rectangle — and
     get_transformed_score_mod() already converts paged physical KV indices to
     logical per-request ones, which is the job BatchView was invented for.
     Nothing in vLLM sets score_mod today, so the work is a registered backend
     plus a metadata builder that carries our per-request slot vector; the
     cost is that FLEX_ATTENTION becomes the whole engine's backend (a build
     fact, and a different numerics baseline for every tenant on it).
  2. return_softmax_lse on the dense FlashAttention path, i.e. #25's original
     design, if a later vLLM plumbs it the way the MLA backends already do.

Until one of those is built and CERTIFIED, probe() fails on this build and
every engine honestly reports NONE for SIDE_ATTENTION — so a spec carrying an
attn_bias is refused at Phase 0 rather than served wrong. The replay half IS
built and proven (rlstack/policy/adapters/attn_bias_torch.py).
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
