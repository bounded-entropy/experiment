"""attn_bias's replay lowering — the half that CAN be proven today.

The rule that makes this side easy: an attention mask already IS a score-level
bias, so handing the base a 4-D float mask instead of the 2-D padding mask
patches no kernel. The bias applies to the REAL tokens' rows only — the
prompt's own positions stay a pure function of the rows, which is what would
let an engine precompute their K/V once per bundle. Version 0 IS the base:
theta starts at zero and both parameterizations map zero to zero bias.

The rollout half is refused on the pinned build (attn_bias_vllm.py), so nothing
here is reachable from a run; this is what a serving path would have to agree
with once one exists.

torch is imported at module scope — this file loads only from the adapter
type's methods (STYLE rule 7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from rlstack.policy.adapters.replay import ReplayRows
from rlstack.policy.siteschema import SiteMeta

BIAS_KEY = "attn_bias.theta"
BIAS_PATH = "attn_scores"        # the path a soft prompt exports the rectangle at

_PROMPT_WIDTH = re.compile(r"prompt\[:(\d+)\]")


def prompt_width(name: str) -> int:
    """How many prompt rows the site name says the rectangle spans.

    The canonical name carries the shape ("queries -> prompt[:8]") and
    non-module names match exactly, so the bias never has to be told `n`
    twice — reading it here is reading the spec, not guessing.
    """
    found = _PROMPT_WIDTH.search(name)
    if found is None:
        raise ValueError(
            f"an attn_bias lives on a soft prompt's exported rectangle, whose "
            f"name carries its width (queries -> prompt[:n]); got {name!r}")
    return int(found.group(1))


# ---------------------------------------------------------------------------
# the parameterizations: theta -> a bias in nats. One named function each.
# ---------------------------------------------------------------------------

def free(theta: torch.Tensor, cap: float) -> torch.Tensor:
    """The parameter IS the bias, in nats, unbounded."""
    return theta


def bounded_sigmoid(theta: torch.Tensor, cap: float) -> torch.Tensor:
    """A bias confined to (-cap, cap): the attention can be nudged but never
    made to ignore or fixate on the prompt, which is what keeps a learned bias
    from collapsing a rollout."""
    return cap * (2.0 * torch.sigmoid(theta) - 1.0)


PARAMETERIZATIONS = {"free": free, "bounded_sigmoid": bounded_sigmoid}


@dataclass
class AttnBiasState:
    """One bank entry's trainable state: theta per (head, prompt row)."""

    n: int
    heads: int
    path: str
    param: str
    cap: float
    theta: torch.nn.Parameter        # [heads, n]

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.theta]

    def value(self) -> torch.Tensor:
        """The bias itself: theta through this entry's parameterization."""
        return PARAMETERIZATIONS[self.param](self.theta, self.cap)


# ---------------------------------------------------------------------------
# the replay lowering: the mask the boundary runs its forward under
# ---------------------------------------------------------------------------

def routed_bias(rows: ReplayRows) -> torch.Tensor | None:
    """[rows, heads, n] — each row's bias — or None when no slot carries one.

    The same shape of rule as the other adapter types: rows of ONE forward may
    carry different biases, but they may not disagree about whether there IS
    one, nor about the rectangle's size.
    """
    present = [BIAS_PATH in slot for slot in rows.slots]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "one forward's slots disagree about the attention bias: some carry "
            "one at queries -> prompt[:n] and some do not")
    shapes = sorted({(slot[BIAS_PATH].heads, slot[BIAS_PATH].n)
                     for slot in rows.slots})
    if len(shapes) > 1:
        raise ValueError(
            f"one forward's slots must agree on the biased rectangle, got {shapes}")
    return torch.stack([slot[BIAS_PATH].value()
                        for slot in rows.slots])[rows.index]


def additive_mask(bias: torch.Tensor, attention: torch.Tensor,
                  prepended: int, dtype: torch.dtype) -> torch.Tensor:
    """The 4-D additive mask a biased forward runs under: causal, padding-aware,
    and carrying `bias` on the prompt columns of every REAL token's row.

    `attention` is the boundary's already-widened [rows, length] padding mask,
    so the virtual positions are inside it and `prepended` says how many. The
    blocked entries are finfo.min rather than -inf on purpose: a fully padded
    query row then softmaxes to something uniform and finite instead of NaN, and
    nobody reads those positions anyway.

    WHAT IT COSTS, stated: [rows, heads, length, length] materialized, where the
    ordinary path passes [rows, length]. That is the price of expressing a score
    bias through the one argument the stock attention takes, and it is not
    optimized here — the rollout half this would be certified against does not
    exist yet.
    """
    rows, length = attention.shape
    heads, n = bias.shape[1], bias.shape[2]
    if n != prepended:
        raise ValueError(
            f"the bias spans {n} prompt rows but this forward prepended "
            f"{prepended} — the rectangle and the soft prompt must agree")
    device = attention.device
    blocked = torch.finfo(dtype).min
    causal = torch.ones((length, length), dtype=torch.bool,
                        device=device).tril()                     # [L, L]
    allowed = causal[None] & attention.bool()[:, None, :]          # [R, L, L]
    mask = torch.where(allowed[:, None], 0.0, blocked).to(dtype)   # [R, 1, L, L]
    mask = mask.expand(rows, heads, length, length).clone()
    mask[:, :, prepended:, :prepended] += bias[:, :, None, :].to(dtype)
    return mask


# ---------------------------------------------------------------------------
# the bodies the AttnBias adapter type's methods call
# ---------------------------------------------------------------------------

def build(sites: tuple[SiteMeta, ...], init: dict) -> AttnBiasState:
    """theta = 0 (the bias starts at the base), one per (head, prompt row)."""
    if len(sites) != 1:
        raise ValueError(
            f"an attn_bias lives at ONE exported rectangle, matched {len(sites)}: "
            f"{', '.join(m.name for m in sites[:4])}")
    meta = sites[0]
    param = str(init.get("param", "free"))
    if param not in PARAMETERIZATIONS:
        raise ValueError(
            f"unknown attn_bias parameterization {param!r}; the registered ones "
            f"are {', '.join(sorted(PARAMETERIZATIONS))}")
    cap = float(init.get("cap", 4.0))
    heads = int(init["heads"])
    n = prompt_width(meta.name)
    return AttnBiasState(
        n=n, heads=heads, path=meta.path, param=param, cap=cap,
        theta=torch.nn.Parameter(torch.zeros(heads, n, dtype=torch.float32)))


def install(model: torch.nn.Module, state: AttnBiasState) -> None:
    """Placement, and the head-count attestation — nothing else.

    There is no module to wrap: the bias reaches the forward through the
    attention mask the prompt boundary owns (soft_prompt_torch), and that
    boundary finds this state the same way every replay lowering finds its own,
    in the row plan's slot. A soft prompt is guaranteed to be in the bank,
    because the rectangle is a site only that entry exports.
    """
    heads = int(model.config.num_attention_heads)
    if state.heads != heads:
        raise ValueError(
            f"this attn_bias declares heads={state.heads}, but the base attends "
            f"with {heads} — a bias is one scalar per head per prompt row")
    embed = model.get_input_embeddings()
    state.theta.data = state.theta.data.to(embed.weight.device)


def uninstall(model: torch.nn.Module, state: AttnBiasState) -> None:
    """install's exact inverse: install wrapped nothing, so this unwraps
    nothing. What leaves the forward is the routability of the state, which is
    the learner dropping the slot."""


def emit(state: AttnBiasState) -> bytes:
    """safetensors with ONE key: theta. The parameterization is spec (it hashes
    into run identity), so the payload carries numbers, never behavior."""
    return st_save({BIAS_KEY: state.theta.data.cpu()})


def load(state: AttnBiasState, payload: bytes) -> None:
    state.theta.data.copy_(st_load(payload)[BIAS_KEY])
