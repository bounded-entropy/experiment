"""The soft prompt's compute half — imported lazily by the SoftPrompt adapter's
methods, so the client library stays importable without torch (STYLE rule 7).

Replay lowering = a BOUNDARY around the base's forward. The n learned rows are
prepended to the embedded batch, the padding mask grows by n ones, and the
positions they added are cut back off the logits before they leave. That last
step is the rule this file exists to enforce: forward_backward returns
[len(batch)] logprobs aligned to batch.token_ids, and a virtual row is a
position with NO token — so it must not survive the boundary. Virtual
positions exist between embed and logits and nowhere else, which is why
flatten, pack and the learner's gather need no knowledge of them at all.

The boundary is ROW-AWARE (#44), exactly as LoraSite is: installation is
additive (I8), every tenant installed on one base joins the one boundary
standing there, and which block of rows a document carries is read from the
forward's row plan (adapters/replay.py). A row whose slot holds no soft prompt
takes no virtual positions — the boundary is transparent for it, which is what
lets a lora-only tenant and a soft-prompt tenant share one learner.

A soft prompt has NO identity element. A LoRA's version 0 (B = 0) IS the base;
n virtual positions change the forward at version 0 by construction. Init is
therefore small, seeded, and part of the policy: `init_std` defaults to 0.02,
the initializer_range transformers gives the embedding matrix itself, so
version 0 is a real reproducible policy rather than a pretence of the base.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from rlstack.policy.adapters.replay import ReplayRows, RowPlan, row_plan
from rlstack.policy.siteschema import SiteMeta

BOUNDARY = "_rlstack_prompt_boundary"
ROWS_KEY = "soft_prompt.rows"
VIRTUAL_TOKEN = 0        # the placeholder id a virtual position wears on the wire


@dataclass
class SoftPromptState:
    """One bank entry's trainable state: n rows of width d at one boundary."""

    n: int
    d: int
    path: str                          # the embedding boundary this entry claims
    rows: torch.nn.Parameter           # [n, d]

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.rows]


class PromptBoundary:
    """The base's edge, owned by the soft prompts installed on it.

    ONE boundary per loaded base serves every installed state — the mirror of
    LoraSite, and for the same reason: whose rows a document carries is a
    property of the batch (the row plan), not of the module tree. `installed`
    is that roster; it is what makes install additive and what tells uninstall
    when the last tenant has left and the hooks come off.

    Two hooks, one rule each: the rows go IN at the embedding boundary
    (prepend_rows), and the positions they added come OUT before the logits
    (trim_virtual_positions) — so nothing above the boundary can tell they were
    ever there.

    The per-tenant rows deliberately do NOT register as parameters of the base:
    the base is shared and frozen, a soft prompt is one tenant's state, and the
    learner owns it through `params.parameters()`.
    """

    def __init__(self, model: torch.nn.Module, path: str, plan: RowPlan) -> None:
        self.path = path
        self.plan = plan
        self.embed = model.get_input_embeddings()
        self.installed: list[SoftPromptState] = []
        self._prepended = 0
        self._pre = model.register_forward_pre_hook(self.prepend_rows,
                                                    with_kwargs=True)
        self._post = model.register_forward_hook(self.trim_virtual_positions,
                                                 with_kwargs=True)

    def add(self, state: SoftPromptState) -> None:
        """Additive install: this state's rows become routable at this base."""
        if state.path != self.path:
            raise RuntimeError(
                f"this base's prompt boundary is {self.path!r}; the joining "
                f"soft prompt claims {state.path!r} — one boundary per base")
        if any(present is state for present in self.installed):
            raise RuntimeError("install at the prompt boundary: this state is "
                               "already installed — install/uninstall out of balance")
        self.installed.append(state)

    def drop(self, state: SoftPromptState) -> None:
        """install's inverse; the caller removes the hooks when empty."""
        kept = [present for present in self.installed if present is not state]
        if len(kept) == len(self.installed):
            raise RuntimeError("uninstall at the prompt boundary: this state was "
                               "never installed — install/uninstall out of balance")
        self.installed = kept

    def remove(self) -> None:
        """Unhook: the base is exactly the module it was before install."""
        self._pre.remove()
        self._post.remove()

    # ---- the two rules ------------------------------------------------------

    def prepend_rows(self, module, args, kwargs):
        """Virtual rows enter as EMBEDDINGS, one block per row of the padded
        batch, and the padding mask grows to cover them.

        The learner calls the base by keyword (torch_learner._batched_logprobs);
        a positional call is a wiring bug and says so rather than guessing which
        argument was the ids.
        """
        if args:
            raise RuntimeError(
                "the replay forward passes input_ids and attention_mask by "
                f"keyword; got {len(args)} positional argument(s)")
        rows = self.rows_for(int(kwargs["input_ids"].shape[0]))
        self._prepended = 0 if rows is None else int(rows.shape[1])
        if rows is None:
            return None                    # no soft prompt on this row's slot
        embeds = self.embed(kwargs["input_ids"])
        attention = kwargs["attention_mask"]
        widened = dict(kwargs)
        widened["input_ids"] = None
        widened["inputs_embeds"] = torch.cat([rows.to(embeds.dtype), embeds],
                                             dim=1)
        widened["attention_mask"] = torch.cat(
            [torch.ones(attention.shape[0], self._prepended,
                        dtype=attention.dtype, device=attention.device),
             attention], dim=1)
        widened["attention_mask"] = self.bias_the_mask(
            widened["attention_mask"], embeds.dtype)
        return args, widened

    def bias_the_mask(self, attention: torch.Tensor,
                      dtype: torch.dtype) -> torch.Tensor:
        """An attn_bias on this base's rectangle, folded into the mask.

        The bias's site (queries -> prompt[:n]) exists only because a soft
        prompt exported it, so the boundary that owns those positions is where
        a bias on them is applied — but the arithmetic stays with the kind
        (attn_bias_torch). Without a bias in the routed slot this returns the
        padding mask it was handed, unchanged and untouched: a base that never
        carries an attn_bias never pays for one.
        """
        from rlstack.policy.adapters import attn_bias_torch

        bias = attn_bias_torch.routed_bias(self.plan.rows)
        if bias is None:
            return attention
        return attn_bias_torch.additive_mask(bias, attention, self._prepended,
                                             dtype)

    def trim_virtual_positions(self, module, args, kwargs, output):
        """The rows leave before the logits do.

        The caller gets exactly one logit row per input token, so
        forward_backward's [len(batch)] alignment to batch.token_ids never sees
        a virtual position — the gather above this boundary is the same gather
        it was without a soft prompt in the bank.
        """
        prepended, self._prepended = self._prepended, 0
        if prepended:
            output.logits = output.logits[:, prepended:]
        return output

    # ---- routing ------------------------------------------------------------

    def rows_for(self, batch_rows: int) -> torch.Tensor | None:
        """The [rows, n, d] block this forward prepends, or None for none."""
        plan = self.plan.rows
        one = plan.uniform()
        if one is not None:
            state = one.get(self.path)
            return None if state is None else _whole_batch_rows(state, batch_rows)
        return _per_row_rows(plan, self.path, batch_rows)


def _whole_batch_rows(state: SoftPromptState, batch_rows: int) -> torch.Tensor:
    """Every row on one soft prompt: the same [n, d] block, broadcast."""
    return state.rows.unsqueeze(0).expand(batch_rows, -1, -1)


def _per_row_rows(rows: ReplayRows, path: str,
                  batch_rows: int) -> torch.Tensor | None:
    """Rows on different soft prompts: gather each row's block out of the slot
    stack — punica's shape, one block per row instead of one (A, B) per row.

    The slots must agree on n (a slot without a soft prompt counts as 0):
    virtual positions shift every position id after them, so rows of ONE padded
    forward cannot disagree on how many there are.
    """
    widths = sorted({0 if path not in slot else slot[path].n
                     for slot in rows.slots})
    if len(widths) > 1:
        raise ValueError(
            f"prompt boundary {path}: one forward's slots must agree on n, got "
            f"{widths} — virtual positions shift every position id after them")
    if widths == [0]:
        return None
    if rows.index.shape[0] != batch_rows:
        raise ValueError(
            f"prompt boundary {path}: the plan routes {rows.index.shape[0]} "
            f"rows, the forward carries {batch_rows}")
    return torch.stack([slot[path].rows for slot in rows.slots])[rows.index]


def _entry_seed(seed: int, path: str) -> int:
    """Init is seeded per site, so two entries at two boundaries differ."""
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# the five members' bodies
# ---------------------------------------------------------------------------

def build(sites: tuple[SiteMeta, ...], init: dict) -> SoftPromptState:
    """n rows of width d, N(0, init_std) seeded per site.

    A soft prompt matches exactly the one site it exported (prompt[:n]); a
    pattern that pulled in two boundaries would be two policies wearing one
    version number, so it is refused here rather than averaged.
    """
    if len(sites) != 1:
        raise ValueError(
            f"a soft prompt lives at ONE exported boundary, matched "
            f"{len(sites)}: {', '.join(m.name for m in sites[:4])}")
    meta = sites[0]
    n, d = int(init["n"]), int(init["d"])
    std = float(init.get("init_std", 0.02))
    generator = torch.Generator().manual_seed(
        _entry_seed(int(init.get("seed", 0)), meta.path))
    rows = torch.randn(n, d, generator=generator, dtype=torch.float32) * std
    return SoftPromptState(n=n, d=d, path=meta.path,
                           rows=torch.nn.Parameter(rows))


def install(model: torch.nn.Module, state: SoftPromptState) -> None:
    """Claim the base's boundary once, then ADD this state to it.

    Installation is additive (I8): a second tenant joins the PromptBoundary it
    finds instead of replacing it, which is exactly what lets one forward route
    its rows to either. Placement happens here — the rows move to the device
    the embedding table lives on, before any optimizer or load exists.
    """
    embed = model.get_input_embeddings()
    width = int(embed.weight.shape[1])
    if state.d != width:
        raise ValueError(
            f"this soft prompt declares d={state.d}, but the base's embedding "
            f"boundary is {width} wide — a virtual row IS an embedding")
    boundary = getattr(model, BOUNDARY, None)
    if boundary is None:
        boundary = PromptBoundary(model, state.path, row_plan(model))
        setattr(model, BOUNDARY, boundary)
    state.rows.data = state.rows.data.to(embed.weight.device)
    boundary.add(state)


def uninstall(model: torch.nn.Module, state: SoftPromptState) -> None:
    """install's exact inverse: drop this state, and unhook the boundary when
    the last one leaves. The params object survives untouched (its tensor is
    held by reference), so re-install restores identical numerics."""
    boundary = getattr(model, BOUNDARY, None)
    if boundary is None:
        raise RuntimeError(
            "uninstall at the prompt boundary: no boundary is installed on "
            "this base — install/uninstall out of balance")
    boundary.drop(state)
    if not boundary.installed:
        boundary.remove()
        delattr(model, BOUNDARY)


def emit(state: SoftPromptState) -> bytes:
    """safetensors with ONE key: the rows, exactly as the engine prepends them.
    n and d are the tensor's own shape, so the payload is self-describing and
    the engine-side consumer needs nothing else."""
    return st_save({ROWS_KEY: state.rows.data.cpu()})


def load(state: SoftPromptState, payload: bytes) -> None:
    """emit's inverse: restore the rows in place."""
    state.rows.data.copy_(st_load(payload)[ROWS_KEY])


def merge_rows(payloads: Mapping[str, bytes]) -> torch.Tensor:
    """Fuse the bank's row blocks into ONE [n, d] block, entry-name order.

    The prompt_embeds twin of lora_torch.merge_fragments: a mechanism compiles
    its adapters JOINTLY, so two soft prompts in one bank are two segments of
    one virtual prompt, concatenated in a deterministic order. They are
    positions in one sequence, so their widths must agree.
    """
    blocks = [st_load(payloads[name])[ROWS_KEY] for name in sorted(payloads)]
    widths = sorted({int(block.shape[1]) for block in blocks})
    if len(widths) > 1:
        raise ValueError(
            f"a bundle's soft prompts are segments of ONE virtual prompt and "
            f"must share a width, got {widths}")
    return torch.cat(blocks) if len(blocks) > 1 else blocks[0]
