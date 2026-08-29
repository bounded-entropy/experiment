"""plora's FROZEN half: the top-k singular directions of each matched weight,
and the content-addressed artifact that ships them.

The trained half of a plora entry is kilobytes and rides in the bundle; the
directions it steers are megabytes per site and are IDENTICAL at every policy
version, so they are computed once, addressed by content, and named from the
spec. That split is the whole reason this file exists separately from
plora_torch.py: one side of the adapter never changes and therefore never
belongs in a payload.

THE FACTORIZATION IS PINNED, NOT MERELY DETERMINISTIC. `ALGO_ID` names the
exact recipe — fp32, the SMALLER Gram matrix through `eigh`, descending
eigenvalues, signs canonicalized by each right-vector's largest-magnitude
entry. It is stamped into the artifact and checked when the artifact is read,
because U and A are the coordinate system both lowerings express the delta in:
two sides disagreeing about a sign is not a small numerical difference, it is a
different policy. The trainer recomputes the same function from the base it
already holds (plora_torch.install); the engine reads it from here, because
vLLM's copy of the weights is not cheaply readable and a second full checkpoint
load is not worth one eigendecomposition.

torch is imported at module scope — this file loads only from the adapter
type's methods (STYLE rule 7).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from rlstack.policy.siteschema import SiteMeta

# The recipe, by name. Editing the body below without editing this string would
# let two artifacts claim one coordinate system; the constant is what makes the
# refusal in read_factors mean something.
ALGO_ID = "gram-eigh-fp32-canonical-sign-v1"

# What the artifact STORES. bf16 because the engine serves bf16 and the
# artifact is megabytes per site — fp32 would double it to buy precision vLLM
# discards on load. The stated cost: the engine's A is the bf16 rounding of the
# fp32 A the trainer recomputes, so the two lowerings differ by exactly that
# rounding. It sits under the bf16 kernel floor `logprob_gap` already measures.
FACTOR_DTYPE = torch.bfloat16
U_SUFFIX = ".U"                     # [out, k] left singular vectors
A_SUFFIX = ".A"                     # [k, in]  Sigma_k V_k^T (Sigma absorbed)


# ---------------------------------------------------------------------------
# the container: ONE representation for both of this adapter type's artifacts
# ---------------------------------------------------------------------------

def pack_artifact(meta: Mapping[str, object],
                  tensors: Mapping[str, torch.Tensor]) -> bytes:
    """A JSON head and a safetensors body, in one blob.

    safetensors carries tensors and nothing else, and plora has scalars to
    ship beside them (the algorithm's name, the base, k, a version counter).
    Rather than smuggling those in as zero-dim tensors or pairing two files,
    the artifact is a length-prefixed JSON header followed by the safetensors
    bytes. Keys are sorted and separators fixed, so the same values always
    produce the same bytes — which is what lets a payload hash into a
    bundle_id and an artifact into a cas address.
    """
    head = json.dumps(dict(meta), sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return len(head).to_bytes(8, "big") + head + st_save(dict(tensors))


def unpack_artifact(payload: bytes) -> tuple[dict, dict[str, torch.Tensor]]:
    """pack_artifact's inverse: (head, tensors)."""
    size = int.from_bytes(payload[:8], "big")
    return json.loads(payload[8:8 + size]), st_load(payload[8 + size:])


# ---------------------------------------------------------------------------
# the factorization
# ---------------------------------------------------------------------------

def canonical_signs(u: torch.Tensor,
                    vh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fix the one freedom an SVD leaves: each direction's SIGN.

    (u_i, v_i) and (-u_i, -v_i) span the same subspace and reconstruct the same
    matrix, so `eigh` may return either on any machine. The convention here is
    the reference implementation's: make each right vector's largest-magnitude
    entry positive, and flip the matching left vector with it. Without this the
    frozen coordinate system would be per-run, and a core trained against one
    sign would be the wrong policy under the other.
    """
    pivots = vh.gather(1, vh.abs().argmax(dim=1, keepdim=True)).squeeze(1)
    signs = torch.where(pivots < 0, -torch.ones_like(pivots),
                        torch.ones_like(pivots))
    return u * signs[None, :], vh * signs[:, None]


def top_svd_factors(weight: torch.Tensor,
                    k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The top-k truncated SVD of one weight, as the pair plora freezes:
    `U_k` [out, k] and `A = Sigma_k V_k^T` [k, in], so U_k C A is exactly a
    rank-k delta whose middle is the only thing that trains.

    Through the SMALLER Gram matrix and `eigh` rather than a full `svd`: a
    [out, in] weight only ever needs the min(out, in) side decomposed, which is
    what makes factoring a whole attention stack cheap. fp32 throughout,
    whatever the checkpoint's dtype — the Gram matrix squares the condition
    number and bf16 would not survive it.

    Sigma is absorbed into A rather than kept, because the engine's consumer is
    peft, whose format is a plain (lora_A, lora_B) pair with no room for a third
    factor. A leading singular value at zero means the top-k subspace is not
    k-dimensional, and it is refused rather than divided by.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"plora factors a weight MATRIX, got shape {tuple(weight.shape)}")
    if not 0 < k <= min(weight.shape):
        raise ValueError(
            f"plora k={k} must be in 1..{min(weight.shape)} for a "
            f"{tuple(weight.shape)} weight — a rank above the matrix's own is "
            f"not a bigger policy, it is an undefined one")
    matrix = weight.detach().to(dtype=torch.float32)
    rows, columns = map(int, matrix.shape)
    if rows <= columns:
        eigenvalues, u = torch.linalg.eigh(matrix @ matrix.mT)
        singular = _leading(eigenvalues, k, matrix.dtype)
        u = u[:, -k:].flip(1).contiguous()
        vh = (u.mT @ matrix) / singular[:, None]
    else:
        eigenvalues, v = torch.linalg.eigh(matrix.mT @ matrix)
        singular = _leading(eigenvalues, k, matrix.dtype)
        v = v[:, -k:].flip(1).contiguous()
        u = (matrix @ v) / singular[None, :]
        vh = v.mT
    u, vh = canonical_signs(u.contiguous(), vh.contiguous())
    return u.contiguous(), (singular[:, None] * vh).contiguous()


def _leading(eigenvalues: torch.Tensor, k: int,
             dtype: torch.dtype) -> torch.Tensor:
    """The k largest singular values, descending. `eigh` returns eigenvalues
    ascending, so the top-k are the tail, flipped; they are clamped at zero
    because a Gram matrix's eigenvalues are non-negative in exact arithmetic
    and slightly negative in floating point."""
    top = eigenvalues[-k:].flip(0).clamp_min(0.0).sqrt()
    if bool((top <= torch.finfo(dtype).eps).any()):
        raise ValueError(
            f"plora's leading {k} singular values are not all positive: this "
            f"weight's top-{k} subspace is rank-deficient, so U_k C A cannot "
            f"be a rank-{k} parameterization of it")
    return top


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------

WeightReader = Callable[[str], torch.Tensor]


def build_factors(base_id: str, site_metas: Sequence[SiteMeta], k: int,
                  weight_reader: WeightReader) -> bytes:
    """Factor every matched site once and pack the result: the bytes a caller
    `cas_put`s and a spec then names.

    `weight_reader` maps a site PATH to that site's [out, in] weight — a seam
    rather than a checkpoint, so the same builder serves an already-loaded
    model, an offline shard reader (hf_weight_reader below), or a test's
    handful of tensors. Sites are taken in path order, so the artifact's bytes
    — and therefore its cas address — are a pure function of (base, k, sites).
    """
    metas = sorted(site_metas, key=lambda meta: meta.path)
    tensors: dict[str, torch.Tensor] = {}
    for meta in metas:
        if not meta.has_weight:
            raise ValueError(
                f"plora factors weighted sites only; {meta.name!r} has no weight")
        u, a = top_svd_factors(weight_reader(meta.path), k)
        tensors[meta.path + U_SUFFIX] = u.to(FACTOR_DTYPE)
        tensors[meta.path + A_SUFFIX] = a.to(FACTOR_DTYPE)
    return pack_artifact({"algo": ALGO_ID, "base": base_id, "k": int(k),
                          "paths": [meta.path for meta in metas]}, tensors)


def read_factors(payload: bytes, base: str | None = None,
                 k: int | None = None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The artifact, back as {site path: (U [out, k], A [k, in])} — and the
    attestation that it is the right one.

    An artifact is addressed by content, so it cannot be the WRONG BYTES; it
    can still be the right bytes for a different base or a different k, which
    a caller that pinned an address in a spec would never otherwise notice.
    `base` and `k` are checked when given, and the algorithm's name always is:
    a coordinate system the reader does not recognize is refused rather than
    interpreted.
    """
    meta, tensors = unpack_artifact(payload)
    if meta.get("algo") != ALGO_ID:
        raise ValueError(
            f"plora factors were built by {meta.get('algo')!r}, this code reads "
            f"{ALGO_ID!r} — the two do not share a sign convention, so the "
            f"cores trained against one are the wrong policy under the other")
    if base is not None and meta.get("base") != base:
        raise ValueError(
            f"plora factors were built for base {meta.get('base')!r}, this "
            f"policy serves {base!r}")
    if k is not None and int(meta.get("k", -1)) != int(k):
        raise ValueError(
            f"plora factors are rank {meta.get('k')!r}, this entry declares "
            f"k={k}")
    return {path: (tensors[path + U_SUFFIX], tensors[path + A_SUFFIX])
            for path in meta["paths"]}


def hf_weight_reader(model_id: str, revision: str | None = None) -> WeightReader:
    """A WeightReader over a HF checkpoint's safetensors shards, WITHOUT
    building the model.

    Factoring needs one matrix at a time; instantiating a base to get them
    costs its full weight memory for nothing. This walks the repo's shard index
    instead and slices out `<path>.weight` on demand, which is what lets the
    factors be built on a small machine, or beside a learner that is already
    holding the base for something else. Imports are guarded because
    huggingface_hub is deploy-time metal, not part of the client library.
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
    except ImportError as exc:                    # the client environment
        raise ImportError(
            "reading factors from a HF checkpoint needs huggingface_hub and "
            "safetensors, which ship in the deploy image; pass your own "
            "weight_reader to build_factors off the metal") from exc

    def shard_of() -> dict[str, str]:
        """tensor key -> shard filename, for a sharded or a single-file repo."""
        try:
            index = hf_hub_download(model_id, "model.safetensors.index.json",
                                    revision=revision)
        except Exception:                          # single-file checkpoint
            return {}
        with open(index, encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]

    weight_map = shard_of()
    handles: dict[str, object] = {}

    def read(path: str) -> torch.Tensor:
        key = f"{path}.weight"
        shard = weight_map.get(key, "model.safetensors")
        if shard not in handles:
            handles[shard] = safe_open(
                hf_hub_download(model_id, shard, revision=revision),
                framework="pt")
        return handles[shard].get_tensor(key)

    return read
