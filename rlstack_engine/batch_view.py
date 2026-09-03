"""BatchView: everything a plugin may know about one batch, and nothing else.

The ONE version-pinned shim. Plugins are written against this frozen record,
and the constructor that reads engine internals — which token belongs to which
request, where in its sequence each token sits, which request carries what
per-request selection — is the only code tracking the engine's metadata
layout across versions. When the engine moves a field, this file changes and
no plugin does.

Two halves. `view_of` is the PURE builder: from the columns any continuous-
batching engine's metadata reduces to (cumulative query offsets, sequence
lengths, per-request facts) to per-token columns — testable with no engine.
`from_vllm` is the version-pinned reader of those columns off vLLM 0.28.0's
forward context and model runner, and it calls the pure builder.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class BatchViewError(ValueError):
    """Misaligned batch view — column lengths must agree."""


@dataclass(frozen=True)
class BatchView:
    """Per-token facts for one layer's forward, aligned to the query rows,
    plus the per-REQUEST facts those rows came from.

    `token_slot[i]` is the bank slot of the bundle token i's request pinned
    (-1: none for this plugin) — the multi-tenant gather index, the plugin's
    twin of punica's per-token adapter index. `position[i]` is token i's
    absolute position in its own sequence (prompt positions at prefill, the
    next position at each decode step), what a per-request WINDOW is judged
    against. `is_decode[i]` distinguishes decode rows from prefill rows.
    `token_request[i]` names the request row token i belongs to, and
    `request_extra[r]` is that request's per-request selection
    (SamplingParams.extra_args as the engine carries it, {} when none) — the
    channel a plugin's adapter type wrote through Levers.extra_args.
    """

    token_slot: tuple[int, ...]
    layer_idx: int
    is_decode: tuple[bool, ...]
    position: tuple[int, ...] = ()
    token_request: tuple[int, ...] = ()
    request_extra: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        columns = {"is_decode": len(self.is_decode)}
        if self.position:
            columns["position"] = len(self.position)
        if self.token_request:
            columns["token_request"] = len(self.token_request)
        bad = {name: n for name, n in columns.items() if n != len(self.token_slot)}
        if bad:
            raise BatchViewError(
                f"token_slot has {len(self.token_slot)} rows but {bad} "
                f"disagree")
        if self.token_request and max(self.token_request) >= len(self.request_extra):
            raise BatchViewError(
                f"token_request names request {max(self.token_request)} but "
                f"only {len(self.request_extra)} request rows are present")

    def __len__(self) -> int:
        return len(self.token_slot)

    @staticmethod
    def from_vllm(model_runner: Any, layer_idx: int,
                  slot_of: Callable[[Mapping[str, Any]], int]) -> "BatchView | None":
        """The view of the forward in flight, read off vLLM 0.28.0's forward
        context and GPU model runner — the engine-side seam, and the one
        function that knows where vLLM keeps these facts.

        The forward context's attention metadata carries the batch geometry
        (cumulative query offsets, sequence lengths; one entry per attention
        layer, all sharing it on a dense model); the runner's input batch
        names the requests in batch order and its request table carries each
        one's sampling params. `slot_of` is the plugin's rule for reading its
        own selection out of a request's extra_args (-1 for none). None means
        there is no batch to view — a warm-up or profiling run with no
        attention metadata — and a hook treats it as nothing to do.
        """
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return None
        if isinstance(metadata, dict):          # per-layer on 0.28; same geometry
            metadata = next(iter(metadata.values()))
        query_start_loc = metadata.query_start_loc.tolist()
        seq_lens = metadata.seq_lens.tolist()
        request_ids = list(model_runner.input_batch.req_ids)[:len(seq_lens)]
        extras = tuple(
            dict(model_runner.requests[rid].sampling_params.extra_args or {})
            for rid in request_ids)
        return view_of(query_start_loc, seq_lens,
                       [slot_of(extra) for extra in extras], extras, layer_idx)


def view_of(query_start_loc: Sequence[int], seq_lens: Sequence[int],
            request_slots: Sequence[int],
            request_extra: Sequence[Mapping[str, Any]],
            layer_idx: int) -> BatchView:
    """The pure builder: per-request columns to per-token ones.

    Request r's tokens are rows query_start_loc[r] .. query_start_loc[r+1] of
    the flattened batch; its sequence is seq_lens[r] long AFTER this forward,
    so its first row sits at absolute position seq_lens[r] - n_query. A
    request scheduling one token is decoding.
    """
    n_requests = len(seq_lens)
    if (len(query_start_loc) != n_requests + 1 or len(request_slots) != n_requests
            or len(request_extra) != n_requests):
        raise BatchViewError(
            f"{n_requests} requests need {n_requests + 1} query offsets and "
            f"{n_requests} slots and extras; got {len(query_start_loc)}, "
            f"{len(request_slots)}, {len(request_extra)}")
    token_slot: list[int] = []
    is_decode: list[bool] = []
    position: list[int] = []
    token_request: list[int] = []
    for r in range(n_requests):
        n_query = int(query_start_loc[r + 1]) - int(query_start_loc[r])
        first = int(seq_lens[r]) - n_query
        for j in range(n_query):
            token_slot.append(int(request_slots[r]))
            is_decode.append(n_query == 1)
            position.append(first + j)
            token_request.append(r)
    return BatchView(token_slot=tuple(token_slot), layer_idx=layer_idx,
                     is_decode=tuple(is_decode), position=tuple(position),
                     token_request=tuple(token_request),
                     request_extra=tuple(dict(e) for e in request_extra))
