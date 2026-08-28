"""The packed forms: sealed trajectories → microbatches.

`flatten` turns one trajectory into a complete flat token record (`Flat`),
`broadcast` turns a per-trajectory postdata column into a per-token channel,
and `pack` fills `TokenBatch`es bounded by microbatch_tokens. Nothing here
knows what an advantage or a loss is, and nothing estimator-shaped rides
along.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from rlstack.data.trajectory import DataError, Trajectory


@dataclass(frozen=True)
class Flat:
    """One trajectory as tokens — everything flatten() captures in one pass.

    token_extras carries the per-token columns the bank's adapter types
    recorded at rollout (e.g. adapter draws); injected positions hold None in
    each column.
    """

    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]              # 1 on generated tokens, 0 on injected ones
    segment_ids: tuple[int, ...]            # turn index on generated tokens, -1 injected
    behavior_logprobs: tuple[float, ...]    # recorded at generation (I6); 0.0 injected
    doc_len: int
    token_extras: Mapping[str, tuple] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.token_ids)
        lengths = {
            "loss_mask": len(self.loss_mask),
            "segment_ids": len(self.segment_ids),
            "behavior_logprobs": len(self.behavior_logprobs),
            **{f"token_extras[{k}]": len(v) for k, v in self.token_extras.items()},
        }
        bad = {k: v for k, v in lengths.items() if v != n}
        if bad:
            raise DataError(f"Flat field lengths disagree with token_ids={n}: {bad}")
        if self.doc_len != n:
            raise DataError(f"doc_len={self.doc_len} != len(token_ids)={n}")


def flatten(traj: Trajectory, tokenize: Callable[[str], tuple[int, ...]]) -> Flat:
    """Walk traj.messages in order into one complete token document.

    A message that IS one of the turns' messages (matched by object identity)
    contributes that turn's token_ids, behavior_logprobs, and token_extras
    VERBATIM — never re-tokenized, never re-aligned (I6) — with loss_mask=1.
    Every other message was injected by the env (system prompt, tool result,
    ...) and is tokenized here with loss_mask=0, logprob 0.0, and None in each
    extras column.
    """
    turn_index = {id(t.message): i for i, t in enumerate(traj.turns)}
    columns = sorted({name for t in traj.turns for name in t.token_extras})
    ids: list[int] = []
    mask: list[int] = []
    seg: list[int] = []
    logprobs: list[float] = []
    extras: dict[str, list] = {name: [] for name in columns}
    for message in traj.messages:
        i = turn_index.get(id(message))
        if i is None:  # injected
            chunk = tuple(tokenize(message.content))
            ids += chunk
            mask += [0] * len(chunk)
            seg += [-1] * len(chunk)
            logprobs += [0.0] * len(chunk)
            for name in columns:
                extras[name] += [None] * len(chunk)
        else:  # generated
            turn = traj.turns[i]
            ids += turn.token_ids
            mask += [1] * len(turn.token_ids)
            seg += [i] * len(turn.token_ids)
            logprobs += turn.behavior_logprobs
            for name in columns:
                column = turn.token_extras.get(name)
                extras[name] += (list(column) if column is not None
                                 else [None] * len(turn.token_ids))
    return Flat(tuple(ids), tuple(mask), tuple(seg), tuple(logprobs), len(ids),
                token_extras={k: tuple(v) for k, v in extras.items()})


def broadcast(columns: Mapping[str, Sequence],
              flats: Sequence[Flat]) -> list[dict[str, tuple[float, ...]]]:
    """Per-trajectory postdata columns → per-token columns, one dict per doc.

    A trajectory's value is a SCALAR (repeated over its generated tokens) or
    a PER-TOKEN VECTOR (one float per generated token, consumed in order —
    the token_level channel). Either way injected positions get 0.0
    (loss_mask=0 carries no credit). Dumb by charter: shape logic only —
    the runner validated vectors against declarations upstream.
    """
    for name, values in columns.items():
        if len(values) != len(flats):
            raise DataError(
                f"postdata column {name!r} has {len(values)} values for "
                f"{len(flats)} trajectories")
    return [
        {name: _per_token(name, values[i], flat)
         for name, values in columns.items()}
        for i, flat in enumerate(flats)
    ]


def _per_token(name: str, value, flat: Flat) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return tuple(float(value) if m else 0.0 for m in flat.loss_mask)
    generated = sum(flat.loss_mask)
    if len(value) != generated:
        raise DataError(
            f"postdata column {name!r} has {len(value)} floats for a doc "
            f"with {generated} generated tokens")
    it = iter(value)
    return tuple(float(next(it)) if m else 0.0 for m in flat.loss_mask)


@dataclass(frozen=True)
class TokenBatch:
    """A packed microbatch: documents concatenated, boundaries in doc_starts.

    `postdata` carries the postprocessing pipeline's columns, broadcast per token —
    a loss reads the ones it declared in `requires` (e.g. postdata["advantage"]).
    """

    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    segment_ids: tuple[int, ...]
    doc_starts: tuple[int, ...]
    postdata: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    token_extras: Mapping[str, tuple] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.token_ids)
        lengths = {
            "loss_mask": len(self.loss_mask),
            "behavior_logprobs": len(self.behavior_logprobs),
            "segment_ids": len(self.segment_ids),
            **{f"postdata[{k}]": len(v) for k, v in self.postdata.items()},
            **{f"token_extras[{k}]": len(v) for k, v in self.token_extras.items()},
        }
        bad = {k: v for k, v in lengths.items() if v != n}
        if bad:
            raise DataError(f"TokenBatch field lengths disagree with token_ids={n}: {bad}")

    def __len__(self) -> int:
        return len(self.token_ids)


# One document ready to pack: its flat record plus its per-token postdata columns.
Doc = tuple[Flat, Mapping[str, tuple[float, ...]]]


def pack(items: Sequence[Doc], microbatch_tokens: int) -> list[TokenBatch]:
    """Greedy fill preserving order; a document is never split across microbatches.

    A single document longer than microbatch_tokens gets its own oversized batch.
    Injected tokens arrive with zeroed postdata columns (broadcast) and behavior
    logprob 0.0 (flatten).
    """
    if microbatch_tokens <= 0:
        raise DataError(f"microbatch_tokens must be positive, got {microbatch_tokens}")
    batches: list[TokenBatch] = []
    current: list[Doc] = []
    used = 0
    for doc in items:
        flat, postdata = doc
        for name, column in postdata.items():
            if len(column) != flat.doc_len:
                raise DataError(
                    f"postdata column {name!r} disagrees with doc_len={flat.doc_len}: "
                    f"{len(column)}")
        if current and used + flat.doc_len > microbatch_tokens:
            batches.append(_concatenate(current))
            current, used = [], 0
        current.append(doc)
        used += flat.doc_len
    if current:
        batches.append(_concatenate(current))
    return batches


def _concatenate(docs: Sequence[Doc]) -> TokenBatch:
    """Concatenate documents into one TokenBatch, recording each doc's offset.

    Every document must carry the same postdata and token_extras columns (one
    pipeline, one bank); a mismatch is a wiring error.
    """
    extra_columns = set(docs[0][0].token_extras)
    postdata_columns = set(docs[0][1])
    for flat, postdata in docs:
        if set(flat.token_extras) != extra_columns:
            raise DataError(
                f"token_extras columns differ across documents: "
                f"{sorted(extra_columns)} vs {sorted(flat.token_extras)}")
        if set(postdata) != postdata_columns:
            raise DataError(
                f"postdata columns differ across documents: "
                f"{sorted(postdata_columns)} vs {sorted(postdata)}")
    ids: list[int] = []
    mask: list[int] = []
    logprobs: list[float] = []
    seg: list[int] = []
    starts: list[int] = []
    postdata_out: dict[str, list[float]] = {name: [] for name in sorted(postdata_columns)}
    extras: dict[str, list] = {name: [] for name in sorted(extra_columns)}
    for flat, postdata in docs:
        starts.append(len(ids))
        ids += flat.token_ids
        mask += flat.loss_mask
        seg += flat.segment_ids
        logprobs += flat.behavior_logprobs
        for name in postdata_out:
            postdata_out[name] += postdata[name]
        for name in extras:
            extras[name] += flat.token_extras[name]
    return TokenBatch(
        token_ids=tuple(ids),
        loss_mask=tuple(mask),
        behavior_logprobs=tuple(logprobs),
        segment_ids=tuple(seg),
        doc_starts=tuple(starts),
        postdata={k: tuple(v) for k, v in postdata_out.items()},
        token_extras={k: tuple(v) for k, v in extras.items()},
    )
