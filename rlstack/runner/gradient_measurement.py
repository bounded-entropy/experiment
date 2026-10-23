"""Exact supervised documents and admission-safe calls for gradient measurement.

The caller chooses its ordinary adapter entry and optional route. These helpers
neither create a special fitting tenant nor perform optimizer steps.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from rlstack.client import GradientRow
from rlstack.data.flatten import Doc, Flat, TokenBatch, pack


def gradient_document(row: GradientRow, route: str | None) -> Doc:
    """Exact supervised token IDs and binary target mask, without retokenizing.

    One route applies to the whole document; a masked-out token is still causal
    context. The first token has no preceding prediction and must be unscored.
    """
    if not isinstance(row, GradientRow):
        raise TypeError("gradient rows must be GradientRow records")
    if not isinstance(row.row_id, str) or not row.row_id:
        raise ValueError("a gradient row needs a nonempty row_id")
    ids, mask = tuple(row.token_ids), tuple(row.loss_mask)
    if not ids or len(ids) != len(mask):
        raise ValueError("gradient token IDs and loss mask must have equal nonzero length")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("gradient token IDs must be nonnegative integers")
    if any(type(value) not in (int, bool) or value not in (0, 1) for value in mask):
        raise ValueError("gradient loss masks must be binary")
    if mask[0] or not any(mask):
        raise ValueError("gradient rows need a scored target after an unscored first token")
    return (Flat(ids, mask, tuple(0 if value else -1 for value in mask),
                 (0.0,) * len(ids), len(ids), turn_extras=() if route is None else ({"route": route},)), {})


def gradient_batches(rows: Sequence[GradientRow], route: str | None,
                     microbatch_tokens: int) -> tuple[TokenBatch, ...]:
    """Pack one loss's documents; its denominator is the whole row group."""
    if not rows:
        raise ValueError("a gradient loss needs at least one document")
    if microbatch_tokens < 1:
        raise ValueError("gradient microbatch_tokens must be positive")
    return tuple(pack([gradient_document(row, route) for row in rows], microbatch_tokens))


def gradient_row_record(row: GradientRow) -> dict[str, Any]:
    """Bounded measurement provenance; exact token/mask bytes remain in CAS."""
    encoded = json.dumps({"token_ids": row.token_ids, "loss_mask": row.loss_mask},
                         separators=(",", ":")).encode()
    return {"row_id": row.row_id, "tokens": len(row.token_ids),
            "selected_tokens": sum(row.loss_mask),
            "token_mask_sha256": hashlib.sha256(encoded).hexdigest()}


async def finish_thread_call(verb: Callable[[], Any]) -> Any:
    """Cancellation waits for the synchronous learner call before admission ends."""
    operation = asyncio.create_task(asyncio.to_thread(verb))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        # A cancellation cannot stop a thread or a remote learner operation.
        # Keep its admission and the fitting lock until that operation returns.
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not operation.cancelled():
            operation.exception()  # Retrieve a failed worker's exception as well.
        raise

