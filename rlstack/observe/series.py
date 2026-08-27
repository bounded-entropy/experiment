"""Per-run series assembly: everything the UI's graphs need, from peeks alone.

One call — run_series(store, run_id) — returns the run's self-description
(dictionary.json, which carries the loss-walkback panel priority) joined with
its committed history: per-update post-column means and train rails from the
ledger, held-out means from the eval summaries, and progress. Pure function
of committed bytes; never attaches, never writes (the observe/ rule)."""

from __future__ import annotations

import json

from rlstack.data.stores.base import Store


def run_series(store: Store, run_id: str) -> dict | None:
    """None when the run does not exist in this store."""
    manifest = store.peek_manifest(run_id)
    if manifest is None:
        return None
    try:
        spec = json.loads(manifest["spec"])
        target = spec["algo"]["schedule"]["n_updates"]
    except (KeyError, TypeError, json.JSONDecodeError):
        target = None

    entries = store.peek_ledger(run_id)
    updates = [{
        "update": int(entry["update"]),
        "post": dict(entry.get("post", {})),
        "train": dict(entry.get("train", {})),
    } for entry in entries]

    evals = [{"update": int(s["update"]), "means": dict(s.get("means", {}))}
             for s in store.peek_eval_summaries(run_id) if "update" in s]

    return {
        "run_id": run_id,
        "dictionary": store.peek_dictionary(run_id),
        "updates": updates,
        "eval": evals,
        "committed": updates[-1]["update"] if updates else 0,
        "target": target,
    }
