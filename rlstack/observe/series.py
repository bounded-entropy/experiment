"""Per-run series assembly: everything the UI's graphs need, from peeks alone.

One call — run_series(store, run_id) — returns the run's self-description
(dictionary.json, which carries the loss-walkback panel priority) joined with
its committed history: per-update post-column means and train rails from the
ledger, held-out means from the eval summaries, and progress. Pure function
of committed bytes; never attaches, never writes (the observe/ rule)."""

from __future__ import annotations

import json

from rlstack.data.stores.base import Store
from rlstack.observe.panels import PanelError, evaluate, missing_args


def run_series(store: Store, run_id: str,
               panels: list[dict] | None = None) -> dict | None:
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

    dictionary = store.peek_dictionary(run_id)
    return {
        "run_id": run_id,
        "dictionary": dictionary,
        "updates": updates,
        "eval": evals,
        "derived": derived_series(
            panels if panels is not None else store.read_panels(),
            dictionary, updates, evals),
        "committed": updates[-1]["update"] if updates else 0,
        "target": target,
    }


def derived_series(panels: list[dict], dictionary: dict | None,
                   updates: list[dict], evals: list[dict]) -> list[dict]:
    """Each panel evaluated per update (post ∪ rails namespace, post wins)
    and — when its arguments all exist there — over the eval means too. A
    panel with arguments outside this run's pipeline carries its missing
    list instead of points: the validation rule, rendered."""
    out = []
    for panel in panels:
        name, expr = str(panel["name"]), str(panel["expr"])
        try:
            missing = missing_args(expr, dictionary)
        except PanelError as err:
            out.append({"name": name, "expr": expr, "error": str(err),
                        "points": [], "eval": []})
            continue
        if missing:
            out.append({"name": name, "expr": expr,
                        "missing": list(missing), "points": [], "eval": []})
            continue
        points = []
        for update in updates:
            value = evaluate(expr, {**update["train"], **update["post"]})
            if value is not None:
                points.append([update["update"], value])
        eval_points = []
        for entry in evals:
            value = evaluate(expr, dict(entry["means"]))
            if value is not None:
                eval_points.append([entry["update"], value])
        out.append({"name": name, "expr": expr,
                    "points": points, "eval": eval_points})
    return out
