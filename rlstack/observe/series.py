"""Per-run series assembly: everything the graphs need, from peeks alone.

run_series(store, run_id) returns the run's self-description (dictionary.json,
which carries the loss-walkback panel priority) joined with its committed
history: per-update post-column means and train rails from the ledger,
held-out means from the eval summaries, and progress. A pure function of
committed bytes.
"""

from __future__ import annotations

from rlstack.data.stores.base import Store, run_progress
from rlstack.observe.panels import PanelError, evaluate, missing_args


def run_series(store: Store, run_id: str,
               panels: list[dict] | None = None) -> dict | None:
    """None when the run does not exist in this store."""
    manifest = store.peek_manifest(run_id)
    if manifest is None:
        return None
    # what the run counts up to, and which plan that is: the train plan's
    # length where a run trains, since one wave is one gradient update (#59),
    # the rollout plan's where it only generates (ADR 0006 Part B). None for a
    # run with no plan bytes at all.
    progress = run_progress(store, run_id)

    entries = store.peek_ledger(run_id)
    updates = [{
        "update": int(entry["update"]),
        "post": dict(entry.get("post", {})),
        "train": dict(entry.get("train", {})),
    } for entry in entries]

    # held-out means, two eras under one shape: PRE-#70 runs measured inside
    # the run dir (eval summaries); everything since is a MEASUREMENT outside
    # it — each named observation becomes its own dashed series
    evals = [{"update": int(s["update"]), "means": dict(s.get("means", {}))}
             for s in store.peek_eval_summaries(run_id) if "update" in s]
    measurements = [
        {"name": name,
         "manifest": told["manifest"],
         "points": [{"update": int(p["update"]),
                     "means": dict(p.get("means", {}))}
                    for p in told["points"] if "update" in p]}
        for name, told in sorted(store.read_measurements(run_id.rsplit("/", 1)[-1]).items())]

    dictionary = store.peek_dictionary(run_id)
    return {
        "run_id": run_id.rsplit("/", 1)[-1],
        "run_ref": run_id,
        "dictionary": dictionary,
        "updates": updates,
        "eval": evals,
        "measurements": measurements,
        "derived": derived_series(
            panels if panels is not None else store.read_panels(),
            dictionary, updates, evals),
        # a run whose extent is ROLLOUTS commits nothing: its ledger panels
        # are empty and `completed` counts the waves its Generator sealed
        "extent": progress.extent,
        "committed": progress.completed,
        "target": progress.planned,
        # the ONE predicate, SERVED rather than re-derived (ADR 0008, F6): a
        # campaign door follows a run to its extent by polling this route, so
        # "is it finished" must be the observer's own answer and not a rule
        # every caller reimplements against `committed` and `target`.
        "done": progress.done,
    }


def derived_series(panels: list[dict], dictionary: dict | None,
                   updates: list[dict], evals: list[dict]) -> list[dict]:
    """Each panel evaluated per update over the rails ∪ post namespace (post
    wins a collision), and over the eval means too when all its arguments
    exist there. A panel with an argument outside this run's pipeline carries
    its missing list instead of points: panels.py's validation, rendered."""
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
