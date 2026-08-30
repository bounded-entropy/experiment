"""Selection and the overlay: many runs, one metric, one chart.

The QUERY GRAMMAR, spoken identically by this file and the web client: a
pipe WITH SPACES (" | ") separates OR clauses — an unspaced | stays inside
its term, so regex alternation like `k(4|8)` survives — whitespace inside a
clause is AND, and each term is tried as a case-insensitive REGEX over
run_id, name, note and every tag (joined), falling back to plain substring
when it does not compile. `prior=0.3 k=4 | latent=64` reads exactly as it
looks. Empty selects everything.

The overlay reads LEDGERS ONLY: a metric is any numeric the train block or
the post means carry, per update — which is every rail, every declared
provide, and every postdata column, with no schema and no registry.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rlstack.data.stores.base import Store
from rlstack.observe.locate import Root, rooted
from rlstack.observe.views import runs_data

SERIES_LIMIT = 80          # an overlay past this is soup; the reply says so
NAME_SAMPLE = 16           # ledgers scanned for the metric name list


def match_expr(row: dict, expr: str) -> bool:
    """One row against the grammar above."""
    expr = (expr or "").strip()
    if not expr:
        return True
    hay = " ".join([row.get("run_id", ""), row.get("name", ""),
                    row.get("note", ""), *(row.get("tags") or [])])
    return any(_clause(hay, clause)
               for clause in re.split(r"\s+\|\s+", expr))


def _clause(hay: str, clause: str) -> bool:
    terms = clause.split()
    if not terms:
        return False
    return all(_term(hay, term) for term in terms)


def _term(hay: str, term: str) -> bool:
    try:
        return re.search(term, hay, re.IGNORECASE) is not None
    except re.error:
        return term.lower() in hay.lower()


def _entry_value(entry: dict, metric: str) -> float | None:
    for block in ("train", "post"):
        value = entry.get(block, {}).get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def metric_names(roots: Sequence[Store | Root]) -> list[str]:
    """Every metric the newest ledgers carry, reward first: the union of
    numeric keys under train and post, read from a sample of runs — a name
    list is a menu, not a census."""
    names: set[str] = set()
    rows = runs_data(rooted(roots))
    for row in rows[-NAME_SAMPLE:]:
        root = _root_of(roots, row)
        if root is None:
            continue
        for entry in root.store.peek_ledger(row["run_id"])[-3:]:
            for block in ("train", "post"):
                names.update(k for k, v in entry.get(block, {}).items()
                             if isinstance(v, (int, float))
                             and not isinstance(v, bool))
    front = [n for n in ("reward", "plora_kl", "loss", "logprob_gap") if n in names]
    return front + sorted(names - set(front))


def overlay(roots: Sequence[Store | Root], metric: str,
            expr: str = "") -> dict:
    """The wandb reading: one series per selected run, x = update, newest
    runs first, capped at SERIES_LIMIT with the cap said out loud."""
    rows = [row for row in runs_data(rooted(roots)) if match_expr(row, expr)]
    rows.reverse()                                  # newest first
    dropped = max(0, len(rows) - SERIES_LIMIT)
    series = []
    for row in rows[:SERIES_LIMIT]:
        root = _root_of(roots, row)
        if root is None:
            continue
        points = []
        for entry in root.store.peek_ledger(row["run_id"]):
            value = _entry_value(entry, metric)
            if value is not None and isinstance(entry.get("update"), int):
                points.append([entry["update"], value])
        if points:
            series.append({"run_id": row["run_id"], "name": row.get("name", ""),
                           "folder": row["folder"], "status": row["status"],
                           "tags": row.get("tags") or [], "points": points})
    return {"metric": metric, "expr": expr, "series": series,
            "matched": len(rows), "dropped": dropped}


def _root_of(roots: Sequence[Store | Root], row: dict) -> Root | None:
    for root in rooted(roots):
        if root.folder == row["folder"]:
            return root
    return None
