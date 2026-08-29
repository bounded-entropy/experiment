"""Building a TASK SET: the generic verbs, dataset-blind.

A task set is pure content — `{id, prompt, meta}` rows, one jsonl line each,
put in the store's CAS so the sha IS the set's identity (I3) and a spec pins it
by uri. A plan's leaf names a task by ID ALONE, so everything about WHICH task
runs WHERE lives in the plan and everything about WHAT a task says lives here.

Nothing in this file knows what a dataset is; one file per dataset beside it
(`dapo_math.py`) turns raw rows into Tasks. Both halves are semantics — the
prompt, the filter, the split rule all hash into a run — so both live under
`data/` and never in `deploy/` (I5).
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Mapping, Sequence

from rlstack.data.stores.base import Store
from rlstack.data.trajectory import Task


def write_tasks(store: Store, tasks: Sequence[Task]) -> str:
    """Put a task set in the CAS and return its uri — `load_tasks`' inverse.

    Canonical bytes: one json object per line, keys sorted, in the order given.
    The sha of those bytes IS the set's identity, so rebuilding the same tasks
    on another machine returns the same uri and stores no second copy.

    Ids must be unique WITHIN the set, for the reason `load_task_sets` refuses
    them across sets: a leaf carries an id alone, and a repeated id makes that
    leaf ambiguous. Refused where the set is made, not where it is read.
    """
    counts = Counter(task.id for task in tasks)
    duplicates = sorted(task_id for task_id, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(
            f"task ids repeat within one set: {duplicates[:5]} "
            f"({len(duplicates)} in all); a plan's leaf names an id alone")
    rows = [{"id": t.id, "prompt": t.prompt, "meta": dict(t.meta)} for t in tasks]
    data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    return store.cas_put(data.encode("utf-8"))


def load_tasks(store: Store, uri: str) -> list[Task]:
    """Materialize a cas://-addressed jsonl of {id, prompt, meta} rows."""
    rows = [json.loads(line) for line in
            store.cas_get(uri).decode("utf-8").splitlines() if line]
    return [Task(id=r["id"], prompt=r["prompt"], meta=r.get("meta", {})) for r in rows]


def split_tasks(tasks: Sequence[Task], fractions: Mapping[str, float],
                seed: int) -> dict[str, list[Task]]:
    """Deterministic named splits: every task lands in exactly one of them.

    A task's split is a function of `(seed, its id)` ALONE — not of its
    position, not of the set it arrived in — so a task never moves when the
    source grows: re-splitting a superset leaves every earlier task where it
    was, and held-out stays held out across dataset versions. The price, paid
    knowingly, is that counts are DRAWN rather than dealt: each split gets its
    fraction in expectation, within the wobble of n draws.

    The fractions must sum to 1 (a task belongs to exactly one split, so the
    splits have to cover the set), declaration order fixes which interval is
    whose, and order inside a split is input order.
    """
    if any(f < 0 for f in fractions.values()):
        raise ValueError(f"negative fraction in {dict(fractions)}")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"fractions must sum to 1 — every task belongs to exactly one "
            f"split — got {dict(fractions)} summing to {sum(fractions.values())}")
    names = list(fractions)
    edges = list(itertools.accumulate(fractions[name] for name in names))
    edges[-1] = 1.0            # the top edge is exact; a float sum is not
    out: dict[str, list[Task]] = {name: [] for name in names}
    for task in tasks:
        out[names[bisect.bisect_right(edges, draw_for(seed, task.id))]].append(task)
    return out


def draw_for(seed: int, task_id: str) -> float:
    """Where this task falls in [0, 1): `h(seed, id)`, the seed tree's rule
    (`runner.seeds.derive`) applied to a task rather than to a call site."""
    material = json.dumps([seed, task_id], separators=(",", ":"))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64
