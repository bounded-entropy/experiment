"""Shared scaffolding for the metal probes (deploy/*_l4.py and friends).

Fixtures only — nothing semantics-bearing lives in deploy/ (I5). Every probe
needs the same two helpers, and each used to paste its own copy (review
finding, post-#47): the toy arithmetic dataset, and a PASS/FAIL line that
tallies into CHECKS. One definition each, shipped into every app's image the
same way the packages are (add_local_python_source("probe", ...)).
"""

from __future__ import annotations

import json
import random

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """One PASS/FAIL line, tallied in CHECKS (import CHECKS to summarize)."""
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""))


def arith_tasks(n: int, seed: int) -> bytes:
    """Two-digit sums phrased for raw completion: the continuation after
    "The answer is" is where the verifier finds its last number."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randrange(10, 99), rng.randrange(10, 99)
        rows.append({"id": f"arith-{i:04d}",
                     "prompt": f"What is {a}+{b}? The answer is",
                     "meta": {"answer": a + b}})
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
