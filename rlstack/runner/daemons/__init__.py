"""Daemons: one per GPU responsibility, synchronized only via the store.

    base.py      — Daemon: the shape (await condition → lease → work →
                   write store → notify)
    generator.py — writes rollouts/<w> at the newest bundle, throttled by the
                   lag buffer
    trainer.py   — awaits rollouts/<u>, runs post + fwd/bwd + commit,
                   publishes the bundle
    evaluator.py — awaits ledger commits on the eval modulus, writes eval/<u>

plan_daemons (runner/loop.py) derives the set from the spec: live rollouts →
a Generator exists; eval declared → an Evaluator exists; the Trainer always.
The daemons' condition methods (may_generate / next_rows / due_updates)
are the overridable seam for custom alternation policies.
"""

from rlstack.runner.daemons.base import Daemon
from rlstack.runner.daemons.generator import Generator
from rlstack.runner.daemons.trainer import Trainer
from rlstack.runner.daemons.evaluator import Evaluator
