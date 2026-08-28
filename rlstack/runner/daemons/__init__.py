"""Daemons: one per GPU responsibility, synchronized only via the store.

    base.py      — Daemon: the shape (await condition → admit → work →
                   write store → notify)
    generator.py — writes waves/<w> at the newest committed bundle, bounded
                   by the lag buffer
    trainer.py   — awaits waves/<u>, runs post + fwd/bwd + commit, publishes
                   the bundle
    evaluator.py — awaits ledger commits on the eval modulus, writes eval/<u>

plan_daemons (runner/loop.py) derives the set from the spec. Each daemon's
condition method is the named, overridable seam for a custom alternation
policy.
"""

from rlstack.runner.daemons.base import Daemon
from rlstack.runner.daemons.generator import Generator
from rlstack.runner.daemons.trainer import Trainer
from rlstack.runner.daemons.evaluator import Evaluator
