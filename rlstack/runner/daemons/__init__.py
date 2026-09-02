"""Daemons: one per GPU responsibility, synchronized only via the store.

    base.py      — Daemon: the shape (await condition → admit → work →
                   write store → notify)
    generator.py — writes waves/<w> at the newest committed bundle, bounded
                   by the lag buffer
    scorer.py    — runs the POOLED half of the post pipeline beside the pools
                   it addresses, writing postdata/<u>.scorer.json
    trainer.py   — awaits waves/<u> and the scorer's part, runs the inline
                   half + fwd/bwd + commit, publishes the bundle
    evaluator.py — awaits ledger commits on the eval modulus, writes eval/<u>

plan_daemons (runner/loop.py) derives the set from the spec. Each daemon's
condition method is the named, overridable seam for a custom alternation
policy.
"""

from rlstack.runner.daemons.base import Daemon
from rlstack.runner.daemons.generator import Generator
from rlstack.runner.daemons.scorer import SCORER, Scorer
from rlstack.runner.daemons.trainer import Trainer
