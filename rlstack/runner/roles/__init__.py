"""Runners: one per GPU responsibility, synchronized only via the store.

    base.py      — Runner: the shape (await condition → admit → work →
                   write store → notify)
    generator.py — writes waves/<w> at the newest committed bundle, bounded
                   by the lag buffer
    scorer.py    — runs the POOLED half of the post pipeline beside the pools
                   it addresses, writing postdata/<u>.scorer.json
    trainer.py   — awaits waves/<u> and the scorer's part, runs the inline
                   half + fwd/bwd + commit, publishes the bundle
    evaluator.py — awaits ledger commits on the eval modulus, writes eval/<u>
    fitter.py    — a fit run's only runner (ADR 0019): fits its plan's jobs
                   on the lanes of a dream_bank entry, writes each as a
                   NAMED adapter, one ledger line per job

plan_runners (runner/loop.py) derives the set from the spec. Each runner's
condition method is the named, overridable seam for a custom alternation
policy.
"""

from rlstack.runner.roles.base import Runner
from rlstack.runner.roles.fitter import Fitter
from rlstack.runner.roles.generator import Generator
from rlstack.runner.roles.scorer import SCORER, Scorer
from rlstack.runner.roles.trainer import Trainer
