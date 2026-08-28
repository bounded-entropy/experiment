"""The training world: post/ (postprocessors) and losses/. Sealed data only.

A postprocessor computes everything ABOUT sealed trajectories — it may sample
or score through a pool, but what it computes lands in postdata beside the
waves and never inside the sealed record (I6). A loss is pure math over the
named columns postdata and the record already hold (I9). Must never import
rlstack.inference (enforced by tests/test_architecture.py).
"""

from rlstack.training import post, losses  # noqa: F401  (registers builtins)
