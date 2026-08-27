"""The training world: postprocessing and losses. Sees sealed data only.

Postprocessors may sample (an LLM judge scores sealed trajectories through the
neutral SampleClient interface) but never mutate the sealed record — what they
compute lives in postdata, beside the sealed waves, never inside them.

Must never import rlstack.inference (enforced by tests/test_architecture.py).
"""

from rlstack.training import post, losses  # noqa: F401  (registers builtins)
