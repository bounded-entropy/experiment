"""The inference world: rollouts, environments, rewards. May sample; runs
before the seal.

Must never import rlstack.training (enforced by tests/test_architecture.py).
"""

from rlstack.inference import rollout  # noqa: F401
from rlstack.inference import environments, rewards  # noqa: F401  (registers builtins)
