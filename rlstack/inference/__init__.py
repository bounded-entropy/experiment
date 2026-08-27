"""The inference world: rollouts and environments. May sample; runs before
the seal. Rewards live in training/post/ (decision #24).

Must never import rlstack.training (enforced by tests/test_architecture.py).
"""

from rlstack.inference import rollout  # noqa: F401
from rlstack.inference import environments  # noqa: F401  (registers builtins)
