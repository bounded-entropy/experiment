"""The inference world: the Rollout and the environments that build one.

Everything here runs before the seal and may sample. Rewards, judges and
advantages are postprocessors (training/post/), never environments. Must never
import rlstack.training (enforced by tests/test_architecture.py).
"""

from rlstack.inference import rollout  # noqa: F401
from rlstack.inference import environments  # noqa: F401  (registers builtins)
