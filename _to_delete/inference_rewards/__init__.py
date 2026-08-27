"""Rewards: one class per file, all inheriting Reward (base.py).

The contract: `async score(rollout, llm) -> tuple[float, ...]`, one value per
declared component. Importing this package registers the builtins.
"""

from rlstack.inference.rewards.base import Reward, RewardDef, reward  # noqa: F401
from rlstack.inference.rewards import constant, verifier  # noqa: F401
