"""Postprocessing: one class per file, all inheriting PostProcessor (base.py).

The contract: `async process(group, data, llm) -> {column: vector}` — per
group, after the seal, before the loss. Importing this package registers the
builtins.
"""

from rlstack.training.post.base import PostDef, PostProcessor, postprocessor  # noqa: F401
from rlstack.training.post import center_reward, constant, grpo_advantage, verifier  # noqa: F401
