"""Postprocessors: one class per file, all inheriting PostProcessor (base.py).

Each declares produces / consumes / token_level / pools / sampling and
implements `async process(group, data, client) -> {column: vector}` — per group,
after the seal, before the loss. Importing this package registers the builtins.
"""

from rlstack.training.post.base import PostDef, PostProcessor, postprocessor  # noqa: F401
from rlstack.training.post import (  # noqa: F401  (registers builtins)
    boxed_verifier, center_reward, constant, grpo_advantage, hinted_logprobs,
    llm_judge, teacher_logprobs, verifier,
)
