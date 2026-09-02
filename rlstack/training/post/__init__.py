"""Postprocessors: one class per file, all inheriting PostProcessor (base.py).

Each declares produces / consumes / token_level / pools / sampling and
implements `async process(group, data, client) -> {column: vector}` — per group,
after the seal, before the loss. Importing this package registers the builtins.
"""

from rlstack.training.post.base import PostDef, PostProcessor, postprocessor  # noqa: F401
from rlstack.training.post import (  # noqa: F401  (registers builtins)
    center_reward, constant, grpo_advantage, group_accuracy,
    final_answer, glyph_grade, hinted_logprobs, llm_judge, stamp_grade,
    teacher_logprobs, verifier,
)
