"""Losses: one registered function per objective, one file each (mirrors
training/post/). Importing this package registers the builtins."""

from rlstack.training.losses.base import (  # noqa: F401
    LossResult, PolicyOutputs, rails, token_tensors,
)
from rlstack.training.losses import (  # noqa: F401  (registers builtins)
    gspo, grpo, opd, ppo, sdft, self_anchor, sft,
)
