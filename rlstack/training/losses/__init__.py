"""The loss zoo: one registered function per objective, one file each (mirrors
training/post/). Importing this package registers the builtins."""

from rlstack.training.losses.base import (  # noqa: F401
    LossResult, PolicyOutputs, rails, token_tensors,
)
from rlstack.training.losses import (  # noqa: F401  (registers builtins)
    grpo, grpo_elbo, grpo_latent_kl, grpo_latent_kl_gated,
    grpo_latent_kl_gated_priced, grpo_reference_kl,
    gspo, opd, opsd, ppo, replay_distill, sdft,
    sdpo, self_anchor, sft, factual_sft,
)
