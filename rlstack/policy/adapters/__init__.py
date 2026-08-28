"""Adapter kinds: one registered class per file, all inheriting Adapter (base.py).

A file here is one intervention the policy can carry, with both of its
lowerings beside it. Importing this package registers the builtins.
"""

from rlstack.policy.adapters.base import (  # noqa: F401
    Adapter, AdapterDef, Mechanism, adapter,
)
from rlstack.policy.adapters import attn_bias, lora, soft_prompt, value_head  # noqa: F401
