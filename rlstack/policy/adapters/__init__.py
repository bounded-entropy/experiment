"""Adapter types: one registered class per file, all inheriting AdapterType (base.py).

A file here is one intervention the policy can carry, with both of its
lowerings beside it. Importing this package registers the builtins.
"""

from rlstack.policy.adapters.base import (  # noqa: F401
    AdapterType, AdapterTypeDef, Mechanism, adapter_type,
)
from rlstack.policy.adapters import (  # noqa: F401  (registers builtins)
    attn_bias, lora, plora, soft_prompt, spectral, value_head,
)
