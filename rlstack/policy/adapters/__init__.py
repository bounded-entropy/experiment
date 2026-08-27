"""Adapters: one class per file, all inheriting Adapter (base.py).

The five-member protocol lives on the base; each file here is one intervention
the policy can carry. Importing this package registers the builtins.
"""

from rlstack.policy.adapters.base import (  # noqa: F401
    Adapter, AdapterDef, Mechanism, adapter,
)
from rlstack.policy.adapters import attn_bias, lora, soft_prompt, value_head  # noqa: F401
