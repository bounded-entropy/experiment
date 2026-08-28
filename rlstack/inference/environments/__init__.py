"""Environments: one class per file, all inheriting Environment (base.py).

The contract is `async run(llm, task) -> Rollout`; a name exists iff the module
defining it was imported, and importing this package registers the builtins.
"""

from rlstack.inference.environments.base import (  # noqa: F401
    Environment, EnvironmentDef, PoolClient, environment,
)
from rlstack.inference.environments import math_single_turn, noop  # noqa: F401
