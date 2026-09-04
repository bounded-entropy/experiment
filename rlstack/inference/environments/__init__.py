"""Environments: one class per file, all inheriting Environment (base.py).

The contract is `async run(client, task) -> Rollout`; a name exists iff the module
defining it was imported, and importing this package registers the builtins.
"""

from rlstack.inference.environments.base import (  # noqa: F401
    Environment, EnvironmentDef, PoolClient, environment,
)
from rlstack.inference.environments import (  # noqa: F401  (registers builtins)
    conditioned_teacher, dapo_math, math_single_turn, noop, single_turn,
)
