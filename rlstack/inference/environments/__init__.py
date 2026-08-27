"""Environments: one class per file, all inheriting Environment (base.py).

The contract: `async run(llm, task) -> Rollout`. Importing this package
registers the builtins.
"""

from rlstack.inference.environments.base import (  # noqa: F401
    Environment, EnvironmentDef, PoolClient, environment,
)
from rlstack.inference.environments import math_single_turn, noop  # noqa: F401
