"""A declared environment that never runs: a name for specs under test."""

from __future__ import annotations

from rlstack.data.trajectory import Task
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout


@environment("noop_env")
class NoopEnv(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        raise NotImplementedError
