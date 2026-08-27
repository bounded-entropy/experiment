"""Built-in environments (inference world).

An env is an async coroutine driving `sample` calls against a resident engine
pool; thousands run concurrently and continuous batching absorbs their latency.
Envs may sample freely — the stage rule routes anything that samples here.
An env receives a SampleClient (`await llm.sample(messages) -> Turn`) and a
Task, and returns the finished (unsealed) Trajectory; the runner runs rewards
and seals.
"""

from __future__ import annotations

from typing import Any

from rlstack.data.trajectory import Message, Role, Task, Trajectory
from rlstack.registry import env


@env("math_single_turn")
async def math_single_turn(llm: Any, task: Task) -> Trajectory:
    """One sample call, one turn, done (SPEC.md Example 4)."""
    prompt = Message(Role.USER, task.prompt)
    turn = await llm.sample([prompt])
    return Trajectory(task=task, messages=[prompt, turn.message], turns=[turn])


@env("noop_env")
async def noop_env(llm: Any, task: Any) -> Any:
    """One-shot env stub (declaration only)."""
    raise NotImplementedError
