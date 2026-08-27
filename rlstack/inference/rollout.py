"""The Rollout: one episode in progress — inference-world, mutable.

ROLLOUT is inference terminology; TRAJECTORY is training terminology. An
Environment builds a Rollout by driving sample calls; rewards fill its
reward_components; then the runner calls .seal() and the episode crosses the
membrane (I1) as a frozen Trajectory. Nothing on the training side ever sees a
Rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from rlstack.data.trajectory import Message, Task, Trajectory, Turn


@dataclass
class Rollout:
    """One episode being generated. Freely mutable until sealed.

    The environment appends to `messages` and `turns` (a Turn's message object
    appears in BOTH — that identity is how flatten later tells generated tokens
    from injected ones). `env_extras` is the environment's open notebook (tool
    logs, intermediate hints). Scores about the episode are NOT written here —
    rewards are postprocessing, after the seal.
    """

    task: Task
    messages: list[Message] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    env_extras: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concatenated content of all messages (convenience for rewards)."""
        return "".join(m.content for m in self.messages)

    def seal(self) -> Trajectory:
        """Cross the membrane: the episode is finished → frozen training data."""
        return Trajectory(
            task=self.task,
            messages=tuple(self.messages),
            turns=tuple(self.turns),
            env_extras=MappingProxyType(dict(self.env_extras)),
        )
