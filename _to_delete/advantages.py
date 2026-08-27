"""Built-in advantages (training world).

An advantage runs once per wave AFTER the seal: wave-scope, detached, CPU-pure
over sealed store data, offline re-runnable. It returns one weight per
trajectory, aligned to the wave's trajectory order (the concatenation of its
groups). The Group primitive carries exactly the scope a partial loss
contribution needs, so a group-relative estimator is a loop over wave.groups —
no parallel id arrays.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from rlstack.data.trajectory import Wave
from rlstack.registry import advantage


def zscore(values: Sequence[float]) -> list[float]:
    """(x − mean) / population std; an all-equal list normalizes to zeros."""
    n = len(values)
    mean = math.fsum(values) / n
    std = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / n)
    return [0.0 if std == 0.0 else (x - mean) / std for x in values]


@advantage("grpo_group_norm", consumes=("reward",))
def grpo_group_norm(wave: Wave, ctx: Any = None) -> list[float]:
    """The GRPO estimator: normalize each trajectory's reward within its group."""
    return [weight
            for group in wave.groups
            for weight in zscore(group.components("reward"))]
