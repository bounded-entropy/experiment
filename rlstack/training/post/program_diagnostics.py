"""Measurement-only descriptions of program rollouts and reward saturation."""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor
from rlstack.training.post.program_correctness import program_text


def program_fingerprint(text: str, domain: str) -> str:
    """Ignore Python comments/formatting; SQL comparison ignores case/spacing."""
    code = program_text(text)
    if domain == "mbpp":
        try:
            return ast.dump(ast.parse(code))
        except SyntaxError:
            return code
    return " ".join(code.casefold().split())


@postprocessor("program_diagnostics")
class ProgramDiagnostics(PostProcessor):
    consumes = ("reward",)
    produces = ("completion_tokens", "truncated", "sampled_surprisal",
                "distinct_program_fraction", "distinct_correct_fraction",
                "all_pass_group", "all_fail_group", "mixed_reward_group")

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        turns = [traj.turns[-1] for traj in group.trajectories]
        rewards = data["reward"]
        fingerprints = [program_fingerprint(turn.message.content, traj.task.meta["domain"])
                        for turn, traj in zip(turns, group.trajectories)]
        count = len(turns)
        correct = {fingerprint for fingerprint, reward in zip(fingerprints, rewards)
                   if reward == 1.0}
        all_pass = float(all(reward == 1.0 for reward in rewards))
        all_fail = float(all(reward == 0.0 for reward in rewards))
        return {
            "completion_tokens": [float(len(turn.token_ids)) for turn in turns],
            "truncated": [float(turn.finish == "length") for turn in turns],
            "sampled_surprisal": [-math.fsum(turn.behavior_logprobs) / max(1, len(turn.token_ids))
                                   for turn in turns],
            "distinct_program_fraction": [len(set(fingerprints)) / count] * count,
            "distinct_correct_fraction": [len(correct) / count] * count,
            "all_pass_group": [all_pass] * count,
            "all_fail_group": [all_fail] * count,
            "mixed_reward_group": [1.0 - all_pass - all_fail] * count,
        }
