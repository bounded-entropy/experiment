"""The teacher channel: a FROZEN OTHER MODEL scores the student's own tokens.

On-policy distillation's one non-student ingredient (#47). The student samples
live; for each sealed trajectory this processor replays the message stream and
asks the "teacher" pool for the logprobs of the tokens the student actually
chose, in one prefill pass per generated turn; the per-token result lands as a
token_level column that the opd loss requires. I9 end to end: anything needing
a GPU is post's job, so the teacher is a POOL — never a pass the loss plans —
and switching teachers means pointing that pool at other metal.

Two preconditions, both real:
  - the teacher shares the student's TOKENIZER. Token ids cross unchanged
    (they are the student's draws), so a teacher with another vocabulary would
    score different text than the student wrote. Same-family pairs
    (Qwen3-8B ← Qwen3-32B) satisfy it; the check belongs at the deploy, which
    is what hands the two pools their metal.
  - the teacher is FROZEN. The pool serves the bare base (a non-policy pool
    gets a payload-free bundle, so no delta of this run is ever applied) and
    scoring is deterministic and seedless — adding this processor never
    shifts the run's sampling seeds.

The walk mirrors hinted_logprobs exactly, minus the hint: flatten order, each
generated turn scored against everything before it, so the column's floats
land token-for-token on the same positions the loss mask selects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group, Message, Trajectory
from rlstack.training.post.base import PostProcessor, postprocessor


async def teacher_scores(traj: Trajectory, llm: PoolClient) -> list[float]:
    """Walk the sealed message stream in flatten order: each generated turn's
    token_ids scored against everything before it — the SAME conditioning the
    student had, which is what makes the difference a pure model difference."""
    turn_of = {id(t.message): t for t in traj.turns}
    context: list[Message] = []
    scores: list[float] = []
    for message in traj.messages:
        turn = turn_of.get(id(message))
        if turn is not None:
            scores.extend(await llm.score(context, turn.token_ids))
        context.append(message)
    return scores


@postprocessor("teacher_logprobs")
class TeacherLogprobs(PostProcessor):
    produces = ("teacher_logprobs",)
    token_level = ("teacher_logprobs",)
    pools = ("teacher",)              # another model's metal, declared at submit

    async def process(self, group: Group, data, llm: PoolClient
                      ) -> Mapping[str, Sequence]:
        teacher = llm.pool("teacher")
        return {"teacher_logprobs": [await teacher_scores(traj, teacher)
                                     for traj in group.trajectories]}
