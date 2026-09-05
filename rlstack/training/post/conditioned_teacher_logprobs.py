"""The CONDITIONED teacher channel: a frozen other model, under a hint the
sampler never saw, scores the sealed tokens (ADR 0005).

`teacher_logprobs` scores through the "teacher" pool under the same
conditioning the student had; `hinted_logprobs` prepends the hint but scores
through "main". This is the one walk with both — the teacher's metal AND the
privileged conditioning — which is what "distill the student toward the
prompt-conditioned model" needs as a number.

THE COLUMN NAME IS `teacher_logprobs`, deliberately the same one. A different
conditioning is a different PROCESSOR, never an edit to a loss (#47's rule,
applied to the hint): `opd` requires the column unchanged, and swapping this
processor for `teacher_logprobs` in a pipeline changes what the teacher was
told and nothing else. Only one of the two may be in a pipeline at a time —
every column has exactly one owner, checked at Phase 0.

The hint is the task's own, by `data.trajectory.hint_for`'s one convention.
Both of `teacher_logprobs`' preconditions ride along: the teacher must share
the student's tokenizer (the ids crossing the wire are the student's draws),
and it is FROZEN — a non-policy pool serves its bare base, and scoring is
seedless, so adding this processor never shifts the run's sampling.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group, Message, Trajectory, hint_for
from rlstack.training.post.base import PostProcessor, postprocessor


async def conditioned_teacher_scores(traj: Trajectory,
                                     client: PoolClient) -> list[float]:
    """Walk the sealed message stream in flatten order: each generated turn's
    token_ids scored against hint + everything before it — `teacher_scores`'
    walk with the hint at the head of the context, which is the conditioning
    the teacher had when it wrote these tokens and the student did not."""
    turn_of = {id(t.message): t for t in traj.turns}
    context: list[Message] = [hint_for(traj.task)]
    scores: list[float] = []
    for message in traj.messages:
        turn = turn_of.get(id(message))
        if turn is not None:
            scores.extend(await client.score(context, turn.token_ids))
        context.append(message)
    return scores


@postprocessor("conditioned_teacher_logprobs")
class ConditionedTeacherLogprobs(PostProcessor):
    produces = ("teacher_logprobs",)
    token_level = ("teacher_logprobs",)
    pools = ("teacher",)              # another model's metal, declared at submit

    async def process(self, group: Group, data, client: PoolClient
                      ) -> Mapping[str, Sequence]:
        teacher = client.pool("teacher")
        # every trajectory of the group at once: the walk inside one is
        # serial (its context grows), the group is not
        return {"teacher_logprobs": list(await asyncio.gather(
            *(conditioned_teacher_scores(traj, teacher)
              for traj in group.trajectories)))}
