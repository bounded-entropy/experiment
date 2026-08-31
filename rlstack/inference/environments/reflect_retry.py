"""One reflect episode: critique, then retry — the iterative-SDPO protocol.

The derived prompt (a reflect maker's output: the whole transcript so far
plus "state what was wrong") gets TWO turns. Turn one is the model's critique
of its own attempt. Then the environment injects the retry request — an
injected message, loss-masked 0 like any env text — and turn two is the
corrected attempt. The graders read `turns[-1]`, so a retry is judged exactly
as a first attempt is; the sdpo loss clones the FINAL turn only, so the
critique conditions the retry without being trained as an answer.

Chat delimiters ride in task.meta["chat"] (the maker's convention, stated
there): the injected retry ask must speak the same template the prompt was
built with, and the plain-text fallback serves unformatted sets."""

from __future__ import annotations

from rlstack.data.trajectory import Message, Role, Task
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.makers.reflect import chat_pieces
from rlstack.inference.rollout import Rollout

RETRY = ("Now write your corrected answer to the original request, "
         "and nothing else.")


@environment("reflect_retry")
class ReflectRetry(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        critique = await client.sample([prompt])
        pieces = chat_pieces(task)
        ask = Message(Role.USER,
                      f"{pieces['assistant_end']}{pieces['user_open']}{RETRY}"
                      f"{pieces['user_close']}{pieces['assistant_open']}")
        retry = await client.sample([prompt, critique.message, ask])
        return Rollout(task=task,
                       messages=[prompt, critique.message, ask, retry.message],
                       turns=[critique, retry])
