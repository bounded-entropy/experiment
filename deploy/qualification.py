"""Synthetic behavior owned by the infrastructure qualification demos.

These rows exercise scheduling, sampling and training services. They are not
a research benchmark and are not screened for a model's answer distribution.
"""

import hashlib
import json
import re

from rlstack.data.trajectory import Message, Role
from rlstack.inference.rollout import Rollout
from rlstack.training.post.base import PostProcessor, postprocessor


def task_bytes(split: str) -> bytes:
    rows = [{"id": f"qualification/{split}-{index}",
             "prompt": f"What is {17 + index}+{26 + index}? Give the numeric answer.",
             "meta": {"answer": 43 + 2 * index}}
            for index in range(4)]
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


TRAIN_ROWS = task_bytes("train")
EVAL_ROWS = task_bytes("eval")
TRAIN_TASKS = "cas://" + hashlib.sha256(TRAIN_ROWS).hexdigest()
EVAL_TASKS = "cas://" + hashlib.sha256(EVAL_ROWS).hexdigest()
QUALIFICATION_TASK = "qualification/train-0"
FUNCTION = "deploy.qualification:interact"
CODE = ("deploy.qualification",)


def write_tasks(store) -> None:
    store.cas_put(TRAIN_ROWS)
    store.cas_put(EVAL_ROWS)


async def interact(client, task):
    prompt = Message(Role.USER, task.prompt)
    sampled = await client.sample([prompt])
    return Rollout(task=task, messages=[prompt, sampled.message], turns=[sampled])


@postprocessor("qualification_reward")
class NumericAgreement(PostProcessor):
    produces = ("reward",)

    async def process(self, group, data, client):
        rewards = []
        for trajectory in group.trajectories:
            numbers = re.findall(r"-?\d+", trajectory.turns[-1].message.content)
            rewards.append(float(bool(numbers) and
                                 numbers[-1] == str(trajectory.task.meta["answer"])))
        return {"reward": rewards}
