"""A tool episode: the asker, reading its context, may ask a memory (a
library adapter set) closed-book questions before it writes its final turn.

Each asker turn is sampled under the task's `route` (the asker ALONE: it
never carries the memory). A turn that ends in a well-formed `ask_memory` call
gets an answer: the question alone, framed by the task's `ask_open`/`ask_close`
(thinking off), is sampled under `tool_route` — `lib:<memory>`, the memory with
nothing in context — and comes back inside the template's tool-response text,
an INJECTED message (loss-masked 0, never trained). The memory's turn is not a
turn of this rollout: it is recorded in `env_extras["asks"]` (with the index
of the asker turn that asked it), so the asker's row trains on its own
tokens only. Any turn without a call is the final turn; what it holds is the
reward's to read.

PARALLEL CALLS. A turn may make several calls; every one is answered, in the
order written, each counting toward `max_asks`, the memory's answers sampled
concurrently and shown in ONE tool turn joined by the template's
`answer_between`. Calls past the cap: the episode ends at that turn
(`past_cap: "end"`), or, under `past_cap: "refuse_once"`, the calls
within the cap are answered, each one past it gets `REFUSAL` in its place, and
the asker gets one more turn, which is final whatever it holds; refused
calls are counted in `env_extras["refused"]`, never asks. A second turn past
the cap ends the episode. A task whose pieces carry no `answer_between` (built
before parallel calls) has each turn's first call answered and the rest
counted in `env_extras["dropped"]`.

ONE TEXT, TWO TOKEN STREAMS. The engine tokenizes each message's text, while
flatten replays a turn's own token ids. A turn closed by the end-of-turn token
carries that token in its ids but not in its text, so the context of the next
call is given the turn's text plus `turn_end`, and the injected answer then
opens without it; a turn whose ids lack it gets `turn_end` at the head of the
injected answer instead. Either way the sampled context and the replayed row
read the same tokens.

The episode inherits the run's sampling, including for the memory's answer;
the ask frame stops the answer at its first blank line, `answer_chars` bounds
what the asker is shown, and `answer_tokens` (when the task sets it) is the
answer's token ceiling — a memory that never writes a blank line otherwise
generates a whole turn's budget while its episode waits.
"""

from __future__ import annotations

import asyncio
import re

from rlstack.data.trajectory import Message, Role, Task, Turn
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout
from rlstack.policy.adapters.dream_bank import Route

ANSWER_STOP = ("\n\n",)
REFUSAL = "You have used all your questions. Write your study texts now."
PAST_CAP = ("end", "refuse_once")
THINK_END = "</think>"
TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=ask_memory>\s*<parameter=question>\s*(.*?)\s*</parameter>"
    r"\s*</function>\s*</tool_call>", re.DOTALL)


def asks_of(turn_text: str) -> tuple[str, ...]:
    """The questions of every well-formed `ask_memory` call in a turn's text
    — after its thinking — in the order written (a turn may make parallel
    calls); empty when it makes none."""
    _, closed, after = turn_text.rpartition(THINK_END)
    return tuple(match.group(1).strip() for match in TOOL_CALL.finditer(after if closed else ""))


def closed_by_turn_end(turn: Turn, turn_end_id: int) -> bool:
    """The turn's own ids end in the end-of-turn token."""
    return bool(turn.token_ids) and turn.token_ids[-1] == turn_end_id


def tool_turn(pieces: dict, shown: list[str], closed: bool) -> str:
    """The injected text after an asker turn that called: every response in
    one tool turn, in call order, then the reopened assistant turn."""
    return (("" if closed else pieces["turn_end"]) + pieces["answer_open"]
            + str(pieces.get("answer_between", "")).join(shown) + pieces["answer_close"])


@environment("ask_memory")
class AskMemory(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        meta = task.meta
        pieces = meta["pieces"]
        asker = (Route(str(meta["route"])),)
        memory = (Route(str(meta["tool_route"])),)
        prompt = Message(Role.USER, task.prompt)
        past_cap = str(meta.get("past_cap", "end"))
        if past_cap not in PAST_CAP:
            raise ValueError(f"past_cap {past_cap!r} is not one of {PAST_CAP}")
        parallel = "answer_between" in pieces
        rollout = Rollout(task=task, messages=[prompt],
                          env_extras={"asks": [], "refused": 0, "dropped": 0})
        extras = rollout.env_extras
        context = [prompt]
        refusing = False
        while True:
            turn = await client.sample(context, directives=asker)
            rollout.messages.append(turn.message)
            rollout.turns.append(turn)
            questions = asks_of(turn.message.content)
            if questions and not parallel:
                extras["dropped"] += len(questions) - 1
                questions = questions[:1]
            room = max(0, int(meta["max_asks"]) - len(extras["asks"]))
            within, past = questions[:room], questions[room:]
            ends = (not questions or turn.finish != "eos" or refusing
                    or (past and (past_cap == "end" or extras["refused"])))
            if ends:
                extras["final"] = {"finish": turn.finish, "asked_past_cap": bool(questions)}
                return rollout
            index = len(rollout.turns) - 1

            async def answer(question: str) -> Turn:
                asked = Message(Role.USER, f"{pieces['ask_open']}{question}{pieces['ask_close']}")
                if "answer_tokens" in meta:
                    return await client.sample([asked], ANSWER_STOP, directives=memory,
                                               max_tokens=int(meta["answer_tokens"]))
                return await client.sample([asked], ANSWER_STOP, directives=memory)

            # one coroutine per call, created in call order: each draws its
            # seed before its first await, so the draws do not depend on
            # which answer finishes first
            answers = await asyncio.gather(*(answer(q) for q in within))
            shown = []
            for question, reply in zip(within, answers):
                text = reply.message.content.strip()[:int(meta["answer_chars"])]
                shown.append(text)
                extras["asks"].append({
                    "question": question, "answer": reply.message.content, "shown": text,
                    "finish": reply.finish, "answer_tokens": len(reply.token_ids),
                    "turn": index})
            if past:
                extras["refused"] += len(past)
                shown += [REFUSAL] * len(past)
                refusing = True
            closed = closed_by_turn_end(turn, int(pieces["turn_end_id"]))
            injected = Message(Role.USER, tool_turn(pieces, shown, closed))
            rollout.messages.append(injected)
            context = context + [
                Message(turn.message.role,
                        turn.message.content + (pieces["turn_end"] if closed else "")),
                injected]
