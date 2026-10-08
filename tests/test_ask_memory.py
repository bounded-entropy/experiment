"""The `ask_memory` tool episode on fakes: which calls parse, what the asker
is shown, what its row trains on, and how calls past the cap end or are
refused."""

from __future__ import annotations

import asyncio
import unittest

from common import char_tokenize, make_turn
from rlstack import Task
from rlstack.data.plan import GroupPlan, Sample, WavePlan
from rlstack.inference.environments.ask_memory import REFUSAL, asks_of
from rlstack.registry import ENVS
from rlstack.runner.names import wave_lib_names

MEMORY = "mem/0001"
END = "<E>"
END_ID = 9999
PIECES = {"answer_open": "\n<U><tool_response>\n", "answer_close": "\n</tool_response><E>\n<A><think>\n",
          "ask_open": "<U>", "ask_close": "<E>\n<A><think>\n\n</think>\n\n",
          "turn_end": END, "turn_end_id": END_ID}


def go(coro):
    return asyncio.run(coro)


def task(*, max_asks: int = 3, parallel: bool = False, answer_tokens: int | None = None,
         past_cap: str | None = None) -> Task:
    pieces = dict(PIECES, answer_between="<BETWEEN>") if parallel else dict(PIECES)
    meta = {"route": "asker", "tool_route": f"lib:{MEMORY}", "max_asks": max_asks,
            "answer_chars": 800, "pieces": pieces}
    if answer_tokens is not None:
        meta["answer_tokens"] = answer_tokens
    if past_cap is not None:
        meta["past_cap"] = past_cap
    return Task("ask/0", "<S>system<U>context", meta)


class StubPool:
    """Asker turns are scripted in order; the memory answers every question
    the same way. Every call is recorded."""

    def __init__(self, asker_turns):
        self.asker_turns = list(asker_turns)
        self.calls = []

    async def sample(self, messages, stop=(), *, directives=(), max_tokens=None):
        route = directives[0].name
        self.calls.append({"route": route, "messages": [m.content for m in messages],
                           "stop": stop, "max_tokens": max_tokens})
        if route == "asker":
            text, closed = self.asker_turns.pop(0)
            ids = char_tokenize(text) + ((END_ID,) if closed else ())
            return make_turn(text, ids, turn_extras={"route": route})
        return self.answer(messages)

    def answer(self, messages):
        return make_turn("It was the keeper.\n\nAnd more", char_tokenize("It was the keeper."))

    def pool(self, name):
        return self


class EchoPool(StubPool):
    """A pool whose memory answers `A:<the question>`."""

    def answer(self, messages):
        question = messages[0].content[len(PIECES["ask_open"]):-len(PIECES["ask_close"])]
        return make_turn(f"A:{question}", char_tokenize(f"A:{question}"))


CALL = ("think</think>\n\n<tool_call>\n<function=ask_memory>\n<parameter=question>\n"
        "Who lit the lamp?\n</parameter>\n</function>\n</tool_call>")
FINAL = "fine</think>\n\nThe keeper lit the lamp."


def CALLS(questions) -> str:
    return "think</think>\n\n" + "\n".join(
        f"<tool_call>\n<function=ask_memory>\n<parameter=question>\n{q}\n</parameter>\n"
        "</function>\n</tool_call>" for q in questions)


def run(pool, t: Task):
    return go(ENVS.get("ask_memory").instance.run(pool, t))


class ParsingTest(unittest.TestCase):
    def test_well_formed_calls_after_thinking_are_questions(self):
        self.assertEqual(asks_of(CALL), ("Who lit the lamp?",))
        self.assertEqual(asks_of(CALLS(["Q1?", "Q2?"])), ("Q1?", "Q2?"))
        self.assertEqual(asks_of("<tool_call><function=ask_memory>"), ())
        self.assertEqual(asks_of("no call</think> here"), ())
        self.assertEqual(asks_of(CALL.replace("</think>", "")), ())     # thinking never closed


class EpisodeTest(unittest.TestCase):
    def test_a_call_is_answered_by_the_memory_and_the_answer_is_injected(self):
        pool = StubPool([(CALL, True), (FINAL, True)])
        rollout = run(pool, task())
        self.assertEqual([c["route"] for c in pool.calls], ["asker", f"lib:{MEMORY}", "asker"])
        asked = pool.calls[1]
        self.assertEqual(asked["messages"], ["<U>Who lit the lamp?<E>\n<A><think>\n\n</think>\n\n"])
        self.assertEqual(asked["stop"], ("\n\n",))
        second = pool.calls[2]["messages"]
        self.assertEqual(second[1], CALL + END)
        self.assertTrue(second[2].startswith("\n<U><tool_response>\nIt was the keeper."))
        self.assertEqual(len(rollout.turns), 2)
        self.assertEqual(len(rollout.messages), 4)
        self.assertIs(rollout.messages[1], rollout.turns[0].message)
        self.assertEqual(rollout.env_extras["asks"][0]["question"], "Who lit the lamp?")

    def test_a_turn_whose_ids_lack_the_end_token_gets_it_at_the_injection(self):
        pool = StubPool([(CALL, False), (FINAL, True)])
        rollout = run(pool, task())
        self.assertEqual(pool.calls[2]["messages"][1], CALL)
        self.assertTrue(rollout.messages[2].content.startswith(END + "\n<U><tool_response>"))

    def test_asks_past_the_cap_end_the_episode(self):
        rollout = run(StubPool([(CALL, True), (CALL, True)]), task(max_asks=1))
        self.assertEqual(len(rollout.env_extras["asks"]), 1)
        self.assertTrue(rollout.env_extras["final"]["asked_past_cap"])

    def test_refuse_once_answers_a_past_cap_call_with_the_refusal_then_ends(self):
        t = task(max_asks=1, past_cap="refuse_once")
        pool = StubPool([(CALL, True), (CALL, True), (FINAL, True)])
        rollout = run(pool, t)
        self.assertEqual(len(rollout.env_extras["asks"]), 1)
        self.assertEqual(rollout.env_extras["refused"], 1)
        self.assertEqual([c["route"] for c in pool.calls], ["asker", f"lib:{MEMORY}", "asker", "asker"])
        self.assertIn(REFUSAL, pool.calls[-1]["messages"][-1])
        self.assertFalse(rollout.env_extras["final"]["asked_past_cap"])
        rollout = run(StubPool([(CALL, True), (CALL, True), (CALL, True)]), t)
        self.assertTrue(rollout.env_extras["final"]["asked_past_cap"])      # refused once only

    def test_parallel_calls_are_all_answered_in_order_in_one_tool_turn(self):
        pool = EchoPool([(CALLS(["Q1?", "Q2?", "Q3?"]), True), (FINAL, True)])
        rollout = run(pool, task(parallel=True))
        self.assertEqual([c["route"] for c in pool.calls], ["asker"] + [f"lib:{MEMORY}"] * 3 + ["asker"])
        self.assertEqual([a["question"] for a in rollout.env_extras["asks"]], ["Q1?", "Q2?", "Q3?"])
        self.assertEqual([a["turn"] for a in rollout.env_extras["asks"]], [0, 0, 0])
        self.assertEqual(rollout.messages[2].content,
                         "\n<U><tool_response>\nA:Q1?<BETWEEN>A:Q2?<BETWEEN>A:Q3?" + PIECES["answer_close"])
        self.assertEqual((rollout.env_extras["refused"], rollout.env_extras["dropped"]), (0, 0))

    def test_parallel_calls_count_toward_the_cap_and_past_it_are_refused_once(self):
        t = task(max_asks=2, parallel=True, past_cap="refuse_once")
        pool = EchoPool([(CALLS(["Q1?"]), True), (CALLS(["Q2?", "Q3?", "Q4?"]), True), (FINAL, True)])
        rollout = run(pool, t)
        self.assertEqual([a["question"] for a in rollout.env_extras["asks"]], ["Q1?", "Q2?"])
        self.assertEqual(rollout.env_extras["refused"], 2)
        self.assertEqual(rollout.messages[4].content,
                         f"\n<U><tool_response>\nA:Q2?<BETWEEN>{REFUSAL}<BETWEEN>{REFUSAL}"
                         + PIECES["answer_close"])
        self.assertEqual(len(rollout.turns), 3)
        self.assertFalse(rollout.env_extras["final"]["asked_past_cap"])
        rollout = run(EchoPool([(CALLS(["Q1?", "Q2?", "Q3?"]), True), (CALLS(["Q4?"]), True)]), t)
        self.assertEqual(len(rollout.turns), 2)
        self.assertTrue(rollout.env_extras["final"]["asked_past_cap"])

    def test_parallel_calls_past_the_cap_end_the_episode_under_end(self):
        pool = EchoPool([(CALLS(["Q1?", "Q2?"]), True)])
        rollout = run(pool, task(max_asks=1, parallel=True, past_cap="end"))
        self.assertEqual(rollout.env_extras["asks"], [])
        self.assertTrue(rollout.env_extras["final"]["asked_past_cap"])
        self.assertEqual(len(pool.calls), 1)

    def test_an_answer_ceiling_rides_every_memory_call_and_only_when_set(self):
        pool = EchoPool([(CALLS(["Q1?", "Q2?"]), True), (FINAL, True)])
        run(pool, task(parallel=True, answer_tokens=64))
        self.assertEqual([c["max_tokens"] for c in pool.calls[1:3]], [64, 64])
        pool = EchoPool([(CALLS(["Q1?"]), True), (FINAL, True)])
        run(pool, task(parallel=True))
        self.assertIsNone(pool.calls[1]["max_tokens"])

    def test_pieces_without_a_separator_answer_the_first_call_and_count_the_rest(self):
        pool = EchoPool([(CALLS(["Q1?", "Q2?"]), True), (FINAL, True)])
        rollout = run(pool, task())
        self.assertEqual([a["question"] for a in rollout.env_extras["asks"]], ["Q1?"])
        self.assertEqual(rollout.env_extras["dropped"], 1)

    def test_the_generator_loads_the_memory_the_tool_answers_under(self):
        t = task()
        wave = WavePlan((GroupPlan("g", (Sample(t.id, "ask_memory", "asker"),)),))
        self.assertEqual(wave_lib_names(wave, {t.id: t}), (MEMORY,))


if __name__ == "__main__":
    unittest.main()
