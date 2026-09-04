"""The task-set verbs: write to the cas, load back, split deterministically.

Dataset-blind by construction — nothing here touches a network, because
nothing in base.py knows what a dataset is.
"""

from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from rlstack import LocalStore, Task, load_tasks, split_tasks, write_tasks
from rlstack.data.tasks import draw_for
from rlstack.data.tasks.concept_prompts import (
    PROSE_CATEGORIES, SYSTEM_PROMPT, check_hint_concatenates, is_prose,
    prompt_splits, renderer, system_block, task_from_row, user_prompt,
)


def tasks(n: int, prefix: str = "t") -> list[Task]:
    return [Task(id=f"{prefix}-{i:04d}", prompt=f"What is {i}+{i}?",
                 meta={"answer": i + i}) for i in range(n)]


class TestWriteAndLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.store = LocalStore(Path(self.dir.name))

    def test_round_trip_through_the_cas(self) -> None:
        """write_tasks is load_tasks' inverse, id, prompt and meta intact."""
        written = tasks(12)
        uri = write_tasks(self.store, written)
        self.assertTrue(uri.startswith("cas://"))
        self.assertEqual(load_tasks(self.store, uri), written)

    def test_the_uri_is_the_content(self) -> None:
        """Identical tasks hash to one uri wherever they were built; a changed
        prompt is a different set (I3) — which is why a spec can pin a uri."""
        self.assertEqual(write_tasks(self.store, tasks(8)),
                         write_tasks(self.store, tasks(8)))
        edited = tasks(8)
        edited[3] = Task(id=edited[3].id, prompt="different", meta=edited[3].meta)
        self.assertNotEqual(write_tasks(self.store, tasks(8)),
                            write_tasks(self.store, edited))

    def test_order_is_content(self) -> None:
        """Rows keep the order given, so a reordered set is a different set."""
        self.assertNotEqual(write_tasks(self.store, tasks(8)),
                            write_tasks(self.store, list(reversed(tasks(8)))))

    def test_a_repeated_id_is_refused(self) -> None:
        """A plan's leaf names an id alone; two rows under one id would make
        that leaf ambiguous, so the set is refused where it is made."""
        with self.assertRaises(ValueError) as caught:
            write_tasks(self.store, tasks(4) + tasks(1))
        self.assertIn("t-0000", str(caught.exception))

    def test_empty_meta_survives(self) -> None:
        uri = write_tasks(self.store, [Task(id="a", prompt="p")])
        self.assertEqual(load_tasks(self.store, uri), [Task(id="a", prompt="p")])


class TestSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = tasks(1000)
        self.fractions = {"train": 0.95, "eval": 0.05}

    def test_every_task_lands_in_exactly_one_split(self) -> None:
        split = split_tasks(self.tasks, self.fractions, seed=17)
        self.assertEqual(sorted(split), ["eval", "train"])
        members = [t for part in split.values() for t in part]
        self.assertEqual(len(members), len(self.tasks))
        self.assertEqual({t.id for t in members}, {t.id for t in self.tasks})
        self.assertEqual(Counter(t.id for t in members).most_common(1)[0][1], 1)

    def test_the_draw_is_deterministic(self) -> None:
        first = split_tasks(self.tasks, self.fractions, seed=17)
        second = split_tasks(self.tasks, self.fractions, seed=17)
        self.assertEqual({k: [t.id for t in v] for k, v in first.items()},
                         {k: [t.id for t in v] for k, v in second.items()})

    def test_another_seed_draws_another_split(self) -> None:
        held = {t.id for t in split_tasks(self.tasks, self.fractions, 17)["eval"]}
        other = {t.id for t in split_tasks(self.tasks, self.fractions, 18)["eval"]}
        self.assertNotEqual(held, other)

    def test_a_task_never_moves_when_the_source_grows(self) -> None:
        """The property the per-task draw buys: a split is a function of
        (seed, id) alone, so held-out stays held out across dataset versions."""
        small = split_tasks(tasks(200), self.fractions, seed=17)
        large = split_tasks(tasks(1000), self.fractions, seed=17)
        for name in ("train", "eval"):
            grown = {t.id for t in large[name]}
            self.assertTrue({t.id for t in small[name]} <= grown)

    def test_position_does_not_decide(self) -> None:
        """Reordering the input moves no task across the boundary; only the
        order WITHIN a split changes, and that is the input's own order."""
        forward = split_tasks(self.tasks, self.fractions, seed=17)
        backward = split_tasks(list(reversed(self.tasks)), self.fractions, 17)
        for name in ("train", "eval"):
            self.assertEqual({t.id for t in forward[name]},
                             {t.id for t in backward[name]})
            self.assertEqual([t.id for t in forward[name]],
                             [t.id for t in reversed(backward[name])])

    def test_the_counts_land_near_the_fractions(self) -> None:
        """Counts are drawn, not dealt: n=1000 at 5% is ~50, never exactly."""
        split = split_tasks(self.tasks, self.fractions, seed=17)
        self.assertAlmostEqual(len(split["eval"]) / 1000, 0.05, delta=0.02)

    def test_three_way_splits_are_ordered_intervals(self) -> None:
        split = split_tasks(self.tasks, {"a": 0.5, "b": 0.3, "c": 0.2}, seed=3)
        self.assertEqual(sum(len(v) for v in split.values()), 1000)
        for name, lo, hi in (("a", 0.0, 0.5), ("b", 0.5, 0.8), ("c", 0.8, 1.0)):
            for task in split[name]:
                self.assertTrue(lo <= draw_for(3, task.id) < hi)

    def test_a_whole_split_takes_everything(self) -> None:
        split = split_tasks(self.tasks, {"train": 1.0}, seed=17)
        self.assertEqual(len(split["train"]), 1000)

    def test_fractions_must_cover_the_set(self) -> None:
        for bad in ({"train": 0.9, "eval": 0.05}, {"train": 0.9, "eval": 0.2},
                    {"train": 1.2, "eval": -0.2}):
            with self.assertRaises(ValueError):
                split_tasks(self.tasks, bad, seed=17)

    def test_the_draw_is_uniform_over_ids(self) -> None:
        """draw_for is h(seed, id) in [0, 1) — the seed tree's rule applied to
        a task, so the intervals mean what the fractions say."""
        draws = [draw_for(17, t.id) for t in self.tasks]
        self.assertTrue(all(0.0 <= d < 1.0 for d in draws))
        self.assertEqual(len(set(draws)), len(draws))
        self.assertAlmostEqual(sum(draws) / len(draws), 0.5, delta=0.03)


class FakeTokenizer:
    """Qwen3's template in the two respects this builder depends on: a
    message renders as its own block, and NO default system block is emitted
    when none is given. `enable_thinking=False` closes the think block, which
    is what makes the generation prompt longer than a bare header."""

    def __init__(self, default_system: str | None = None) -> None:
        self.default_system = default_system

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            enable_thinking):
        assert tokenize is False and enable_thinking is False
        rendered = ""
        roles = [m["role"] for m in messages]
        if self.default_system is not None and "system" not in roles:
            rendered += f"<|im_start|>system\n{self.default_system}<|im_end|>\n"
        for message in messages:
            rendered += (f"<|im_start|>{message['role']}\n"
                         f"{message['content']}<|im_end|>\n")
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return rendered


ROW = {"prompt_id": "abc123", "prompt": "Write a haiku about rain.",
       "category": "Generation"}


class ConceptPromptsTest(unittest.TestCase):
    """The pure half of ADR 0005's corpus builder — the network path is not
    run here, and nothing in it is: a row becomes a Task, the categories
    filter, and the concatenation refusal is the builder's own gate."""

    def setUp(self) -> None:
        self.render = renderer(FakeTokenizer())
        self.system = SYSTEM_PROMPT.format(concept="happiness")
        self.hint = system_block(self.render, self.system)

    def task(self, row=None, render=None):
        return task_from_row(row or ROW, render or self.render, self.hint,
                             self.system, "happiness")

    def test_the_prose_categories_are_kept_and_the_others_dropped(self) -> None:
        for category in ("Generation", "Open QA", "Brainstorm", "Chat",
                         "Rewrite", "Summarize"):
            self.assertTrue(is_prose(category))
        for category in ("Coding", "Classify", "Closed QA", "Extract"):
            self.assertFalse(is_prose(category))
            self.assertIsNone(self.task({**ROW, "category": category}))
        self.assertEqual(len(PROSE_CATEGORIES), 6)

    def test_a_row_becomes_a_task_carrying_the_hint(self) -> None:
        task = self.task()
        self.assertEqual(task.id, "no-robots/abc123")
        self.assertEqual(task.prompt, user_prompt(self.render, ROW["prompt"]))
        self.assertEqual(task.meta["hint"], self.hint)
        self.assertEqual(task.meta["concept"], "happiness")
        self.assertEqual(task.meta["category"], "Generation")

    def test_the_prompt_carries_no_system_block(self) -> None:
        """The student is asked the instruction ALONE; everything the teacher
        knew that the student does not is in the hint."""
        task = self.task()
        self.assertNotIn("system", task.prompt)
        self.assertNotIn("happiness", task.prompt)
        self.assertIn("<|im_start|>system", task.meta["hint"])

    def test_the_hint_plus_the_prompt_is_the_templated_conversation(self) -> None:
        task = self.task()
        self.assertEqual(
            task.meta["hint"] + task.prompt,
            self.render([{"role": "system", "content": self.system},
                         {"role": "user", "content": ROW["prompt"]}]))

    def test_hint_for_reads_it_back(self) -> None:
        """The convention the environment and the processors share."""
        from rlstack import hint_for

        self.assertEqual(hint_for(self.task()).content, self.hint)

    def test_a_template_with_a_default_system_block_refuses(self) -> None:
        """The failure the check exists for: a template that injects its own
        system block when none is given makes the prompt carry one, so
        hint + prompt would carry two — silently conditioning the teacher on
        text no chat model was trained to read."""
        render = renderer(FakeTokenizer(default_system="You are Qwen."))
        with self.assertRaises(ValueError) as refused:
            task_from_row(ROW, render, system_block(render, self.system),
                          self.system, "happiness")
        self.assertIn("concatenate", str(refused.exception))

    def test_the_check_is_the_one_named_rule(self) -> None:
        check_hint_concatenates(self.render, self.hint,
                                user_prompt(self.render, "hi"),
                                self.system, "hi")
        with self.assertRaises(ValueError):
            check_hint_concatenates(self.render, self.hint + "x",
                                    user_prompt(self.render, "hi"),
                                    self.system, "hi")

    def test_the_concept_is_the_one_substitution(self) -> None:
        other = SYSTEM_PROMPT.format(concept="melancholy")
        self.assertIn("melancholy", other)
        self.assertNotIn("happiness", other)
        # and it reaches the task's identity: another concept is another set
        first = self.task()
        render = self.render
        second = task_from_row(ROW, render, system_block(render, other), other,
                               "melancholy")
        self.assertEqual(first.prompt, second.prompt)   # the ask is the same
        self.assertNotEqual(first.meta, second.meta)    # the telling is not

    def test_the_splits_are_the_asked_sizes_as_fractions(self) -> None:
        splits = prompt_splits(8850, train=2048, heldout=128)
        self.assertAlmostEqual(sum(splits.values()), 1.0, places=12)
        self.assertAlmostEqual(splits["train"], 2048 / 8850, places=12)
        self.assertAlmostEqual(splits["heldout"], 128 / 8850, places=12)
        self.assertEqual(sorted(splits), ["heldout", "rest", "train"])

    def test_the_counts_are_drawn_not_dealt(self) -> None:
        """split_tasks places each task by h(seed, id), so the sizes land
        near the fractions and never on them."""
        corpus = [Task(id=f"no-robots/{i:05d}", prompt="p") for i in range(8850)]
        drawn = split_tasks(corpus, prompt_splits(8850), seed=5)
        self.assertAlmostEqual(len(drawn["train"]) / 8850, 2048 / 8850,
                               delta=0.02)
        self.assertAlmostEqual(len(drawn["heldout"]) / 8850, 128 / 8850,
                               delta=0.02)
        self.assertEqual(sum(len(v) for v in drawn.values()), 8850)

    def test_more_prompts_than_the_corpus_holds_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            prompt_splits(100, train=2048, heldout=128)


if __name__ == "__main__":
    unittest.main()
