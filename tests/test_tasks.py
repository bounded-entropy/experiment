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


if __name__ == "__main__":
    unittest.main()
