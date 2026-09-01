"""GSM-Symbolic task shaping: golds, ids, chat pieces, and the split rule
(train and eval sets are DISJOINT — the campaign's generalization claim
rests on it)."""

import unittest

from rlstack.data.tasks.gsm_symbolic import (
    ASK, QWEN_CHAT, gold_of, gsm_eval_tasks, gsm_tasks, gsm_train_tasks,
)


def row(family: int, instance: int, question: str = "Q?",
        gold: str = "#### 20") -> dict:
    return {"id": family, "instance": instance, "question": question,
            "answer": f"work...\n{gold}"}


def rows_for(families, instances) -> list[dict]:
    return [row(f, i, question=f"Q {f}-{i}?", gold=f"#### {10 * f + i}")
            for f in families for i in instances]


class GoldTest(unittest.TestCase):
    def test_plain_and_comma_integers_parse(self) -> None:
        self.assertEqual(gold_of(row(0, 0, gold="#### 20")), 20)
        self.assertEqual(gold_of(row(0, 0, gold="#### 1,234")), 1234)
        self.assertEqual(gold_of(row(0, 0, gold="#### -7")), -7)

    def test_non_integer_golds_are_dropped(self) -> None:
        self.assertIsNone(gold_of(row(0, 0, gold="#### 3/4")))
        self.assertIsNone(gold_of(row(0, 0, gold="no marker at all")))
        tasks = gsm_tasks([row(0, 0, gold="#### 3/4"), row(0, 1)])
        self.assertEqual(list(tasks), [(0, 1)])


class ShapeTest(unittest.TestCase):
    def test_task_id_prompt_and_meta(self) -> None:
        task = gsm_tasks([row(3, 7, question="How many?")])[(3, 7)]
        self.assertEqual(task.id, "gsm-symbolic/t03-i07")
        self.assertIn("How many?", task.prompt)
        self.assertIn(ASK, task.prompt)
        self.assertEqual(task.meta["answer"], 20)
        self.assertEqual(task.meta["family"], 3)
        self.assertNotIn("chat", task.meta)

    def test_chat_formatter_formats_and_stashes_pieces(self) -> None:
        task = gsm_tasks([row(0, 0)], chat=lambda t: f"<u>{t}</u>")[(0, 0)]
        self.assertTrue(task.prompt.startswith("<u>"))
        self.assertEqual(task.meta["chat"], QWEN_CHAT)


class SplitTest(unittest.TestCase):
    def test_train_and_eval_are_disjoint_and_sized(self) -> None:
        rows = rows_for(families=(2, 5, 9), instances=range(12))
        train = gsm_train_tasks(rows, family=5, instances=(0, 1))
        held = gsm_eval_tasks(rows, train_family=5, train_instances=(0, 1),
                              families=(2, 5, 9), near=6, far=4)
        self.assertEqual([t.id for t in train],
                         ["gsm-symbolic/t05-i00", "gsm-symbolic/t05-i01"])
        held_ids = [t.id for t in held]
        self.assertEqual(len(held_ids), 6 + 4 + 4)
        self.assertFalse(set(held_ids) & {t.id for t in train})
        near = [i for i in held_ids if i.startswith("gsm-symbolic/t05")]
        self.assertEqual(near, [f"gsm-symbolic/t05-i{n:02d}"
                                for n in range(2, 8)])

    def test_unknown_train_instance_refuses(self) -> None:
        rows = rows_for(families=(1,), instances=(0,))
        with self.assertRaises(KeyError):
            gsm_train_tasks(rows, family=1, instances=(0, 1))


if __name__ == "__main__":
    unittest.main()
