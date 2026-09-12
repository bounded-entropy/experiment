"""Known programs exercise reward validity before model results are trusted."""

import unittest

from rlstack.training.post.program_correctness import (
    program_text, score_python, score_sql,
)


class PythonCorrectnessTest(unittest.TestCase):
    def test_all_tests_must_pass(self):
        meta = {"tests": ["assert twice(3)==6", "assert twice(4)==8"]}
        self.assertEqual(score_python("def twice(x):\n return 2*x", meta).reward, 1)
        self.assertEqual(score_python("def twice(x):\n return 6", meta).reward, 0)

    def test_broken_code_is_invalid_and_harmful_import_is_refused(self):
        meta = {"tests": ["assert True"]}
        self.assertEqual(score_python("def wrong(:", meta).valid, 0)
        self.assertEqual(score_python("import os", meta).reason, "restricted")
        self.assertEqual(score_python("open('file','w')", meta).reason, "restricted")

    def test_fences_and_thinking_are_not_executed(self):
        self.assertEqual(program_text("thinking</think>\n```python\nx=1\n```"), "x=1")


class SqlCorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.meta = {"types": ["text", "real"],
                     "rows": [["Alice", 2], ["Bob", 3], ["Alice", 4]],
                     "sql": {"sel": 1, "agg": 4, "conds": [[0, 0, "ALICE"]]}}

    def test_equivalent_queries_and_wrong_query(self):
        self.assertEqual(score_sql("SELECT SUM(c1) FROM data WHERE c0='alice'", self.meta).reward, 1)
        self.assertEqual(score_sql("SELECT SUM(c1) FROM data WHERE c0 IN ('ALICE')", self.meta).reward, 1)
        self.assertEqual(score_sql("SELECT SUM(c1) FROM data", self.meta).reward, 0)

    def test_writes_and_multiple_statements_are_rejected(self):
        self.assertEqual(score_sql("DELETE FROM data", self.meta).valid, 0)
        self.assertEqual(score_sql("SELECT 6; DROP TABLE data", self.meta).valid, 0)

    def test_result_duplicates_are_not_discarded(self):
        self.meta["sql"] = {"sel": 0, "agg": 0, "conds": []}
        self.assertEqual(score_sql("SELECT c0 FROM data ORDER BY c1 DESC", self.meta).reward, 1)
        self.assertEqual(score_sql("SELECT DISTINCT c0 FROM data", self.meta).reward, 0)


if __name__ == "__main__":
    unittest.main()
