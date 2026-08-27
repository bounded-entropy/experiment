"""The observer UI (rlstack.observe.series / .ui): graphs with zero core
interference.

Claims under test: run_series is a pure function of peeks (dictionary +
ledger + eval summaries joined per update); the WSGI app serves the page and
the JSON API from store bytes alone; and the page's panel data carries the
loss-walkback priority (feeds_loss) so the UI never re-derives a declaration.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO

from common import arith_spec, arith_store
from rlstack import FakeEngine, FakeLearner, fake_qwen_schema, run_experiment
from rlstack.observe.panels import PanelError, evaluate, missing_args, panel_args
from rlstack.observe.series import run_series
from rlstack.observe.ui import ui_app

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def call(app, path: str):
    """Minimal WSGI invocation; returns (status, headers, body bytes)."""
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    environ = {"PATH_INFO": path, "REQUEST_METHOD": "GET",
               "wsgi.input": BytesIO(), "wsgi.errors": BytesIO()}
    body = b"".join(app(environ, start_response))
    return captured["status"], captured["headers"], body


class UiTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, train, heldout = arith_store(tmp.name)
        self.report = run_experiment(arith_spec(train, heldout), SCHEMA,
                                     self.store, FakeEngine(), FakeLearner())

    def test_series_joins_ledger_dictionary_and_eval(self) -> None:
        series = run_series(self.store, self.report.run_id)
        self.assertEqual(series["committed"], 4)
        self.assertEqual(series["target"], 4)
        self.assertEqual([u["update"] for u in series["updates"]], [1, 2, 3, 4])
        for update in series["updates"]:
            self.assertIn("reward", update["post"])
            self.assertIn("logprob_gap", update["train"])
        self.assertEqual([e["update"] for e in series["eval"]], [2, 4])
        self.assertIn("reward", series["eval"][0]["means"])
        # the walkback priority rides in, ready for the page
        feeding = [c["name"] for c in series["dictionary"]["columns"]
                   if c["feeds_loss"]]
        self.assertEqual(sorted(feeding), ["advantage", "reward"])

    def test_series_is_none_for_an_unknown_run(self) -> None:
        self.assertIsNone(run_series(self.store, "nope"))

    def test_wsgi_app_serves_page_and_api(self) -> None:
        refreshes = []
        app = ui_app([self.store], refresh=lambda: refreshes.append(1))

        status, headers, body = call(app, "/")
        self.assertEqual(status, "200 OK")
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"feeds the loss", body)

        status, _, body = call(app, f"/run/{self.report.run_id}")
        self.assertEqual(status, "200 OK")          # same document, JS routes

        status, headers, body = call(app, "/api/runs")
        self.assertEqual(status, "200 OK")
        runs = json.loads(body)
        self.assertEqual(runs, [])                  # no host journal: raw run

        status, _, body = call(app, f"/api/run/{self.report.run_id}")
        self.assertEqual(status, "200 OK")
        payload = json.loads(body)
        self.assertEqual(payload["committed"], 4)
        self.assertEqual(len(payload["updates"]), 4)

        status, _, _ = call(app, "/api/run/nope")
        self.assertEqual(status, "404 Not Found")
        self.assertEqual(len(refreshes), 3)         # every API read refreshed

    def test_api_reads_never_mutate_a_live_run(self) -> None:
        staged = self.store.path_of(
            f"runs/{self.report.run_id}/waves/000099.jsonl.gz")
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"staged-not-committed")
        app = ui_app([self.store])
        call(app, f"/api/run/{self.report.run_id}")
        call(app, "/api/runs")
        self.assertTrue(staged.exists())


if __name__ == "__main__":
    unittest.main()


class PanelTest(unittest.TestCase):
    """Custom panels: expressions over logged columns, validated against the
    run's own dictionary (Samarth's rule: every argument must be in the
    pipeline) — never against the submit gate, so a graph never forks a run."""

    def test_arguments_are_extracted(self) -> None:
        self.assertEqual(panel_args("logprob_gap / tokens"),
                         frozenset({"logprob_gap", "tokens"}))
        self.assertEqual(panel_args("log(grad_norm + 1e-9)"),
                         frozenset({"grad_norm"}))

    def test_expressions_are_data_not_code(self) -> None:
        for evil in ("__import__('os')", "reward.real", "reward[0]",
                     "(lambda: 1)()", "max(reward, 1)", "reward if 1 else 2"):
            with self.assertRaises(PanelError):
                panel_args(evil)

    def test_evaluate_matches_arithmetic(self) -> None:
        self.assertAlmostEqual(
            evaluate("reward - 0.5 * advantage", {"reward": 1.0, "advantage": 0.4}),
            0.8)
        self.assertIsNone(evaluate("log(reward)", {"reward": 0.0}))   # domain
        self.assertIsNone(evaluate("reward / tokens", {"reward": 1.0}))  # absent

    def test_missing_args_checks_the_dictionary(self) -> None:
        dictionary = {"columns": [
            {"name": "reward", "phase": "post", "stored": True},
            {"name": "logprob_gap", "phase": "train", "stored": True},
            {"name": "reward", "phase": "eval", "stored": True},
        ]}
        self.assertEqual(missing_args("reward - logprob_gap", dictionary), ())
        self.assertEqual(missing_args("reward - baseline", dictionary),
                         ("baseline",))


class DerivedSeriesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, train, heldout = arith_store(tmp.name)
        self.report = run_experiment(arith_spec(train, heldout), SCHEMA,
                                     self.store, FakeEngine(), FakeLearner())

    def test_derived_panels_compute_with_eval_overlay(self) -> None:
        panels = [{"name": "excess", "expr": "reward - 0.5"},
                  {"name": "gap_ratio", "expr": "logprob_gap / (tokens + 1)"},
                  {"name": "broken", "expr": "reward - baseline"}]
        series = run_series(self.store, self.report.run_id, panels=panels)
        derived = {d["name"]: d for d in series["derived"]}

        excess = derived["excess"]
        self.assertEqual(len(excess["points"]), 4)
        self.assertEqual([u for u, _ in excess["eval"]], [2, 4])   # same expr,
        for (_, train_v) in excess["points"]:                       # both series
            self.assertLessEqual(abs(train_v), 0.5)

        self.assertEqual(len(derived["gap_ratio"]["points"]), 4)
        self.assertEqual(derived["gap_ratio"]["eval"], [])   # tokens not in eval

        self.assertEqual(derived["broken"]["missing"], ["baseline"])
        self.assertEqual(derived["broken"]["points"], [])

    def test_panels_json_in_the_store_is_read(self) -> None:
        self.store.path_of("panels.json").write_text(
            '[{"name": "excess", "expr": "reward - 0.5"}]')
        series = run_series(self.store, self.report.run_id)
        self.assertEqual(series["derived"][0]["name"], "excess")
        self.assertEqual(len(series["derived"][0]["points"]), 4)
