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
