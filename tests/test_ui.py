"""The observer UI (rlstack.observe.series / .host_series / .ui): graphs with
zero core interference.

Claims under test: run_series is a pure function of peeks (dictionary +
ledger + eval summaries joined per update); host_series is a pure function of
one host journal (birth facts, tenancy, gpu channels, and the open numeric
slot); the WSGI app serves the page and the JSON API from store bytes alone;
and the page's panel data carries the loss-walkback priority (feeds_loss) so
the UI never re-derives a declaration.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO

from common import arith_spec, arith_store, generation_spec
from rlstack import (
    FakeEngine, FakeLearner, LocalStore, fake_qwen_schema, run_experiment,
)
from rlstack.observe.host_series import (
    fleet_data, host_series, metric_series, tenancy_lanes,
)
from rlstack.observe.page import asset, document
from rlstack.observe.panels import PanelError, evaluate, missing_args, panel_args
from rlstack.observe.series import run_series
from rlstack.observe.ui import ui_app
from rlstack.observe.views import (partition_metal, render_hosts, run_row, runs_data)

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def call(app, path: str):
    """Minimal WSGI invocation; returns (status, headers, body bytes). A "?"
    splits the query the way a server would — #58's ?root=<folder> rides
    there, not in the path."""
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    path, _, query = path.partition("?")
    environ = {"PATH_INFO": path, "QUERY_STRING": query, "REQUEST_METHOD": "GET",
               "wsgi.input": BytesIO(), "wsgi.errors": BytesIO()}
    body = b"".join(app(environ, start_response))
    return captured["status"], captured["headers"], body


def fabricate_heldout(store, run_id) -> None:
    """Both eras of held-out data, hand-written: LEGACY in-run summaries (a
    pre-#70 run's eval/ — nothing writes these any more, but old runs must
    render) and a MEASUREMENT (the #70 shape)."""
    import json as _json
    for update in (2, 4):
        store._write(
            f"{store.run_prefix(run_id)}/eval/{update}/summary.json",
            _json.dumps({"update": update, "episodes": 16,
                         "means": {"reward": 0.25 * update}}).encode())
    store.open_measurement(run_id, "heldout",
                           {"every": 2, "post": ["verifier"]})
    for update in (2, 4):
        store.append_measurement_point(
            run_id, "heldout",
            {"update": update, "episodes": 16,
             "means": {"reward": 0.1 * update}})


class UiTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, train, heldout = arith_store(tmp.name)
        self.train_uri = train
        self.report = run_experiment(arith_spec(train, heldout), SCHEMA,
                                     self.store, FakeEngine(), FakeLearner())
        fabricate_heldout(self.store, self.report.run_id)

    def test_unhosted_run_page_has_the_same_row_as_the_index(self) -> None:
        app = ui_app([self.store])
        status, _, body = call(app, f"/api/run/{self.report.run_id}/page")
        self.assertEqual(status, "200 OK")
        row = json.loads(body)["row"]
        self.assertIsNotNone(row)
        self.assertEqual(row, runs_data([self.store])[0])

    def test_committed_completion_supersedes_a_stale_parked_note(self) -> None:
        self.store.append_fleet_event({
            "event": "parked", "t": 10, "run_id": self.report.run_id,
            "reason": "waiting for metal"})
        app = ui_app([self.store])
        status, _, body = call(app, "/api/runs")
        self.assertEqual(status, "200 OK")
        row = json.loads(body)["runs"][0]
        self.assertEqual((row["committed"], row["target"], row["status"]),
                         (4, 4, "done"))
        self.assertNotIn("parked", row)
        # Reading completion must not rewrite the historical desk event.
        self.assertEqual(self.store.read_fleet_log()[0]["event"], "parked")

    def test_unhosted_incomplete_run_is_discovered_without_inventing_liveness(self) -> None:
        run = self.store.open_run("unhosted", {"run_id": "unhosted", "spec": "{}"},
                                  subdir="research/replication")
        run.write_plan("train", self.store.peek_plan(self.report.run_id, "train"))
        self.store.annotate_run("unhosted", name="replication", tags=["research"])
        # Unsealed state must survive observer reads: attaching would sweep it.
        key = f"{run.run_dir}/adapters/pi@9.bin"
        self.store._write(key, b"unsealed")
        before = {key: self.store._read(key) for key in self.store._list("")}
        row = run_row([self.store], "research/replication/unhosted", "")
        self.assertEqual((row["status"], row["committed"], row["target"]),
                         ("unknown", 0, 4))
        self.assertEqual(row["hosts"], [])
        self.assertEqual(row["open_hosts"], [])
        self.assertEqual(row["subdir"], "research/replication")
        self.assertEqual(row["name"], "replication")
        self.assertEqual(before, {key: self.store._read(key)
                                  for key in self.store._list("")})
        self.store.append_host_event("host", {"event": "attach", "t": 10,
                                              "run_id": "unhosted"})
        rows = [r for r in runs_data([self.store]) if r["run_id"] == "unhosted"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "running")
        self.assertEqual(rows[0]["hosts"], ["host"])

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
        # ...and the #70 shape beside it: named measurements, points intact
        told = {m["name"]: m for m in series["measurements"]}
        self.assertEqual([pt["update"] for pt in told["heldout"]["points"]],
                         [2, 4])
        self.assertEqual(told["heldout"]["manifest"]["every"], 2)
        # the walkback priority rides in, ready for the page
        feeding = [c["name"] for c in series["dictionary"]["columns"]
                   if c["feeds_loss"]]
        self.assertEqual(sorted(feeding), ["advantage", "reward"])

    def test_series_is_none_for_an_unknown_run(self) -> None:
        self.assertIsNone(run_series(self.store, "nope"))

    def test_a_generation_only_runs_series_counts_rollouts(self) -> None:
        """ADR 0006 Part B: the observer reads the same progress predicate the
        desk does, so a run with no ledger shows its SEALED ROLLOUTS against
        its rollout plan and says which — its update panels are legitimately
        empty rather than a stalled training run's."""
        report = run_experiment(generation_spec(self.train_uri), SCHEMA,
                                self.store, FakeEngine(), None)
        series = run_series(self.store, report.run_id)
        self.assertEqual((series["extent"], series["committed"],
                          series["target"]), ("rollout", 4, 4))
        self.assertEqual(series["updates"], [])

        # the runs list reads the same predicate: a journaled tenancy whose
        # rollouts are all sealed is DONE, whatever its detach said
        self.store.append_host_event("l4-a", {
            "event": "attach", "t": 1.0, "run_id": report.run_id,
            "pools": ["main"], "store": self.store.describe()})
        row = next(r for r in runs_data([self.store])
                   if r["run_id"] == report.run_id)
        self.assertEqual((row["extent"], row["committed"], row["target"],
                          row["status"]), ("rollout", 4, 4, "done"))

    def test_a_measurement_is_a_first_class_chart_metric(self) -> None:
        """The charts page plots `<measurement>:<mean>` beside the ledger
        metrics: the name appears in the menu, and its overlay series carries
        the measurement's points at the measured versions."""
        from rlstack.observe.select import metric_names, overlay

        self.store.append_host_event("h", {
            "event": "attach", "t": 1.0, "run_id": self.report.run_id})
        names = metric_names([self.store])
        self.assertIn("heldout:reward", names)
        self.assertIn("reward", names)
        told = overlay([self.store], "heldout:reward")
        (series,) = told["series"]
        self.assertEqual(series["run_id"], self.report.run_id)
        self.assertEqual(series["points"], [[2, 0.2], [4, 0.4]])
        # the ledger reading is untouched by the qualified grammar
        plain = overlay([self.store], "reward")
        self.assertTrue(plain["series"][0]["points"])

    def test_wsgi_app_serves_page_and_api(self) -> None:
        refreshes = []
        app = ui_app([self.store], refresh=lambda: refreshes.append(1))

        status, headers, body = call(app, "/")
        self.assertEqual(status, "200 OK")
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"/web/app.js", body)      # the document loads the modules

        status, _, body = call(app, f"/run/{self.report.run_id}")
        self.assertEqual(status, "200 OK")          # same document, JS routes

        status, headers, body = call(app, "/api/runs")
        body = json.dumps(json.loads(body)["runs"]).encode()
        self.assertEqual(status, "200 OK")
        runs = json.loads(body)
        self.assertEqual(len(runs), 1)             # manifest without a host journal
        self.assertEqual(runs[0]["run_id"], self.report.run_id)
        self.assertEqual(runs[0]["status"], "done")
        self.assertEqual(runs[0]["hosts"], [])

        status, _, body = call(app, f"/api/run/{self.report.run_id}")
        self.assertEqual(status, "200 OK")
        payload = json.loads(body)
        self.assertEqual(payload["committed"], 4)
        self.assertEqual(len(payload["updates"]), 4)

        status, _, _ = call(app, "/api/run/nope")
        self.assertEqual(status, "404 Not Found")
        self.assertEqual(len(refreshes), 3)         # every API read refreshed

    def test_a_racing_read_is_a_503_never_a_lying_404(self) -> None:
        """An exception inside an API read (a reload swapping files under a
        scan) answers 503 "try again" — the page must be able to tell a
        transient from the server's own positive "no such run", and the
        worker must survive it."""
        def racing():
            raise FileNotFoundError("swapped mid-scan")
        app = ui_app([self.store], refresh=racing)
        status, _, body = call(app, f"/api/run/{self.report.run_id}")
        self.assertEqual(status, "503 Service Unavailable")
        self.assertIn("transient", json.loads(body)["error"])
        status, _, _ = call(ui_app([self.store]),
                            f"/api/run/{self.report.run_id}")
        self.assertEqual(status, "200 OK")          # the worker lived

    def test_api_reads_never_mutate_a_live_run(self) -> None:
        staged = self.store.path_of(
            f"runs/{self.report.run_id}/waves/000099.jsonl.gz")
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"staged-not-committed")
        app = ui_app([self.store])
        call(app, f"/api/run/{self.report.run_id}")
        call(app, "/api/runs")
        call(app, "/api/hosts")
        self.assertTrue(staged.exists())

    def test_the_switcher_reads_the_runs_api(self) -> None:
        """The run page's dropdown is /api/runs — every field its option
        label prints must be in that payload, or switching is blind."""
        self.store.append_host_event("l4-a", {
            "event": "attach", "t": 10.0, "run_id": self.report.run_id,
            "pools": ["policy"], "n_updates": 4, "store": self.store.describe()})
        app = ui_app([self.store])
        _, _, body = call(app, "/api/runs")
        row = json.loads(body)["runs"][0]
        for field in ("run_id", "status", "committed", "target", "hosts"):
            self.assertIn(field, row)
        self.assertEqual(row["run_id"], self.report.run_id)
        # the dropdown itself lives in the nav module
        self.assertIn('title: "switch experiment"',
                      asset("nav.js")[0].decode("utf-8"))

    def test_the_modules_carry_the_hover_and_switch_machinery(self) -> None:
        """Hover is a page fact (the API carries the numbers, the modules
        reveal them); so are the switcher and the two host routes. The
        document is now a loader, so each claim is checked where it lives."""
        for module, machinery in (
                ("charts.js", ("showTip", "getScreenCTM", "stacked", "histogramCard")),
                ("dom.js", ("export function raw(", "poll")),
                ("nav.js", ("syncSwitcher", "wavePath")),
                ("fleet.js", ("drawFleet", "placement", "fleet throughput")),
                ("host.js", ("drawHost", "journaled metrics", "throughput")),
                ("run.js", ("feeds the loss", "logprob_gap", "sealed waves")),
                ("wave.js", ("drawWave", "distribution")),
        ):
            source = asset(module)[0].decode("utf-8")
            for claim in machinery:
                self.assertIn(claim, source, f"{module} lost {claim!r}")
        self.assertIn(b"/web/style.css", document())

    def test_the_api_carries_every_number_the_hover_reveals(self) -> None:
        _, _, body = call(ui_app([self.store]), f"/api/run/{self.report.run_id}")
        payload = json.loads(body)
        point = payload["updates"][0]
        self.assertIsInstance(point["update"], int)
        self.assertIsInstance(point["post"]["reward"], float)
        self.assertIsInstance(payload["eval"][0]["means"]["reward"], float)


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
        self.train_uri = train
        self.report = run_experiment(arith_spec(train, heldout), SCHEMA,
                                     self.store, FakeEngine(), FakeLearner())
        fabricate_heldout(self.store, self.report.run_id)

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


HOST_JOURNAL = [
    {"event": "host-up", "t": 100.0, "engines": ["Qwen/Qwen3-0.6B"],
     "partition": {"metal": "L4:2", "devices": [0, 1], "memory": 0.9},
     "regimes": [{"name": "policy", "capability": "inference",
                  "base": "Qwen/Qwen3-0.6B", "shape": 2}],
     "store": "modal://rlstack-store"},
    {"event": "attach", "t": 110.0, "run_id": "aaa", "pools": ["policy"],
     "remotes": [], "n_updates": 4, "store": "modal://rlstack-store"},
    {"event": "stats", "t": 120.0,
     "gpus": [{"util": 40, "mem_used": 1000, "mem_total": 23000},
              {"util": 10, "mem_used": 500, "mem_total": 23000}]},
    {"event": "stats", "t": 150.0,
     "gpus": [{"util": 80, "mem_used": 2000, "mem_total": 23000},
              {"util": 20, "mem_used": 700, "mem_total": 23000}]},
    # an event NO reading knows about — the throughput slot, standing in for
    # the emission #50 designs but does not build
    {"event": "throughput", "t": 155.0, "tokens_per_s": 812.5,
     "pool": {"requests": 12}, "sleeping": False, "labels": ["policy"]},
    {"event": "detach", "t": 160.0, "run_id": "aaa", "status": "done",
     "updates_completed": 4},
    {"event": "attach", "t": 170.0, "run_id": "bbb", "pools": ["policy"],
     "remotes": ["teacher"], "n_updates": 8, "store": "modal://rlstack-store"},
]


class HostPageTest(unittest.TestCase):
    """Host-by-host analysis: everything the page shows comes out of
    hosts/<name>/log.jsonl — birth facts (#43's partition + regimes),
    residencies, gpu channels, and the open numeric slot."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        for event in HOST_JOURNAL:
            self.store.append_host_event("l4-a", event)

    def test_birth_facts_come_from_host_up(self) -> None:
        host = host_series([self.store], "l4-a")
        self.assertEqual(host["engines"], ["Qwen/Qwen3-0.6B"])
        self.assertEqual(host["partition"]["metal"], "L4:2")
        self.assertEqual(host["regimes"][0]["capability"], "inference")
        self.assertEqual(len(host["boots"]), 1)
        self.assertEqual((host["first_seen"], host["last_seen"]), (100.0, 170.0))
        self.assertEqual(host["journal_stores"], [self.store.describe()])

    def test_a_host_older_than_partitions_renders_without_them(self) -> None:
        self.store.append_host_event("old", {"event": "host-up", "t": 1.0,
                                             "engines": ["*"]})
        host = host_series([self.store], "old")
        self.assertIsNone(host["partition"])
        self.assertEqual(host["regimes"], [])

    def test_tenancy_pairs_attach_with_detach(self) -> None:
        lanes = tenancy_lanes(HOST_JOURNAL)
        self.assertEqual([lane["run_id"] for lane in lanes], ["aaa", "bbb"])
        done, running = lanes
        self.assertEqual((done["attached"], done["detached"]), (110.0, 160.0))
        self.assertEqual((done["status"], done["completed"]), ("done", 4))
        self.assertIsNone(running["detached"])        # still on the host
        self.assertEqual(running["status"], "running")
        self.assertEqual(running["remotes"], ["teacher"])

    def test_a_run_that_attached_twice_is_two_residencies(self) -> None:
        lanes = tenancy_lanes([
            {"event": "attach", "t": 1.0, "run_id": "aaa"},
            {"event": "detach", "t": 2.0, "run_id": "aaa", "status": "failed"},
            {"event": "attach", "t": 3.0, "run_id": "aaa"},
        ])
        self.assertEqual([(lane["attached"], lane["detached"]) for lane in lanes],
                         [(1.0, 2.0), (3.0, None)])
        self.assertEqual(lanes[0]["status"], "failed")

    def test_gpu_channels_are_per_device_series(self) -> None:
        gpus = host_series([self.store], "l4-a")["gpus"]
        self.assertEqual([g["device"] for g in gpus], [0, 1])
        self.assertEqual(gpus[0]["util"], [[120.0, 40], [150.0, 80]])
        self.assertEqual(gpus[1]["mem_used"], [[120.0, 500], [150.0, 700]])
        self.assertEqual(gpus[0]["mem_total"], 23000)

    def test_unclaimed_numeric_fields_become_generic_series(self) -> None:
        """THE THROUGHPUT SLOT: an event no reading recognizes is plotted
        anyway, keyed <event>.<field> — nested numbers included, bools and
        lists excluded (a flag is not a series)."""
        metrics = {m["key"]: m for m in metric_series(HOST_JOURNAL)}
        self.assertEqual(metrics["throughput.tokens_per_s"]["points"],
                         [[155.0, 812.5]])
        self.assertEqual(metrics["throughput.pool.requests"]["points"],
                         [[155.0, 12.0]])
        self.assertEqual(metrics["throughput.tokens_per_s"]["event"], "throughput")
        self.assertNotIn("throughput.sleeping", metrics)     # a bool is a flag
        self.assertNotIn("throughput.labels", metrics)       # a list is a facet

    def test_the_named_readings_claim_their_own_fields(self) -> None:
        """What the tenancy/gpu readings already render never doubles as a
        generic metric — the slot is for facts nothing else shows."""
        keys = {m["key"] for m in metric_series(HOST_JOURNAL)}
        for claimed in ("attach.n_updates", "detach.updates_completed",
                        "stats.gpus", "host-up.partition.memory"):
            self.assertNotIn(claimed, keys)

    def test_host_series_is_none_for_an_unknown_host(self) -> None:
        self.assertIsNone(host_series([self.store], "nope"))


class DuplicateAdoptionTest(unittest.TestCase):
    """One run carried on two hosts at once: the copy that died says failed,
    the copy still attached says running, and the run IS running — a
    detach closes one residency, not the run (observed live, ADR 0005's
    layer-10 arm)."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        self.store.append_host_event("metal-b", {
            "event": "attach", "t": 26.0, "run_id": "arm", "pools": ["main"]})
        self.store.append_host_event("metal-a", {
            "event": "attach", "t": 23.0, "run_id": "arm", "pools": ["main"]})
        self.store.append_host_event("metal-a", {
            "event": "detach", "t": 73.0, "run_id": "arm", "status": "failed"})

    def test_a_live_residency_outranks_a_newer_death_elsewhere(self) -> None:
        row = next(r for r in runs_data([self.store]) if r["run_id"] == "arm")
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["open_hosts"], ["metal-b"])
        self.assertEqual(sorted(row["hosts"]), ["metal-a", "metal-b"])

    def test_with_every_residency_closed_the_newest_detach_speaks(self) -> None:
        self.store.append_host_event("metal-b", {
            "event": "detach", "t": 90.0, "run_id": "arm", "status": "done"})
        row = next(r for r in runs_data([self.store]) if r["run_id"] == "arm")
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["open_hosts"], [])


class FleetPageTest(unittest.TestCase):
    """The global reading: placement and load across hosts; per-run facts
    stay on the run pages."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        for event in HOST_JOURNAL:
            self.store.append_host_event("l4-a", event)
        # l4-b is journaled in the PRE-#55 spelling on purpose: partition
        # "gpuset" (now "metal") and regime "kind" (now "capability"). Journals
        # are append-only history, so both spellings live on the volume forever
        # and the observer must read either — see the tolerance test below.
        self.store.append_host_event("l4-b", {
            "event": "host-up", "t": 200.0, "engines": ["Qwen/Qwen3-32B"],
            "partition": {"gpuset": "L4:4", "devices": [0, 1, 2, 3],
                          "memory": 0.9},
            "regimes": [{"name": "teacher", "kind": "inference",
                         "base": "Qwen/Qwen3-32B", "shape": 4}]})
        self.store.append_host_event("l4-b", {
            "event": "attach", "t": 210.0, "run_id": "bbb", "pools": ["teacher"],
            "n_updates": 8, "store": "modal://rlstack-store"})

    def test_fleet_joins_hosts_runs_and_one_window(self) -> None:
        fleet = fleet_data([self.store])
        by_host = {h["host"]: h for h in fleet["hosts"]}
        self.assertEqual(sorted(by_host), ["l4-a", "l4-b"])
        self.assertEqual([lane["run_id"] for lane in by_host["l4-a"]["tenancy"]],
                         ["aaa", "bbb"])
        self.assertEqual(by_host["l4-a"]["util"], [[120.0, 40], [150.0, 80]])
        self.assertEqual(by_host["l4-b"]["regimes"][0]["name"], "teacher")
        self.assertEqual(fleet["window"], [100.0, 210.0])
        # a run resident on two hosts is a FLEET fact, and renders as one
        placed = {run["run_id"]: run["hosts"] for run in fleet["runs"]}
        self.assertEqual(sorted(placed["bbb"]), ["l4-a", "l4-b"])

    def test_a_pre_rename_journal_still_names_its_metal(self) -> None:
        """#55 renamed Partition.gpuset -> .metal, and a journal is history: a
        host booted before the rename still says "gpuset" on the volume. The
        observer reads either spelling, so an old host renders its metal
        instead of a blank."""
        self.assertEqual(partition_metal({"gpuset": "L4:4"}), "L4:4")
        self.assertEqual(partition_metal({"metal": "L4:2"}), "L4:2")
        rendered = render_hosts([self.store])
        self.assertIn("L4:4", rendered)     # l4-b, journaled as "gpuset"
        self.assertIn("L4:2", rendered)     # l4-a, journaled as "metal"

    def test_wsgi_serves_the_host_routes(self) -> None:
        app = ui_app([self.store])
        status, _, body = call(app, "/api/hosts")
        self.assertEqual(status, "200 OK")
        self.assertEqual(len(json.loads(body)["hosts"]), 2)

        status, _, body = call(app, "/api/host/l4-a")
        self.assertEqual(status, "200 OK")
        self.assertEqual(json.loads(body)["host"], "l4-a")

        status, _, _ = call(app, "/api/host/nope")
        self.assertEqual(status, "404 Not Found")

        for page in ("/hosts", "/host/l4-a"):
            status, headers, _ = call(app, page)
            self.assertEqual(status, "200 OK")     # same document, JS routes
            self.assertIn("text/html", headers["Content-Type"])


class PageRoutesTest(unittest.TestCase):
    """ONE REQUEST PER PAGE (observe/ui.py): a page route carries what its
    page used to fetch as two or three requests, and the run page's row is
    the index's row for that run, read without walking the index."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, train, heldout = arith_store(tmp.name)
        self.report = run_experiment(arith_spec(train, heldout), SCHEMA,
                                     self.store, FakeEngine(), FakeLearner())
        self.run_id = self.report.run_id
        # the index speaks of a run its HOST JOURNALS mention: one attach
        self.store.append_host_event("l4-a", {
            "event": "attach", "t": 1.0, "run_id": self.run_id})
        self.app = ui_app([self.store])

    def body(self, path: str) -> dict:
        status, _, body = call(self.app, path)
        self.assertEqual(status, "200 OK", path)
        return json.loads(body)

    def test_the_run_page_is_one_answer(self) -> None:
        page = self.body(f"/api/run/{self.run_id}/page")
        self.assertEqual(set(page), {"run", "timing", "waves", "row", "now"})
        self.assertEqual(page["run"], self.body(f"/api/run/{self.run_id}"))
        self.assertEqual(page["timing"], self.body(f"/api/run/{self.run_id}/timing"))
        self.assertEqual(page["waves"], self.body(f"/api/run/{self.run_id}/waves"))
        [row] = [r for r in self.body("/api/runs")["runs"]
                 if r["run_id"] == self.run_id]
        for key in ("run_id", "folder", "hosts", "name", "tags", "subdir",
                    "committed", "target", "extent", "status"):
            self.assertEqual(page["row"][key], row[key], key)

    def test_an_unknown_run_has_no_page(self) -> None:
        status, _, _ = call(self.app, "/api/run/nope/page")
        self.assertEqual(status, "404 Not Found")

    def test_run_row_is_the_index_row_without_the_walk(self) -> None:
        [row] = [r for r in runs_data([self.store]) if r["run_id"] == self.run_id]
        self.assertEqual(run_row([self.store], self.run_id, row["folder"]), row)
        self.assertIsNone(run_row([self.store], "nope", row["folder"]))

    def test_the_fleet_page_is_one_answer(self) -> None:
        page = self.body("/api/fleet/page?hours=24")
        self.assertEqual(set(page), {"fleet", "flow", "now"})
        self.assertEqual(set(page["fleet"]), set(self.body("/api/hosts?hours=24")))
        self.assertEqual(set(page["flow"]), set(self.body("/api/fleet?hours=24")))
        self.assertIn(self.run_id, [r["run_id"] for r in page["fleet"]["runs"]])

    def test_the_charts_page_draws_a_metric_that_exists(self) -> None:
        page = self.body("/api/charts/page?metric=reward&q=")
        self.assertEqual(set(page), {"metrics", "metric", "series"})
        self.assertIn("reward", page["metrics"])
        self.assertEqual(page["metric"], "reward")
        self.assertIn(self.run_id, [s["run_id"] for s in page["series"]["series"]])
        # a metric no run carries: the first that exists is drawn instead
        page = self.body("/api/charts/page?metric=nonsense&q=")
        self.assertEqual(page["metric"], page["metrics"][0])
