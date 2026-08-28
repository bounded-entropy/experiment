"""The observer's second half (#56): the static page, the wave browser, the
step economics, and the fleet aggregates.

Claims under test: the UI is FILES (index.html plus native ES modules, every
relative import resolving to a shipped asset) and every page route still
returns the same document; the wave browser reads SEALED artifacts only and
computes its distributions server-side from the run's own bytes; run_timing and
the traffic channels turn the emission contract's journal lines into rails; and
the fleet aggregate sums per bucket — each host's own mean rate, summed across
hosts, so a host that sampled twice never counts twice.

The journal rows written here are the contract the emission plane implements:

    {"event": "traffic", "t", "window_s", "prefill_tokens", "decode_tokens",
     "requests", "ttft_ms_mean", "admit_wait_ms_mean", "admit_wait_ms_max",
     "inflight"}
    {"event": "update", "t", "run_id", "update", "seconds",
     "phases": {"collect", "post", "train", "seal"}}
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest

from common import make_turn
from rlstack import Group, LocalStore, Message, Role, Task, Trajectory, Wave
from rlstack.data.trajectory import wave_to_rows
from rlstack.observe.aggregate import (
    fleet_throughput, moments, run_timing, traffic_channels,
)
from rlstack.observe.page import asset, asset_names, document
from rlstack.observe.ui import ui_app
from rlstack.observe.waves import histogram, wave_detail, wave_list
from test_ui import call

RUN = "wave0000"
IMPORTS = re.compile(r'from "\./([a-z_]+\.js)"')


def manifest(n_updates: int = 2) -> dict:
    return {"run_id": RUN,
            "spec": json.dumps({"algo": {"schedule": {"n_updates": n_updates}}})}


def trajectory(task_id: str, answer: str, finish: str = "stop") -> Trajectory:
    """One sealed episode: a user prompt and one generated reply."""
    turn = make_turn(answer, tuple(ord(c) for c in answer), finish=finish,
                     logprobs=tuple(-0.5 for _ in answer))
    return Trajectory(
        task=Task(task_id, f"What is {task_id}?", {"answer": answer}),
        messages=(Message(Role.USER, f"What is {task_id}?"), turn.message),
        turns=(turn,))


def sealed_wave(store: LocalStore, update: int = 1) -> None:
    """A committed wave with two groups of two, plus a scalar column, a
    token_level teacher column, and its ledger line — the trainer's own order:
    wave, postdata, then THE commit."""
    groups = [Group("t1", [trajectory("t1", "42"), trajectory("t1", "43", "length")]),
              Group("t2", [trajectory("t2", "7"), trajectory("t2", "8")])]
    run = store.open_run(RUN, manifest())
    run.write_wave(update, wave_to_rows(Wave(groups)))
    run.write_postdata(update, {
        "reward": [1.0, 0.0, 1.0, 0.5],
        "teacher_logprobs": [[-1.0, -1.0], [-1.0, -1.0], [-0.25], [-0.25]],
    })
    run.append_ledger({"update": update, "bundle_id": "bundle:abc123",
                       "versions": {"pi": update},
                       "wave": {"trajectories": 4, "groups": 2},
                       "post": {"reward": 0.625},
                       "train": {"logprob_gap": 1e-6}})


class AssetTest(unittest.TestCase):
    """THE DOCUMENT IS A FILE: no build step, no CDN, no bundler — index.html
    plus ES modules the browser loads itself, served from the package."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        self.app = ui_app([self.store])

    def test_every_page_route_returns_the_one_document(self) -> None:
        for path in ("/", "/run/abc", "/run/abc/wave/3", "/hosts", "/host/l4-a"):
            status, headers, body = call(self.app, path)
            self.assertEqual(status, "200 OK")
            self.assertIn("text/html", headers["Content-Type"])
            self.assertEqual(body, document())

    def test_assets_are_served_with_their_content_types(self) -> None:
        for name, kind in (("app.js", "text/javascript"), ("style.css", "text/css"),
                           ("index.html", "text/html")):
            status, headers, body = call(self.app, "/web/" + name)
            self.assertEqual(status, "200 OK")
            self.assertIn(kind, headers["Content-Type"])
            self.assertEqual(body, asset(name)[0])

    def test_an_asset_is_one_file_name(self) -> None:
        for path in ("/web/../ui.py", "/web/nope.js", "/web/", "/web/sub/dir.js"):
            status, _, _ = call(self.app, path)
            self.assertEqual(status, "404 Not Found", path)

    def test_every_module_import_resolves_to_a_shipped_file(self) -> None:
        """A relative import that names nothing is a 404 the browser reports
        to nobody — so the suite resolves the module graph instead."""
        names = set(asset_names())
        self.assertIn("index.html", names)
        for module in sorted(n for n in names if n.endswith(".js")):
            for target in IMPORTS.findall(asset(module)[0].decode("utf-8")):
                self.assertIn(target, names, f"{module} imports missing {target}")
        page = document().decode("utf-8")
        for reference in ("/web/app.js", "/web/style.css"):
            self.assertIn(reference, page)


class WaveBrowserTest(unittest.TestCase):
    """THE CHARTER AMENDMENT: the observer may read SEALED artifacts through
    the peek verbs — never live state, never an attach, never a write."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        sealed_wave(self.store)
        self.app = ui_app([self.store])

    def test_the_listing_is_the_ledger_tail(self) -> None:
        listing = wave_list(self.store, RUN)
        self.assertEqual(listing["committed"], 1)
        wave = listing["waves"][0]
        self.assertEqual((wave["update"], wave["trajectories"], wave["groups"]),
                         (1, 4, 2))
        self.assertEqual(wave["bundle_id"], "bundle:abc123")
        self.assertEqual(wave["post"]["reward"], 0.625)
        self.assertIsNone(wave_list(self.store, "nope"))

    def test_the_detail_groups_trajectories_as_they_were_trained(self) -> None:
        wave = wave_detail(self.store, RUN, 1)
        self.assertEqual([g["key"] for g in wave["groups"]], ["t1", "t2"])
        first = wave["groups"][0]["trajectories"][0]
        self.assertEqual(first["index"], 0)
        self.assertEqual(first["task"]["id"], "t1")
        self.assertEqual([m["role"] for m in first["messages"]], ["user", "assistant"])
        self.assertIsNone(first["messages"][0]["turn"])       # the prompt
        self.assertEqual(first["messages"][1]["turn"], 0)     # generated here
        self.assertEqual(first["tokens"], 2)
        self.assertEqual(first["finish"], "stop")
        self.assertEqual(first["bundle_id"], "bundle:abc123")
        self.assertEqual(first["policy_version"], {"pi": 3})

    def test_postdata_aligns_to_wave_order(self) -> None:
        wave = wave_detail(self.store, RUN, 1)
        rewards = [t["post"][0]["value"] for group in wave["groups"]
                   for t in group["trajectories"]]
        self.assertEqual(rewards, [1.0, 0.0, 1.0, 0.5])       # row order, exactly
        facts = {f["name"]: f for f in wave["groups"][0]["trajectories"][0]["post"]}
        self.assertEqual(facts["reward"]["kind"], "scalar")
        self.assertEqual(facts["teacher_logprobs"]["kind"], "token_level")
        self.assertEqual(facts["teacher_logprobs"]["tokens"], 2)

    def test_the_distributions_come_out_of_the_sealed_bytes(self) -> None:
        summary = wave_detail(self.store, RUN, 1)["summary"]
        self.assertEqual((summary["trajectories"], summary["groups"]), (4, 2))
        self.assertEqual(summary["tokens"], 6)                # 2+2+1+1
        self.assertEqual(summary["finish"], {"eos": 0, "length": 1, "stop": 3})
        names = [panel["name"] for panel in summary["panels"]]
        self.assertEqual(names[0], "reward")                  # reward leads
        self.assertIn("generation length", names)
        self.assertIn("sampled KL", names)
        reward = summary["panels"][0]["histogram"]
        self.assertEqual((reward["n"], reward["min"], reward["max"]), (4, 0.0, 1.0))
        self.assertEqual(sum(b["count"] for b in reward["bins"]), 4)
        columns = {c["name"]: c for c in summary["columns"]}
        self.assertEqual(columns["reward"]["mean"], 0.625)
        self.assertEqual(columns["teacher_logprobs"]["n"], 6)  # over all tokens

    def test_the_sampled_kl_is_behavior_minus_teacher(self) -> None:
        """OPD-style: the sealed behavior logprobs against the teacher column
        the pipeline scored — same tokens, so the wave carries its own KL."""
        wave = wave_detail(self.store, RUN, 1)
        first = wave["groups"][0]["trajectories"][0]
        self.assertAlmostEqual(first["kl"], 0.5)              # -0.5 − (-1.0)
        third = wave["groups"][1]["trajectories"][0]
        self.assertAlmostEqual(third["kl"], -0.25)            # -0.5 − (-0.25)
        self.assertAlmostEqual(wave["summary"]["kl_mean"], 0.125)

    def test_the_routes_serve_the_browser(self) -> None:
        status, _, body = call(self.app, f"/api/run/{RUN}/waves")
        self.assertEqual(status, "200 OK")
        self.assertEqual(json.loads(body)["waves"][0]["update"], 1)

        status, _, body = call(self.app, f"/api/run/{RUN}/wave/1")
        self.assertEqual(status, "200 OK")
        self.assertEqual(len(json.loads(body)["groups"]), 2)

        for missing in (f"/api/run/{RUN}/wave/9", "/api/run/nope/waves",
                        f"/api/run/{RUN}/wave/x"):
            status, _, _ = call(self.app, missing)
            self.assertEqual(status, "404 Not Found", missing)

    def test_reading_a_wave_never_sweeps_unsealed_work(self) -> None:
        """peek_wave reads; open_run would DISCARD the staged update. The
        difference is the whole charter."""
        staged = self.store.path_of(f"runs/{RUN}/waves/000099.jsonl.gz")
        staged.write_bytes(b"staged-not-committed")
        call(self.app, f"/api/run/{RUN}/waves")
        call(self.app, f"/api/run/{RUN}/wave/1")
        self.assertTrue(staged.exists())

    def test_a_constant_column_is_one_bucket(self) -> None:
        flat = histogram([2.0, 2.0, 2.0])
        self.assertEqual(len(flat["bins"]), 1)
        self.assertEqual(flat["bins"][0]["count"], 3)
        self.assertEqual(histogram([])["bins"], [])


TRAFFIC = {"event": "traffic", "t": 100.0, "window_s": 10.0,
           "prefill_tokens": 1000, "decode_tokens": 500, "requests": 5,
           "ttft_ms_mean": 42.5, "admit_wait_ms_mean": 1.5,
           "admit_wait_ms_max": 9.0, "inflight": 3}
UPDATE = {"event": "update", "t": 100.0, "run_id": RUN, "update": 1,
          "seconds": 20.0,
          "phases": {"collect": 8.0, "post": 4.0, "train": 7.0, "seal": 1.0}}


class EmissionReadingTest(unittest.TestCase):
    """The contract's two journal lines, read as rails: rates divide by the
    window the host itself declared, and an update's seconds decompose."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        self.store.open_run(RUN, manifest())
        for event in (
                {"event": "host-up", "t": 90.0, "engines": ["Qwen/Qwen3-0.6B"]},
                {"event": "attach", "t": 95.0, "run_id": RUN, "pools": ["policy"]},
                TRAFFIC,
                dict(TRAFFIC, t=110.0, prefill_tokens=2000, inflight=1),
                UPDATE,
                dict(UPDATE, t=130.0, update=2, seconds=10.0,
                     phases={"collect": 3.0, "post": 1.0, "train": 5.0, "seal": 1.0}),
        ):
            self.store.append_host_event("l4-a", event)

    def test_traffic_becomes_six_rails(self) -> None:
        channels = traffic_channels(self.store.read_host_log("l4-a"))
        self.assertEqual(channels["prefill_tok_s"], [[100.0, 100.0], [110.0, 200.0]])
        self.assertEqual(channels["decode_tok_s"][0], [100.0, 50.0])
        self.assertEqual(channels["requests_s"][0], [100.0, 0.5])
        self.assertEqual(channels["ttft_ms"][0], [100.0, 42.5])
        self.assertEqual(channels["admit_wait_ms"][0], [100.0, 1.5])
        self.assertEqual(channels["inflight"], [[100.0, 3.0], [110.0, 1.0]])

    def test_run_timing_decomposes_each_update(self) -> None:
        timing = run_timing(self.store, RUN)
        self.assertEqual([u["update"] for u in timing["updates"]], [1, 2])
        first = timing["updates"][0]
        self.assertEqual((first["t"], first["seconds"]), (100.0, 20.0))
        self.assertEqual(first["phases"],
                         {"collect": 8.0, "post": 4.0, "train": 7.0, "seal": 1.0})
        self.assertEqual(run_timing(self.store, "nope")["updates"], [])

    def test_moments_are_the_events_that_are_not_samples(self) -> None:
        marks = moments(self.store.read_host_log("l4-a"))
        self.assertEqual([(m["event"], m["t"]) for m in marks],
                         [("host-up", 90.0), ("attach", 95.0)])

    def test_the_host_route_carries_rails_and_moments(self) -> None:
        _, _, body = call(ui_app([self.store]), "/api/host/l4-a")
        host = json.loads(body)
        self.assertEqual(host["channels"]["prefill_tok_s"][0], [100.0, 100.0])
        self.assertEqual(host["moments"][0]["event"], "host-up")

    def test_the_timing_route_404s_for_an_unknown_run(self) -> None:
        app = ui_app([self.store])
        status, _, body = call(app, f"/api/run/{RUN}/timing")
        self.assertEqual(status, "200 OK")
        self.assertEqual(len(json.loads(body)["updates"]), 2)
        status, _, _ = call(app, "/api/run/nope/timing")
        self.assertEqual(status, "404 Not Found")


class FleetAggregateTest(unittest.TestCase):
    """The two sums no host and no run knows: inference tokens/s across every
    serving host, training updates/s across every run."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        # one bucket, three samples: l4-a twice (100 and 300 tok/s -> 200) and
        # l4-b once (200) — the fleet reads 400, not 600
        self.store.append_host_event("l4-a", dict(TRAFFIC, t=100.0,
                                                  prefill_tokens=1000))
        self.store.append_host_event("l4-a", dict(TRAFFIC, t=100.5,
                                                  prefill_tokens=3000))
        self.store.append_host_event("l4-b", dict(TRAFFIC, t=100.2,
                                                  prefill_tokens=2000))
        self.store.append_host_event("l4-a", dict(TRAFFIC, t=400.0))
        self.store.append_host_event("l4-a", UPDATE)
        self.store.append_host_event("l4-b", dict(UPDATE, t=100.4, run_id="bbb"))

    def test_a_host_that_sampled_twice_does_not_count_twice(self) -> None:
        flow = fleet_throughput([self.store])
        self.assertEqual(flow["window"], [100.0, 400.0])
        first = flow["inference"][0]
        self.assertAlmostEqual(first["prefill_tok_s"], 400.0)
        self.assertEqual((first["hosts"], first["samples"]), (2, 3))
        self.assertEqual(first["prefill_tokens"], 6000.0)     # the raw sum, hovered
        self.assertEqual(flow["hosts"], ["l4-a", "l4-b"])

    def test_training_sums_updates_across_runs(self) -> None:
        flow = fleet_throughput([self.store])
        first = flow["training"][0]
        self.assertEqual((first["updates"], first["runs"]), (2, 2))
        self.assertAlmostEqual(first["updates_s"], 2 / flow["bucket_s"])
        self.assertEqual(first["seconds_mean"], 20.0)
        self.assertEqual(first["phases"]["train"], 7.0)
        self.assertEqual(sorted(flow["runs"]), ["bbb", RUN])
        self.assertEqual(flow["totals"]["updates"], 2)

    def test_the_fleet_route_serves_the_aggregate(self) -> None:
        status, _, body = call(ui_app([self.store]), "/api/fleet")
        self.assertEqual(status, "200 OK")
        self.assertEqual(len(json.loads(body)["inference"]), 2)

    def test_an_empty_fleet_is_empty_rather_than_wrong(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        flow = fleet_throughput([LocalStore(tmp.name)])
        self.assertIsNone(flow["window"])
        self.assertEqual((flow["inference"], flow["training"]), ([], []))


if __name__ == "__main__":
    unittest.main()
