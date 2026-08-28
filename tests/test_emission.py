"""The emission plane: what a host measures, where it may put it, and how the
observer reads it back (rlstack.runner.meters + rlstack.observe.host_series).

The claims under test: counters accumulate at the seams the runner owns and are
DRAINED once per stats tick into one windowed `traffic` event (never a row per
request, and a silent tick still says zero); the Trainer journals one `update`
event per commit carrying its four real phases; traffic through the wire is
counted at the SERVING host and nowhere else; the observer turns windows into
the six named channels and update rows into run_timing; and — the hard one —
none of this wall clock reaches a run directory: a hosted run's bytes still
match a raw run's, which is what resume-equivalence is made of.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest

from common import arith_spec, arith_store
from rlstack import (
    Bundle, FakeEngine, FakeLearner, Host, HostService, LocalStore,
    LocalTransport, Message, RemotePool, Role, SamplingSpec, Seeds,
    fake_qwen_schema, run_experiment,
)
from rlstack.observe.host_series import (
    host_series, metric_series, run_timing, traffic_channels,
)
from rlstack.runner.meters import TrafficMeter, UpdateClock

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")

TRAFFIC_KEYS = {"event", "t", "window_s", "prefill_tokens", "decode_tokens",
                "requests", "ttft_ms_mean", "admit_wait_ms_mean",
                "admit_wait_ms_max", "inflight"}
UPDATE_KEYS = {"event", "t", "run_id", "update", "seconds", "phases"}
PHASE_KEYS = {"collect", "post", "train", "seal"}


def go(coro):
    return asyncio.run(coro)


def snapshot(store: LocalStore, run_id: str) -> dict[str, str]:
    """{relative path: sha256} over a whole run directory — the same reading
    resume-equivalence bites on."""
    root = store.path_of(f"runs/{run_id}")
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def events_of(host: Host, kind: str) -> list[dict]:
    return [e for e in host.store.read_host_log(host.name) if e["event"] == kind]


async def one_tick(host: Host) -> None:
    """Let the host's own stats loop take exactly one tick. The loop samples
    before it sleeps, so the first window lands immediately."""
    task = asyncio.create_task(host.run_stats(every=3600.0))
    while not events_of(host, "traffic"):
        await asyncio.sleep(0.001)
    task.cancel()


# ---------------------------------------------------------------------------
# the records themselves
# ---------------------------------------------------------------------------

class TrafficMeterTest(unittest.TestCase):
    """The counter record alone: windowed sums, one gauge, honest nulls."""

    def test_a_window_is_the_counts_over_the_span_they_were_counted_in(self) -> None:
        meter = TrafficMeter(started=100.0)
        meter.opened_request(40)
        meter.first_token_after(0.020)
        meter.decoded(3)
        meter.decoded(2)
        meter.opened_request(10)          # a scoring prefill: no decode, no ttft
        meter.admitted(0.5)
        meter.admitted(0.1)
        meter.released()
        window = meter.drain(110.0)

        self.assertEqual(
            window.row(),
            {"window_s": 10.0, "prefill_tokens": 50, "decode_tokens": 5,
             "requests": 2, "ttft_ms_mean": 20.0, "admit_wait_ms_mean": 300.0,
             "admit_wait_ms_max": 500.0, "inflight": 1})

    def test_a_silent_window_is_zeros_and_nulls(self) -> None:
        """A tick that served nothing still measures: zeros for the counts,
        None for the means — a fabricated 0ms latency would be a lie."""
        window = TrafficMeter(started=1.0).drain(31.0)
        self.assertEqual(
            window.row(),
            {"window_s": 30.0, "prefill_tokens": 0, "decode_tokens": 0,
             "requests": 0, "ttft_ms_mean": None, "admit_wait_ms_mean": None,
             "admit_wait_ms_max": None, "inflight": 0})

    def test_the_drain_resets_the_counts_and_keeps_the_gauge(self) -> None:
        meter = TrafficMeter(started=0.0)
        meter.admitted(0.0)               # still running when the tick lands
        meter.decoded(7)
        self.assertEqual(meter.drain(1.0).decode_tokens, 7)
        second = meter.drain(2.0)
        self.assertEqual(second.decode_tokens, 0)
        self.assertEqual(second.window_s, 1.0)
        self.assertEqual(second.inflight, 1)      # a state, not a window sum
        meter.released()
        self.assertEqual(meter.drain(3.0).inflight, 0)


class UpdateClockTest(unittest.TestCase):
    """Four laps at the Trainer's own boundaries; every second attributed."""

    def test_the_phases_are_exactly_the_update(self) -> None:
        clock = UpdateClock()
        clock.collected()
        clock.posted()
        clock.trained()
        clock.sealed()
        row = clock.row("run-x", 3)

        self.assertEqual(set(row), UPDATE_KEYS)
        self.assertEqual(set(row["phases"]), PHASE_KEYS)
        self.assertEqual((row["event"], row["run_id"], row["update"]),
                         ("update", "run-x", 3))
        self.assertAlmostEqual(row["seconds"], sum(row["phases"].values()),
                               places=9)
        self.assertEqual(row["t"], clock.mark)    # journaled when it is true

    def test_an_unlapped_phase_is_zero(self) -> None:
        clock = UpdateClock()
        clock.collected()
        self.assertEqual(clock.row("r", 1)["phases"]["train"], 0.0)


# ---------------------------------------------------------------------------
# a run on a host
# ---------------------------------------------------------------------------

class HostEmissionTest(unittest.TestCase):
    """The whole pipeline on fakes: engines and arbiter count, the stats tick
    drains, the Trainer commits and journals."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)

    def host(self, name: str = "test-host", **kwargs) -> Host:
        defaults = dict(engines=(FakeEngine(),), learner=FakeLearner(),
                        store=self.store, sampler=lambda: None)
        defaults.update(kwargs)
        return Host(name, **defaults)

    def test_a_run_counts_its_own_traffic(self) -> None:
        """One meter per host: the engine's prefill and decode tokens and the
        arbiter's admissions all land in it, and nothing is left in flight."""
        host = self.host()
        go(host.submit(arith_spec(self.train), SCHEMA))
        window = host.meter.drain(host.meter.started + 1.0)

        self.assertGreater(window.prefill_tokens, 0)
        self.assertGreater(window.decode_tokens, 0)
        self.assertEqual(window.requests, 16)     # 4 updates x 4 trajectories
        self.assertIsNotNone(window.ttft_ms_mean)
        self.assertIsNotNone(window.admit_wait_ms_mean)
        self.assertEqual(window.inflight, 0)

    def test_the_stats_tick_journals_one_traffic_window(self) -> None:
        host = self.host()
        go(host.submit(arith_spec(self.train), SCHEMA))
        go(one_tick(host))

        rows = events_of(host, "traffic")
        self.assertEqual(len(rows), 1)            # one row per TICK, never per request
        row = rows[0]
        self.assertEqual(set(row), TRAFFIC_KEYS)
        self.assertGreater(row["decode_tokens"], 0)
        self.assertEqual(row["requests"], 16)
        self.assertGreater(row["window_s"], 0.0)

    def test_a_zero_traffic_tick_still_emits(self) -> None:
        """A host that served nothing this window says so — a hole in the
        series would be indistinguishable from a dead host."""
        host = self.host("idle-host")
        go(one_tick(host))

        row = events_of(host, "traffic")[0]
        self.assertEqual(set(row), TRAFFIC_KEYS)
        self.assertEqual((row["prefill_tokens"], row["decode_tokens"],
                          row["requests"], row["inflight"]), (0, 0, 0, 0))
        self.assertIsNone(row["ttft_ms_mean"])
        self.assertIsNone(row["admit_wait_ms_mean"])
        self.assertIsNone(row["admit_wait_ms_max"])

    def test_one_update_event_per_commit(self) -> None:
        host = self.host()
        report = go(host.submit(arith_spec(self.train), SCHEMA))

        rows = events_of(host, "update")
        self.assertEqual([row["update"] for row in rows], [1, 2, 3, 4])
        for row in rows:
            self.assertEqual(set(row), UPDATE_KEYS)
            self.assertEqual(set(row["phases"]), PHASE_KEYS)
            self.assertEqual(row["run_id"], report.run_id)
            self.assertGreater(row["seconds"], 0.0)
            self.assertAlmostEqual(row["seconds"],
                                   sum(row["phases"].values()), places=9)
            self.assertGreater(row["phases"]["train"], 0.0)

    def test_two_tenants_journal_their_own_updates(self) -> None:
        """A shared host's traffic is one partition-wide window, but an update
        belongs to the run that committed it."""
        host = self.host()
        a = arith_spec(self.train)
        b = arith_spec(self.train, seeds=Seeds(master=99))

        async def both():
            return await asyncio.gather(host.submit(a, SCHEMA),
                                        host.submit(b, SCHEMA))

        reports = go(both())

        rows = events_of(host, "update")
        self.assertEqual(len(rows), 8)
        for report in reports:
            mine = [row for row in rows if row["run_id"] == report.run_id]
            self.assertEqual([row["update"] for row in mine], [1, 2, 3, 4])

    def test_wall_clock_never_reaches_the_run_directory(self) -> None:
        """THE HARD INVARIANT: the emission plane adds custody, never bytes —
        a hosted run's whole run directory still matches a raw run's."""
        host = self.host()
        hosted = go(host.submit(arith_spec(self.train, self.heldout), SCHEMA))

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        other, train, heldout = arith_store(tmp.name)
        raw = run_experiment(arith_spec(train, heldout), SCHEMA, other,
                             FakeEngine(), FakeLearner())

        self.assertEqual(hosted.run_id, raw.run_id)
        self.assertEqual(snapshot(self.store, hosted.run_id),
                         snapshot(other, raw.run_id))

    def test_a_run_on_no_host_journals_nothing(self) -> None:
        """No host, no journal: run_experiment takes no HostJournal and emits
        no update events rather than inventing a host to blame."""
        run_experiment(arith_spec(self.train), SCHEMA, self.store,
                       FakeEngine(), FakeLearner())
        self.assertEqual(self.store.list_hosts(), [])


class WireEmissionTest(unittest.TestCase):
    """Traffic through the wire is counted where it is SERVED."""

    def test_the_serving_host_counts_the_remote_traffic(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, _, _ = arith_store(tmp.name)
        serving = Host("srv", engines=(FakeEngine(),), learner=FakeLearner(),
                       store=store)
        client = Host("cli", engines=(FakeEngine(),), learner=FakeLearner(),
                      store=store)
        remote = RemotePool(LocalTransport(HostService(serving)))
        remote.add_bundle(Bundle("bundle:x", {"pi": 0}))

        async def sample():
            return [event async for event in remote.sample_tokens(
                (Message(Role.USER, "What is 2+2?"),), SamplingSpec(), (),
                "bundle:x", seed=7)]

        self.assertGreater(len(go(sample())), 1)
        served = serving.meter.drain(serving.meter.started + 1.0)
        self.assertEqual(served.requests, 1)
        self.assertGreater(served.decode_tokens, 0)
        self.assertEqual(served.admit_wait_ms_mean, 0.0)   # a free resident
        # the client end counts nothing: another partition's load is not this
        # host's, and the pool object never sees a token
        self.assertEqual(client.meter.drain(1.0).requests, 0)
        self.assertEqual(remote.meter.requests, 0)


# ---------------------------------------------------------------------------
# the observer's reading
# ---------------------------------------------------------------------------

TRAFFIC_JOURNAL = [
    {"event": "host-up", "t": 100.0, "engines": ["Qwen/Qwen3-0.6B"]},
    {"event": "traffic", "t": 110.0, "window_s": 10.0, "prefill_tokens": 200,
     "decode_tokens": 50, "requests": 4, "ttft_ms_mean": 25.0,
     "admit_wait_ms_mean": 3.0, "admit_wait_ms_max": 9.0, "inflight": 2},
    {"event": "traffic", "t": 120.0, "window_s": 10.0, "prefill_tokens": 0,
     "decode_tokens": 0, "requests": 0, "ttft_ms_mean": None,
     "admit_wait_ms_mean": None, "admit_wait_ms_max": None, "inflight": 0},
    {"event": "update", "t": 121.0, "run_id": "aaa", "update": 1,
     "seconds": 2.0,
     "phases": {"collect": 0.5, "post": 0.25, "train": 1.0, "seal": 0.25}},
]


class TrafficChannelTest(unittest.TestCase):
    """The six channels, by name and by arithmetic."""

    def test_the_six_channels_are_rates_levels_and_the_gauge(self) -> None:
        channels = {c["key"]: c["points"] for c in traffic_channels(TRAFFIC_JOURNAL)}
        self.assertEqual(list(channels), ["prefill_tok_s", "decode_tok_s",
                                          "requests_s", "ttft_ms",
                                          "admit_wait_ms", "inflight"])
        self.assertEqual(channels["prefill_tok_s"], [[110.0, 20.0], [120.0, 0.0]])
        self.assertEqual(channels["decode_tok_s"], [[110.0, 5.0], [120.0, 0.0]])
        self.assertEqual(channels["requests_s"], [[110.0, 0.4], [120.0, 0.0]])
        self.assertEqual(channels["inflight"], [[110.0, 2.0], [120.0, 0.0]])

    def test_a_null_mean_is_no_point_not_a_zero(self) -> None:
        channels = {c["key"]: c["points"] for c in traffic_channels(TRAFFIC_JOURNAL)}
        self.assertEqual(channels["ttft_ms"], [[110.0, 25.0]])
        self.assertEqual(channels["admit_wait_ms"], [[110.0, 3.0]])

    def test_a_host_that_never_journaled_a_window_has_no_channels(self) -> None:
        self.assertEqual(traffic_channels(TRAFFIC_JOURNAL[:1]), [])

    def test_the_named_readings_claim_the_new_events(self) -> None:
        """traffic and update are READ, so their fields never double as
        generic <event>.<field> series in the open slot."""
        keys = {c["key"] for c in metric_series(TRAFFIC_JOURNAL)}
        for claimed in ("traffic.decode_tokens", "traffic.window_s",
                        "traffic.inflight", "update.seconds",
                        "update.phases.train"):
            self.assertNotIn(claimed, keys)
        self.assertEqual(keys, {"prefill_tok_s", "decode_tok_s", "requests_s",
                                "ttft_ms", "admit_wait_ms", "inflight"})


class RunTimingTest(unittest.TestCase):
    """One run's update clock, joined across the hosts it ran on."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)

    def test_the_rows_come_back_raw_and_sorted(self) -> None:
        for event in TRAFFIC_JOURNAL:
            self.store.append_host_event("l4-a", event)
        self.store.append_host_event("l4-b", {
            "event": "update", "t": 90.0, "run_id": "aaa", "update": 2,
            "seconds": 1.0,
            "phases": {"collect": 0.1, "post": 0.2, "train": 0.6, "seal": 0.1}})
        self.store.append_host_event("l4-b", {
            "event": "update", "t": 95.0, "run_id": "bbb", "update": 1,
            "seconds": 5.0, "phases": {"train": 5.0}})

        timing = run_timing(self.store, "aaa")
        self.assertEqual([row["update"] for row in timing["updates"]], [1, 2])
        self.assertEqual(timing["updates"][0],
                         {"update": 1, "t": 121.0, "seconds": 2.0,
                          "phases": {"collect": 0.5, "post": 0.25,
                                     "train": 1.0, "seal": 0.25}})
        # a phase the emitter omitted reads as 0.0, never as missing
        self.assertEqual(run_timing(self.store, "bbb")["updates"][0]["phases"],
                         {"collect": 0.0, "post": 0.0, "train": 5.0, "seal": 0.0})

    def test_a_run_with_no_journalled_updates_is_empty(self) -> None:
        self.assertEqual(run_timing(self.store, "nope"), {"updates": []})

    def test_a_real_run_is_readable_end_to_end(self) -> None:
        """The whole plane, journal to reading: submit on a host, tick once,
        and the observer renders both new events off the store alone."""
        store, train, _ = arith_store(self.store.root)
        host = Host("l4-live", engines=(FakeEngine(),), learner=FakeLearner(),
                    store=store, sampler=lambda: None)
        report = go(host.submit(arith_spec(train), SCHEMA))
        go(one_tick(host))

        page = host_series([store], "l4-live")
        channels = {c["key"]: c["points"] for c in page["metrics"]}
        self.assertEqual(len(channels["decode_tok_s"]), 1)
        self.assertGreater(channels["decode_tok_s"][0][1], 0.0)
        timing = run_timing(store, report.run_id)
        self.assertEqual([row["update"] for row in timing["updates"]],
                         [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
