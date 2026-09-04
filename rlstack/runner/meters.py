"""The emission plane: what a host measures about its own load, and where it
may put it.

Every number here is taken at a seam the runner already owns — the engine's own
token loop, the arbiter's admission door, the Trainer's phase boundaries — and
never inside vLLM or torch, whose internals are version-coupled and not ours.
Every number here is also WALL CLOCK, which decides where it may land: a run
directory is a pure function of (spec, code, data) and resume-equivalence is
byte-identical run dirs, so a duration may never enter a run dir, a sealed
artifact or the ledger. The one home for these numbers is the host journal
(hosts/<name>/log.jsonl) — observability, outside identity, never read by
correctness.

Two records, two journal events:

    TrafficMeter   counters the engines and the arbiter add into as work
                   happens, DRAINED once per stats tick into one `traffic`
                   event — a window, never a row per request.
    UpdateClock    the four wall-clock phases of ONE update (collect / post /
                   train / seal), journaled as one `update` event by the
                   Trainer through the HostJournal its host handed down.
"""

from __future__ import annotations

from collections.abc import Sequence

import time
from dataclasses import dataclass

from rlstack.data.stores.base import Store


# ---------------------------------------------------------------------------
# the traffic window
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrafficWindow:
    """One drained window: what this host's metal served since the previous
    tick. Counts are sums over the window, `inflight` is a GAUGE read at the
    drain instant, and a mean is None when nothing measurable happened — a
    silent window says zero honestly rather than plotting a fabricated 0ms."""

    window_s: float
    prefill_tokens: int
    decode_tokens: int
    requests: int
    ttft_ms_mean: float | None
    admit_wait_ms_mean: float | None
    admit_wait_ms_max: float | None
    inflight: int

    def row(self) -> dict:
        """The window as a JSON row — ONE home for the shape the `traffic`
        event journals and the observer's metric_series reads back. The host
        adds "event" and "t"; everything else the window owns."""
        return {"window_s": self.window_s,
                "prefill_tokens": self.prefill_tokens,
                "decode_tokens": self.decode_tokens,
                "requests": self.requests,
                "ttft_ms_mean": self.ttft_ms_mean,
                "admit_wait_ms_mean": self.admit_wait_ms_mean,
                "admit_wait_ms_max": self.admit_wait_ms_max,
                "inflight": self.inflight}


def merged_traffic(own: dict, residents: Sequence[dict]) -> dict:
    """ONE partition's window from the host's door plus its residents' meters.

    Token and request counts SUM — each resident counted what its own engine
    served, and the door counted nothing of that. The first-token mean is
    weighted by each window's requests, the closest thing a row carries to
    its sample count. The door's own gauges — admission waits and inflight —
    stay the host's, because the door is the host's and no resident stands
    behind a second one. A resident that answered nothing (no meter) adds
    nothing."""
    out = dict(own)
    weighted, weight = 0.0, 0
    for row in (own, *residents):
        mean, requests = row.get("ttft_ms_mean"), int(row.get("requests") or 0)
        if mean is not None and requests:
            weighted += mean * requests
            weight += requests
    for row in residents:
        for field in ("prefill_tokens", "decode_tokens", "requests"):
            out[field] = int(out.get(field) or 0) + int(row.get(field) or 0)
    out["ttft_ms_mean"] = weighted / weight if weight else None
    return out


class TrafficMeter:
    """The host's own load, counted at the seams the host owns.

    THE RULE IS THE WINDOW: engines and the arbiter add into these counters as
    work happens, and the host's existing stats loop DRAINS them once per tick
    into one `traffic` event. Never a journal row per request — a busy host
    would bloat the volume with the very thing it is trying to describe — and
    never a second timer, because two sampling cadences on one host cannot be
    read against each other.

    One meter per host, shared by its engines and its arbiter, so a `traffic`
    event describes the PARTITION rather than any one tenant (a tenant's share
    of a shared engine is a run fact, and the ledger already carries it).
    Updates are plain int/float adds on the one event loop that runs the
    engines: no lock, and nothing allocated per token.
    """

    def __init__(self, started: float | None = None) -> None:
        self.started = time.time() if started is None else started
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.requests = 0
        self.ttft_ms_total = 0.0
        self.ttft_samples = 0
        self.admit_wait_ms_total = 0.0
        self.admit_wait_ms_max = 0.0
        self.admit_waits = 0
        self.inflight = 0

    # ---- what the engine counts ---------------------------------------------

    def opened_request(self, prefill_tokens: int) -> None:
        """One request entered an engine with a prompt of KNOWN length —
        len(prompt_ids) before generate(), or the whole scored prefill for
        score_tokens, which is one prefill and no decode. Both are requests:
        the door does not care which verb walked through it."""
        self.requests += 1
        self.prefill_tokens += prefill_tokens

    def first_token_after(self, seconds: float) -> None:
        """Time to first token: the wall-clock gap from the request leaving
        the seam to the first output coming back. Only a sampling request has
        one; a scoring prefill contributes none, which is why the mean is over
        its own sample count and not over requests."""
        self.ttft_ms_total += seconds * 1000.0
        self.ttft_samples += 1

    def decoded(self, tokens: int) -> None:
        """Tokens streamed out of a generate loop, counted where the loop
        already walks them."""
        self.decode_tokens += tokens

    # ---- what the arbiter counts --------------------------------------------

    def admitted(self, waited_seconds: float) -> None:
        """One unit of work cleared this host's admission door after waiting
        this long — engine traffic and gradient steps alike, because the wait
        is a property of the door, not of the resident behind it. A free
        resident never waits, so it contributes an honest zero."""
        self.admit_wait_ms_total += waited_seconds * 1000.0
        self.admit_wait_ms_max = max(self.admit_wait_ms_max,
                                     waited_seconds * 1000.0)
        self.admit_waits += 1
        self.inflight += 1

    def released(self) -> None:
        """That work left the metal. `inflight` is the only gauge here: it
        survives the drain, because "how much is running right now" is a
        state, not a window sum."""
        self.inflight -= 1

    # ---- what the host drains -----------------------------------------------

    def drain(self, now: float) -> TrafficWindow:
        """Take the window and reset the counters — called once per stats
        tick, by the host, and by nobody else. A tick with no traffic still
        drains: zeros and Nones ARE the measurement, and a hole in the series
        would be indistinguishable from a dead host."""
        window = TrafficWindow(
            window_s=now - self.started,
            prefill_tokens=self.prefill_tokens,
            decode_tokens=self.decode_tokens,
            requests=self.requests,
            ttft_ms_mean=(self.ttft_ms_total / self.ttft_samples
                          if self.ttft_samples else None),
            admit_wait_ms_mean=(self.admit_wait_ms_total / self.admit_waits
                                if self.admit_waits else None),
            admit_wait_ms_max=(self.admit_wait_ms_max
                               if self.admit_waits else None),
            inflight=self.inflight)
        self.started = now
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.requests = 0
        self.ttft_ms_total = 0.0
        self.ttft_samples = 0
        self.admit_wait_ms_total = 0.0
        self.admit_wait_ms_max = 0.0
        self.admit_waits = 0
        return window


# ---------------------------------------------------------------------------
# one update's wall clock
# ---------------------------------------------------------------------------

class UpdateClock:
    """The wall-clock shape of ONE update, measured at the Trainer's own phase
    boundaries — the calls it already makes, in the order it already makes
    them: collect (await the wave) → post (the pipeline, postdata written) →
    train (flatten, pack, the gradient, emit) → seal (blobs, bundle, the
    ledger line). Four laps, one named method per boundary, so every second
    between the update's first await and its commit is attributed to exactly
    one phase and `seconds` is their exact sum.

    A phase that does not apply stays 0.0 — it was never lapped.
    """

    def __init__(self, started: float | None = None) -> None:
        self.started = time.time() if started is None else started
        self.mark = self.started
        self.collect = 0.0
        self.post = 0.0
        self.train = 0.0
        self.seal = 0.0

    def collected(self) -> None:
        """The wave's rows exist (the feed produced them, or the generator
        did) — waiting for data is the first phase, not free time."""
        self.collect = self._lap()

    def posted(self) -> None:
        """The post pipeline ran and its postdata is written: every judge and
        teacher call this update paid for."""
        self.post = self._lap()

    def trained(self) -> None:
        """Flatten through optim_step and emit — the gradient itself, plus
        the packing that feeds it."""
        self.train = self._lap()

    def sealed(self) -> None:
        """Blobs written, bundle registered, LEDGER APPENDED: the commit
        point, and the moment the update becomes a fact."""
        self.seal = self._lap()

    def seconds(self) -> float:
        """The whole update, first await to commit."""
        return self.mark - self.started

    def row(self, run_id: str, update: int) -> dict:
        """The update as its journal event — ONE home for the shape. `t` is
        the last lap's instant: an update event is journaled when the update
        is already true."""
        return {"event": "update", "t": self.mark, "run_id": run_id,
                "update": update, "seconds": self.seconds(),
                "phases": {"collect": self.collect, "post": self.post,
                           "train": self.train, "seal": self.seal}}

    def _lap(self) -> float:
        now = time.time()
        elapsed = now - self.mark
        self.mark = now
        return elapsed


# ---------------------------------------------------------------------------
# the write door
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HostJournal:
    """A tenant's write door onto the journal of the host it landed on: the
    store that journals that host, and the host's name.

    Handed down from Host.submit, because a daemon knows its run and its own
    phases but not which metal it is running on — and a raw run_experiment
    (no host, no journal) simply has None and emits nothing. Observability
    only: nothing written through this door is ever read back by correctness,
    and nothing written through it may exist inside a run directory.
    """

    store: Store
    host: str

    def append(self, entry: dict) -> None:
        self.store.append_host_event(self.host, entry)
