"""The observer UI beside the store volume.

    modal deploy deploy/ui.py          # -> https://<workspace>--rlstack-ui.modal.run

Deployment only (I5): an image, a volume MOUNT, and the SAME stdlib WSGI app
`python -m rlstack ui <store-root>` serves locally, over the SAME `LocalStore`
— because a mounted volume IS a POSIX tree, and a store over one is
`data/stores/local.py` with a locator (ADR 0007, Q7). The second Modal store
backend that used to live in this file is retired: a complete `Store`
implementation outside `data/stores/`, outside `open_store` and outside the
fakes suite is a backend nothing tests.

THE ONE VENUE-SPECIFIC THING is freshness. Readers of a Modal volume see a
snapshot until they reload; the store deliberately knows nothing about that
(it commits on write and never reads stale — only an OUTSIDE observer can).

AND THE ONE MEASURED TRAP, kept because the reasoning is the artifact: the
retired SDK-backed store was written after a mount was observed to misbehave
under a RELOAD TAKEN MID-SCAN — runs hopped root -> subdir on click-out, and
reads queued behind a slow reload until /api/runs timed out at 60 s, both
live. So the reload here is taken at ONE point only: just before a cache entry
is rebuilt, under the same lock that already serializes rebuilds, and never
while a response is being assembled. A page therefore reads one consistent
snapshot, at most TTL seconds old.

Deployed 2026-09-04 on the yu-masala workspace over the mount: pages read
fresh and nothing hopped. If the observer ever reads stale, the reload
placement here is the first thing to look at, and the SDK-backed reader is in
git at 90bbc8e.

THE DESK IS HANDED IN as the observer's pulse (`RemoteDesk.pulse`): one
status probe per listing, alive plus the ROSTER, so a run the journal leaves
open on a live host that does not carry it reads LOST rather than running — a
carve name recurs per container generation and a killed container writes no
detach, so the journal alone cannot tell (found on the yu-masala volume). A
workspace with no desk deployed is a journal-only day: the probe fails and
the heartbeats stand.
"""

import modal

from modal_venue import desk

# the one workspace rule, stated in the chassis and checked here too because
# this file stands alone (it ships no chassis into its image)
from modal_venue import require_workspace   # noqa: E402

require_workspace()

app = modal.App("rlstack-ui")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # copy=True on BOTH: add_local_python_source ships only .py files, so
    # the web assets ride as their own layer — and a runtime MOUNT of the
    # package tree would shadow any dir added under it, so both go in as
    # image layers, assets after sources (later layer wins). Measured both
    # ways: without the assets every page is a 200 with no body; with a
    # runtime mount the asset route 404s. `modal_venue` rides for `desk()`.
    .add_local_python_source("rlstack", "modal_venue", copy=True)
    .add_local_dir("rlstack/observe/web", "/root/rlstack/observe/web",
                   copy=True)
)

STORE_MOUNT = "/store"
STORE = "modal://rlstack-store"
RELOAD_EVERY_S = 12.0
"""Seconds between reloads of the mount: how stale the snapshot a scan reads
may be. The design's freshness bound, unchanged."""

TTL = 30.0
"""Seconds a GET's response is served from memory. One reader pays the walk;
every poll inside the window answers instantly (measured before the rewrite:
17 s -> instant). LONGER THAN A SWEEP PLUS ITS REST: a warmed key must be
rebuilt before it expires, or a reader meets the very miss the prewarm exists
to prevent — at 12 s against a 40 s sweep, every reader did. Freshness is
RELOAD_EVERY_S's job, not this one's: the prewarm overwrites a held response
the moment it rebuilds, whatever the response's age."""

WARM = (("/api/runs", ""),
        ("/api/hosts", "hours=24"), ("/api/fleet", "hours=24"),
        ("/api/hosts", ""), ("/api/fleet", ""))
"""The keys a reader ACTUALLY asks for, which is not the same as the routes.
nav.js's withHours() appends the window to every fleet call, so warming a
bare "/api/hosts?" warmed a key nothing requests: measured on the venue,
/api/runs (whose key did match) answered in 0.20 s while /api/hosts?hours=24
paid a 1.4 s rebuild on EVERY poll, forever. "hours=24" is the default a
fresh visitor gets; "" is the "all" window, which withHours spells as the
bare url."""

WARM_EVERY_S = 6.0
"""The FLOOR of the prewarm's rest between sweeps; rest_after() raises it to
the sweep's own length. The old rule here ("a sweep costs ~0.4 s per key")
was false on the venue — a sweep cost 15 to 48 s — so a 6 s rest had the
walker holding the store nine tenths of the time, and every reload, and so
every reader, waited on it."""


def rest_after(sweep_s: float) -> float:
    """THE WALKER HOLDS THE STORE AT MOST HALF THE TIME: it rests at least as
    long as it just walked, and never less than WARM_EVERY_S."""
    return max(WARM_EVERY_S, sweep_s)

PULSE_DEADLINE_S = 15.0
"""How long ONE probe of the desk's pulse may take before it is abandoned. The
wire's deadline (ASK_DEADLINE_S) is the fleet's; this is the page's, and
shorter, because a page is looked at."""

PULSE_EVERY_S = 10.0
PULSE_STALE_S = 90.0
"""THE PULSE IS NEVER ON A REQUEST'S PATH. One thread asks the desk every
PULSE_EVERY_S and keeps the last answer; a rebuild reads that answer in no
time at all, and an answer older than PULSE_STALE_S is withheld so the page
falls back to the journal's own heartbeat rather than show a probe that has
gone quiet. Measured before this: a rebuild paid the whole probe (15 s while a
metal was unreachable) INSIDE the rebuild lock, the prewarm re-took that lock
every 10 s, and everything else — a run's page, even a 404 — queued behind
it for 20 to 74 s."""


@app.function(image=image, volumes={STORE_MOUNT: store_volume},
              scaledown_window=300, timeout=600, max_containers=1)
@modal.concurrent(max_inputs=32)
@modal.wsgi_app(label="rlstack-ui")
def ui():
    """The CLI's own app over the mount, behind a response cache."""
    import threading
    import time

    from rlstack.data.stores.modal_volume import ModalVolumeStore
    from rlstack.observe.liveness import bounded
    from rlstack.observe.ui import ui_app

    # The store the metal writes through, read here and never written: the
    # LocalStore over the mount, named by the fleet's locator so `in_stores`
    # says modal://rlstack-store the way every host journal does. Found on
    # the venue: a bare LocalStore takes no locator, every container of the
    # first mount deploy died on this line, and Modal kept serving the old
    # SDK reader in its place — which is why the run page never learned
    # about extents.
    # the fleet's pulse, kept CURRENT by one thread and read by every
    # rebuild for free (PULSE_EVERY_S / PULSE_STALE_S); each probe is bounded
    # and runs on a thread of its own, so a metal nobody can reach costs the
    # page nothing (found live: an unscheduled class held / and /api/runs
    # past 75 s behind one probe)
    latest: dict = {"at": 0.0, "pulse": {}}
    probe = bounded(lambda: desk().pulse(), PULSE_DEADLINE_S)

    def keep_pulsing() -> None:
        while True:
            try:
                told = probe()
            except Exception:
                told = {}
            if told:
                latest.update(at=time.time(), pulse=told)
            time.sleep(PULSE_EVERY_S)

    def last_pulse() -> dict:
        fresh = time.time() - latest["at"] < PULSE_STALE_S
        return latest["pulse"] if fresh else {}

    threading.Thread(target=keep_pulsing, daemon=True).start()
    inner = ui_app([ModalVolumeStore(STORE_MOUNT, volume=store_volume,
                                     locator=STORE)],
                   desk=last_pulse)
    cache: dict[str, tuple[float, str, list, bytes]] = {}
    # ONE LOCK PER ENDPOINT: a run's page is never queued behind the hosts
    # rebuild, and the prewarm's three keys never hold anyone else's
    locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()

    def lock_for(key: str) -> threading.Lock:
        with locks_guard:
            return locks.setdefault(key, threading.Lock())

    # THE RELOAD IS EXCLUSIVE OF EVERY SCAN and taken at most once per
    # RELOAD_EVERY_S:
    # scans of different endpoints run side by side (their locks are their
    # own), but a reload waits until none is in flight and no scan starts
    # while one runs — the measured trap (a reload under a scan hopped runs
    # root -> subdir) stays closed with the global lock gone
    snapshot = threading.Condition()
    scanning = [0]
    reloaded_at = [0.0]

    def reload_then(scan):
        with snapshot:
            if time.time() - reloaded_at[0] >= RELOAD_EVERY_S:
                while scanning[0]:
                    snapshot.wait()
                try:
                    store_volume.reload()
                except Exception:
                    pass        # a reload that fails serves the snapshot we have
                reloaded_at[0] = time.time()
            scanning[0] += 1
        try:
            return scan()
        finally:
            with snapshot:
                scanning[0] -= 1
                snapshot.notify_all()

    def call_inner(key: str, environ: dict) -> tuple[str, list, bytes]:
        """One response built from a FRESHLY RELOADED mount. The reload is
        here and nowhere else: before a scan, never during one."""
        caught: dict = {}

        def catch(status, headers, exc_info=None):
            caught["status"], caught["headers"] = status, list(headers)

        body = reload_then(lambda: b"".join(inner(environ, catch)))
        cache[key] = (time.time(), caught["status"], caught["headers"], body)
        return caught["status"], caught["headers"], body

    def cached_app(environ, start_response):
        """GET responses cached for TTL seconds; non-GETs and misses pass
        through. One rebuild at a time — late arrivals reuse it."""
        key = (environ.get("PATH_INFO", "") + "?"
               + environ.get("QUERY_STRING", ""))
        # only the API is cached and serialized: the page and its assets are
        # static and must never queue behind a rebuild (found live: / hung
        # behind a prewarm holding the lock on a stalled pulse)
        if (environ.get("REQUEST_METHOD") != "GET"
                or not environ.get("PATH_INFO", "").startswith("/api/")):
            return inner(environ, start_response)
        held = cache.get(key)
        if held is not None and time.time() - held[0] < TTL:
            _, status, headers, body = held
        else:
            with lock_for(key):
                held = cache.get(key)
                if held is not None and time.time() - held[0] < TTL:
                    _, status, headers, body = held
                else:
                    status, headers, body = call_inner(key, environ)
        start_response(status, headers)
        return [body]

    def prewarm() -> None:
        """The hot endpoints never go cold: the walker pays, readers do not."""
        while True:
            began = time.time()
            for path, query in WARM:
                key = path + "?" + query
                try:
                    with lock_for(key):
                        call_inner(key, {
                            "REQUEST_METHOD": "GET", "PATH_INFO": path,
                            "QUERY_STRING": query, "SERVER_NAME": "prewarm",
                            "SERVER_PORT": "80", "wsgi.url_scheme": "http",
                            "wsgi.input": None, "wsgi.errors": None})
                except Exception:
                    pass
            time.sleep(rest_after(time.time() - began))

    threading.Thread(target=prewarm, daemon=True).start()
    return cached_app
