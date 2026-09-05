"""A resident is a process (ADR 0002).

Every engine and every learner a host wears is a supervised CHILD of the metal
process, born with CUDA_VISIBLE_DEVICES equal to its partition's devices and,
for a learner, the torch allocator capped at its partition's fraction — the two
knobs a substrate offers for WHERE and HOW MUCH are process-granular, so the
partition has to be a process for the books to be enforceable. The Host stays
in the metal process as the DOOR (arbiter, roster, runner, journal) and holds
proxies: RemotePool for an engine, RemoteLearner for its learner, both over the
resident's own local transport.

Four rules, one named thing each:

    BUILDING IS UNIVERSAL. Given (regime, partition, build record) the resident
    is determined on every venue: build_engine / build_learner own the class,
    the base, the width, the device and the fraction; the venue declares only
    the capacity knobs (EngineBuild / LearnerBuild) and where its store is. No
    callable ever crosses to a child.

    THE DOOR SPEAKS FRAMES. JSON-safe dicts, request-id multiplexed over one
    pipe (Q5, Q10): an engine child dispatches concurrently on its own loop
    so sampling keeps batching; a learner child answers one frame at a time,
    which is what its verbs are. `hello` (birth facts), `sleep`/`wake` (the
    alternation seam, both kinds) and `stop` are the door's own verbs; the
    Engine / Learner verbs pass through EngineService / LearnerService.

    A DEAD RESIDENT IS A DEAD HOST. A watcher thread waits on the child's
    sentinel; if it exits unbidden, the metal is told and decarves the host
    (Q7). Nothing is
    restarted in place — the tenant comes back through resubmit and recarve,
    so the desk and the observer see what happened.

    ENDING IS A LADDER. stop frame → SIGTERM → SIGKILL, each rung joining what
    it signalled, the whole thing bounded — lifted from the rank chorus (#53),
    which now imports it from here (Q9).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import multiprocessing
import multiprocessing.connection
import os
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from multiprocessing.process import BaseProcess
from typing import Any

from rlstack.data.stores.address import open_store
from rlstack.data.stores.base import Store, StoreAddress
from rlstack.runner.host import Partition, Regime
from rlstack.runner.interfaces import Engine, Learner
from rlstack.runner.remote import (
    EngineService, LearnerService, Transport, check_epoch, json_roundtrip,
)


class ResidentError(RuntimeError):
    """A resident that could not be born, refused its partition, or whose
    door closed under a frame."""


# ---------------------------------------------------------------------------
# the build records: what a venue declares, and nothing placement-shaped
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineBuild:
    """The capacity knobs of a vLLM engine resident — everything a partition
    cannot tell you. base / tp / fraction / device are NOT here: they follow
    from (regime, partition)."""

    max_model_len: int = 4096
    max_bundles: int = 32
    max_rank: int = 16
    max_members: int = 0
    serves: tuple[str, ...] = ("lora",)
    enable_sleep_mode: bool = False
    enforce_eager: bool = True


@dataclass(frozen=True)
class LearnerBuild:
    """The knobs of a torch learner resident. fsdp / device / cap follow from
    (regime, partition)."""

    dtype: str = "bfloat16"
    grad_clip: float = 1.0
    checkpoint_activations: bool = True


@dataclass(frozen=True)
class FakeEngineBuild:
    """A FakeEngine resident — the fakes suite's engine, in a real process
    when a test asks for one."""

    p_correct: float = 0.5
    record_draws: bool = False
    record_latent: bool = False
    sleeps: bool = False


@dataclass(frozen=True)
class FakeLearnerBuild:
    sleeps: bool = False


Build = EngineBuild | LearnerBuild | FakeEngineBuild | FakeLearnerBuild
BUILD_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (EngineBuild, LearnerBuild, FakeEngineBuild, FakeLearnerBuild)}


def encode_build(build: Build) -> dict:
    """A build record as a tagged JSON row; the tag is the closed table's key."""
    row: dict[str, Any] = {"type": type(build).__name__}
    for f in fields(build):
        value = getattr(build, f.name)
        row[f.name] = list(value) if isinstance(value, tuple) else value
    return row


def decode_build(row: dict) -> Build:
    tag = row["type"]
    if tag not in BUILD_TYPES:
        raise ResidentError(
            f"unknown build record {tag!r}: the table holds "
            f"{sorted(BUILD_TYPES)}, and an untabled tag is refused")
    cls = BUILD_TYPES[tag]
    kwargs = {f.name: row[f.name] for f in fields(cls) if f.name in row}
    if "serves" in kwargs:
        kwargs["serves"] = tuple(kwargs["serves"])
    return cls(**kwargs)


@dataclass(frozen=True)
class Builds:
    """A metal's recipe: how it builds an engine and how it builds a learner.
    Declared at bring-up from the deploy's own constants, so a restarted
    container carves the same residents unattended; journaled on the `metal`
    registration and every `host-up` (ADR 0002, Q4a)."""

    engine: EngineBuild | FakeEngineBuild
    learner: LearnerBuild | FakeLearnerBuild

    def for_regime(self, regime: Regime) -> Build:
        return self.engine if regime.capability == "inference" else self.learner

    def row(self) -> dict:
        return {"engine": encode_build(self.engine),
                "learner": encode_build(self.learner)}

    @classmethod
    def from_row(cls, row: dict) -> "Builds":
        return cls(engine=decode_build(row["engine"]),
                   learner=decode_build(row["learner"]))

    @classmethod
    def fakes(cls, *, engine_sleeps: bool = False,
              learner_sleeps: bool = False) -> "Builds":
        return cls(engine=FakeEngineBuild(sleeps=engine_sleeps),
                   learner=FakeLearnerBuild(sleeps=learner_sleeps))


# ---------------------------------------------------------------------------
# the birth: what a child is told, and nothing else
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidentBirth:
    """Everything a child needs to become a resident, as values: its label
    (`<host>:<regime>`), the partition to pin and cap to, the regime to wear,
    the build record, and the store to reopen. JSON-safe by construction
    because it crosses a spawn."""

    label: str
    partition: Partition
    regime: Regime
    build: Build
    store: StoreAddress
    # THE INSTANCE THIS RESIDENT IS (ADR 0008, F2): its host's epoch, which is
    # its metal container's. A resident's door refuses a frame addressed to
    # any other, so a proxy held past a decarve fails by name instead of
    # reaching whatever was born at the same label afterwards.
    epoch: str = ""

    def row(self) -> dict:
        return {"label": self.label, "partition": self.partition.row(),
                "regime": {"name": self.regime.name,
                           "capability": self.regime.capability,
                           "base": self.regime.base,
                           "shape": self.regime.shape},
                "build": encode_build(self.build), "store": self.store.row(),
                "epoch": self.epoch}

    @classmethod
    def from_row(cls, row: dict) -> "ResidentBirth":
        p, r = row["partition"], row["regime"]
        return cls(
            label=row["label"],
            partition=Partition(p["metal"], tuple(p["devices"]), p["memory"],
                                p.get("gpu", "")),
            regime=Regime(r["name"], r["capability"], r["base"], int(r["shape"])),
            build=decode_build(row["build"]),
            store=StoreAddress.from_row(row["store"]),
            epoch=row.get("epoch", ""))


# ---------------------------------------------------------------------------
# building is universal
# ---------------------------------------------------------------------------

def build_engine(regime: Regime, partition: Partition, build: Build,
                 store: Store) -> Engine:
    """THE engine builder: the class, base, width and fraction follow from
    (regime, partition); the record supplies the capacity knobs. Runs inside
    the child, after the pin, so vLLM sees exactly the partition's devices."""
    if isinstance(build, FakeEngineBuild):
        from rlstack.runner.fakes import FakeEngine

        return FakeEngine(p_correct=build.p_correct,
                          record_draws=build.record_draws,
                          record_latent=build.record_latent,
                          base=regime.base, tp=regime.shape, sleeps=build.sleeps)
    if isinstance(build, EngineBuild):
        from rlstack.runner.engines.vllm_engine import VllmEngine

        return VllmEngine(
            regime.base, tp=regime.shape,
            gpu_memory_utilization=partition.memory,
            max_model_len=build.max_model_len, max_bundles=build.max_bundles,
            max_rank=build.max_rank, max_members=build.max_members,
            cas_get=store.cas_get, serves=build.serves,
            enable_sleep_mode=build.enable_sleep_mode,
            enforce_eager=build.enforce_eager)
    raise ResidentError(
        f"{type(build).__name__} is not an engine build; regime "
        f"{regime.name!r} is inference")


def build_learner(regime: Regime, partition: Partition, build: Build) -> Learner:
    """THE learner builder: shape 1 is a TorchLearner on cuda:0 — which IS the
    partition's first device once CUDA_VISIBLE_DEVICES is pinned — and shape
    n leads a chorus of n, every follower capped at the partition's fraction
    on its own device."""
    if isinstance(build, FakeLearnerBuild):
        from rlstack.runner.fakes import FakeLearner

        return FakeLearner(fsdp=regime.shape, sleeps=build.sleeps)
    if isinstance(build, LearnerBuild):
        import torch

        dtype = getattr(torch, build.dtype)
        if regime.shape == 1:
            from rlstack.runner.learners.torch_learner import TorchLearner

            return TorchLearner(device="cuda:0", dtype=dtype,
                                grad_clip=build.grad_clip,
                                checkpoint_activations=build.checkpoint_activations)
        from rlstack.runner.learners.fsdp_torch import lead_fsdp_learner

        return lead_fsdp_learner(
            regime.shape, dtype=dtype, grad_clip=build.grad_clip,
            checkpoint_activations=build.checkpoint_activations,
            memory_fraction=partition.memory)
    raise ResidentError(
        f"{type(build).__name__} is not a learner build; regime "
        f"{regime.name!r} is training")


def is_real(build: Build) -> bool:
    """Does this build touch a GPU? Decides whether the child measures its
    devices and caps its allocator — a fake in a real process does neither,
    and never imports torch."""
    return isinstance(build, (EngineBuild, LearnerBuild))


# ---------------------------------------------------------------------------
# the pin, the cap, the measurement — process facts, set before CUDA exists
# ---------------------------------------------------------------------------

def pin_devices(devices: Sequence[int]) -> None:
    """CUDA_VISIBLE_DEVICES = the partition's devices, in order. Read once at
    CUDA init and global to the process — which is the whole reason a
    resident is a process. Must run before torch is imported."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)


def cap_memory(fraction: float, devices: Sequence[int] | None = None) -> None:
    """torch's per-process allocator cap, at the partition's fraction, on
    every visible device (or the given local indices). An allocation past it
    is an OOM in THIS process and never a neighbor's. No-op off the metal."""
    import torch

    if not torch.cuda.is_available():
        return
    for device in (devices if devices is not None
                   else range(torch.cuda.device_count())):
        torch.cuda.set_per_process_memory_fraction(min(fraction, 1.0), device)


def devices_seen() -> int | None:
    """How many devices this process can see — None where CUDA is absent
    (fakes, laptops, CPU CI), so the metal knows measured from unmeasured."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.device_count()


def check_devices_seen(hello: dict, partition: Partition) -> None:
    """The pin, asserted: a measured device count that is not the partition's
    is a resident on the wrong metal, refused before it is ever listed."""
    seen = hello.get("devices_seen")
    if seen is not None and seen != len(partition.devices):
        raise ResidentError(
            f"resident {hello.get('label')!r} sees {seen} device(s) but its "
            f"partition is {list(partition.devices)}: the pin did not take")


def hello_of(birth: ResidentBirth, obj: Engine | Learner,
             seen: int | None) -> dict:
    """The birth facts a resident reports through its door, once, first."""
    row: dict[str, Any] = {
        "label": birth.label, "kind": birth.regime.capability,
        "pid": os.getpid(), "sleeps": bool(obj.sleeps),
        "epoch": birth.epoch,
        "devices_seen": seen, "devices": list(birth.partition.devices),
        "memory": birth.partition.memory}
    if birth.regime.capability == "inference":
        row["base"] = obj.base
        row["tp"] = obj.tp
    else:
        row["fsdp"] = obj.fsdp
        # why a learner will not sleep, when it will not: at fsdp > 1 that is
        # a fact about the pinned torch, not about the width (#82), and the
        # only place it is ever visible is this frame.
        row["sleep_refusal"] = obj.sleep_refusal
    return row


# ---------------------------------------------------------------------------
# the door: door verbs plus the object's own service, one dispatch
# ---------------------------------------------------------------------------

class Door:
    """The resident-side dispatch: `sleep`/`wake` are the door's (the
    alternation seam, both kinds); `hello` and `heartbeat` are answered from
    the birth facts and the clock; everything else is the object's own
    service. `stop` is the serving loop's, because it ends the loop.

    THE EPOCH IS CHECKED HERE (ADR 0008, F2) and not inside EngineService or
    LearnerService: the Door is the resident's WHOLE door — it owns sleep,
    wake and hello, which reach no service at all — and the services are also
    constructed inline by `HostService`, which has already checked at its own
    door. One receiver, one check."""

    def __init__(self, obj: Engine | Learner,
                 service: EngineService | LearnerService, hello: dict) -> None:
        self.obj = obj
        self.service = service
        self.hello = hello
        self.epoch = str(hello.get("epoch") or "")

    async def call(self, verb: str, payload: dict) -> dict:
        check_epoch(payload, self.epoch, f"resident {self.hello['label']!r}")
        if verb == "sleep":
            self.check_sleeps(verb)
            await self.obj.sleep()
            return {}
        if verb == "wake":
            self.check_sleeps(verb)
            await self.obj.wake()
            return {}
        return await self.service.serve(verb, payload)

    def answer(self, verb: str, payload: dict) -> dict:
        check_epoch(payload, self.epoch, f"resident {self.hello['label']!r}")
        if verb == "hello":
            return dict(self.hello)
        if verb == "heartbeat":
            return self.beat()
        return self.service.answer(verb, payload)

    def beat(self) -> dict:
        """THE RESIDENT'S PULSE (ADR 0008, F1): I am here, at this instant,
        and I am this instance. Answered on the ADMISSION-FREE path so it
        costs the resident nothing but a dispatch — and so a resident wedged
        inside one long synchronous verb cannot answer it, which is exactly
        the reading the host's watchdog wants: silence past a phase's known
        bound is a stall, not a pause."""
        return {"t": time.time(), "label": self.hello["label"],
                "epoch": self.epoch}

    def check_sleeps(self, verb: str) -> None:
        if not self.hello["sleeps"]:
            raise ResidentError(
                f"resident {self.hello['label']!r} cannot {verb}: its build "
                f"reported sleeps=false, so no host should have wired the hook")


class DoorTransport:
    """An IN-PROCESS resident's door: the same frames, JSON round-tripped
    both ways, no child. What tests and hand-built hosts use, and what makes
    the process-backed path byte-identical to it by construction."""

    def __init__(self, door: Door) -> None:
        self.door = door

    async def call(self, verb: str, payload: dict) -> dict:
        return json_roundtrip(await self.door.call(verb, json_roundtrip(payload)))

    def ask(self, verb: str, payload: dict) -> dict:
        if verb == "stop":
            self.door.obj.shutdown()
            return {}
        return json_roundtrip(self.door.answer(verb, json_roundtrip(payload)))


# ---------------------------------------------------------------------------
# the frames: request-id multiplexed JSON over one pipe
# ---------------------------------------------------------------------------

HELLO_ID = 0
"""The child's first frame is unsolicited: its hello, or the build's failure."""


def _send_frame(conn, lock: threading.Lock, frame: dict) -> None:
    with lock:
        conn.send_bytes(json.dumps(frame).encode("utf-8"))


class _Waiter:
    """One outstanding frame's reply slot — sync (an Event) or async (a
    Future on its loop); the reader thread delivers to either."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.loop = loop
        self.future = loop.create_future() if loop is not None else None
        self.event = threading.Event()
        self.frame: dict | None = None

    def deliver(self, frame: dict) -> None:
        self.frame = frame
        if self.future is not None:
            def settle() -> None:
                if not self.future.done():
                    self.future.set_result(frame)
            self.loop.call_soon_threadsafe(settle)
        self.event.set()


class PipeTransport:
    """The metal process's end of a resident's door.

    `ask` blocks its caller for one round trip (the in-process semantics it
    replaces — a learner's forward_backward blocked the loop before, and
    blocks it now, Q6); `call` awaits a Future. Both may be outstanding at
    once — a Trainer's `tokenize` while a Generator's `sample_tokens` is in
    flight — so ONE reader thread dispatches replies by request id and the
    child answers in whatever order it finishes. When the pipe closes every
    waiter is failed with the same word: the resident is gone."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._send_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, _Waiter] = {HELLO_ID: _Waiter()}
        self._ids = itertools.count(1)
        self.closed = False
        self._reader = threading.Thread(target=self._read_forever, daemon=True,
                                        name="resident-door-reader")
        self._reader.start()

    def _read_forever(self) -> None:
        while True:
            try:
                raw = self._conn.recv_bytes()
            except (EOFError, OSError):
                break
            frame = json.loads(raw)
            with self._pending_lock:
                waiter = self._pending.pop(frame["id"], None)
            if waiter is not None:
                waiter.deliver(frame)
        self.closed = True
        with self._pending_lock:
            orphans = list(self._pending.values())
            self._pending.clear()
        for waiter in orphans:
            waiter.deliver({"ok": False, "type": "ResidentGone",
                            "error": "the resident's door closed: its process "
                                     "exited"})

    def _register(self, waiter: _Waiter) -> int:
        with self._pending_lock:
            request_id = next(self._ids)
            self._pending[request_id] = waiter
        return request_id

    def _send(self, frame: dict) -> None:
        if self.closed:
            raise ResidentError("the resident's door is closed")
        _send_frame(self._conn, self._send_lock, frame)

    def hello(self, timeout_s: float) -> dict:
        """The child's first, unsolicited frame: its birth facts, or why it
        could not be born. Bounded, because an engine boot is minutes and a
        hang is not a boot."""
        waiter = self._pending[HELLO_ID]
        if not waiter.event.wait(timeout_s):
            raise ResidentError(f"no hello from the resident within {timeout_s:g}s")
        return _unwrap(waiter.frame)

    def ask(self, verb: str, payload: dict) -> dict:
        waiter = _Waiter()
        request_id = self._register(waiter)
        self._send({"id": request_id, "path": "ask", "verb": verb,
                    "payload": payload})
        waiter.event.wait()
        return _unwrap(waiter.frame)

    async def call(self, verb: str, payload: dict) -> dict:
        waiter = _Waiter(asyncio.get_running_loop())
        request_id = self._register(waiter)
        self._send({"id": request_id, "path": "call", "verb": verb,
                    "payload": payload})
        return _unwrap(await waiter.future)

    def close(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass


def _unwrap(frame: dict) -> dict:
    if frame.get("ok"):
        return frame["reply"]
    raise ResidentError(f"{frame.get('type', 'error')}: {frame.get('error')}")


def serve_door(conn, door: Door) -> None:
    """The child's whole life after its build: frames in, replies out, until
    `stop`. Calls dispatch CONCURRENTLY on this loop (an engine keeps
    batching); asks run inline, in arrival order (a learner's verbs are
    ordered). A reader thread feeds the loop so a blocked recv never holds
    the loop, and the parent's death (EOF) ends the loop like a stop."""
    send_lock = threading.Lock()

    async def main() -> None:
        loop = asyncio.get_running_loop()
        frames: asyncio.Queue = asyncio.Queue()

        def read_forever() -> None:
            while True:
                try:
                    raw = conn.recv_bytes()
                except (EOFError, OSError):
                    loop.call_soon_threadsafe(frames.put_nowait, None)
                    return
                loop.call_soon_threadsafe(frames.put_nowait, json.loads(raw))

        threading.Thread(target=read_forever, daemon=True,
                         name="resident-door-serve").start()

        def reply(request_id: int, result: dict | None,
                  failure: BaseException | None) -> None:
            if failure is None:
                frame = {"id": request_id, "ok": True, "reply": result}
            else:
                frame = {"id": request_id, "ok": False,
                         "type": type(failure).__name__, "error": str(failure)}
            _send_frame(conn, send_lock, frame)

        async def handle_call(frame: dict) -> None:
            try:
                out = await door.call(frame["verb"], frame["payload"])
            except BaseException as failure:      # noqa: BLE001 — every failure is the caller's
                reply(frame["id"], None, failure)
                return
            reply(frame["id"], out, None)

        while True:
            frame = await frames.get()
            if frame is None:
                return                             # the parent is gone
            if frame["verb"] == "stop":
                reply(frame["id"], {}, None)
                return
            if frame["path"] == "call":
                loop.create_task(handle_call(frame))
                continue
            try:
                out = door.answer(frame["verb"], frame["payload"])
            except BaseException as failure:      # noqa: BLE001
                reply(frame["id"], None, failure)
                continue
            reply(frame["id"], out, None)

    asyncio.run(main())


def resident_main(birth_row: dict, conn) -> None:
    """The child's entry point: pin, cap, build, hello, serve, release.

    Spawned, never forked, so this module is imported fresh here and torch is
    not yet loaded when the pin lands. A build that fails sends its failure
    as the hello and exits; the metal releases the booking. Whatever ends the
    serving loop — a stop, or the parent's death — the object's `shutdown`
    runs on the way out, so a learner's chorus and an engine's core never
    outlive the resident."""
    birth = ResidentBirth.from_row(birth_row)
    pin_devices(birth.partition.devices)
    lock = threading.Lock()
    obj: Engine | Learner | None = None
    try:
        real = is_real(birth.build)
        if birth.regime.capability == "inference":
            store = open_store(birth.store)
            obj = build_engine(birth.regime, birth.partition, birth.build, store)
            service: EngineService | LearnerService = EngineService(obj)
        else:
            if real:
                cap_memory(birth.partition.memory)
            obj = build_learner(birth.regime, birth.partition, birth.build)
            service = LearnerService(obj)
        hello = hello_of(birth, obj, devices_seen() if real else None)
    except BaseException as failure:              # noqa: BLE001 — the hello IS the report
        traceback.print_exc()
        _send_frame(conn, lock, {"id": HELLO_ID, "ok": False,
                                 "type": type(failure).__name__,
                                 "error": str(failure)})
        return
    _send_frame(conn, lock, {"id": HELLO_ID, "ok": True, "reply": hello})
    try:
        serve_door(conn, Door(obj, service, hello))
    finally:
        obj.shutdown()


# ---------------------------------------------------------------------------
# the ladder: ending a process that holds a GPU, bounded (lifted from #53)
# ---------------------------------------------------------------------------

GRACE_S = 10.0
"""How long a child gets to hear the polite word and leave on its own. Short
because the whole teardown has to fit inside the venue's shutdown grace —
Modal's is 30 seconds, and what does not finish inside it is killed with its
stdout unflushed (#53)."""

SIGNAL_GRACE_S = 5.0
"""How long a signalled child gets to die before the next rung. A process that
will die at all dies immediately here; the wait is for the kernel."""


@dataclass(frozen=True)
class Teardown:
    """What ending a process group actually cost: one field per rung.

    Positions are 1-based indices into the children as given — for a rank
    chorus, position IS rank. A child that had to be killed is a fact about
    the run, and the silence that hid it is what made #53's wedge look like a
    container that died for no reason."""

    heard_the_farewell: bool = True
    deaf: tuple[int, ...] = ()      # still there after the polite word → SIGTERM
    wedged: tuple[int, ...] = ()    # still there after SIGTERM → SIGKILL
    lost: tuple[int, ...] = ()      # still there after SIGKILL

    @property
    def graceful(self) -> bool:
        return (self.heard_the_farewell
                and not (self.deaf or self.wedged or self.lost))

    def line(self) -> str:
        """The one honest line, in escalation order."""
        parts = []
        if not self.heard_the_farewell:
            parts.append("STOP never went out (the chorus was past hearing)")
        if self.deaf:
            parts.append(f"ranks {list(self.deaf)} ignored STOP → SIGTERM")
        if self.wedged:
            parts.append(f"ranks {list(self.wedged)} survived SIGTERM → SIGKILL")
        if self.lost:
            parts.append(f"ranks {list(self.lost)} SURVIVED SIGKILL")
        return "[chorus] teardown: " + "; ".join(parts)


def living(children: Sequence[BaseProcess]) -> tuple[tuple[int, BaseProcess], ...]:
    """The children still running, each with its 1-based position."""
    return tuple((index, child)
                 for index, child in enumerate(children, start=1)
                 if child.is_alive())


def join_survivors(children: Sequence[BaseProcess],
                   timeout_s: float) -> tuple[tuple[int, BaseProcess], ...]:
    """Wait out ONE shared deadline for every child, then say who is left.
    Shared, not per-child: the budget is a wall-clock promise."""
    deadline = time.monotonic() + timeout_s
    for _, child in living(children):
        child.join(timeout=max(0.0, deadline - time.monotonic()))
    return living(children)


def escalate(children: Sequence[BaseProcess], *, grace_s: float,
             signal_grace_s: float) -> Teardown:
    """The ladder, in the order a shutdown should try it: left on its own,
    SIGTERM, SIGKILL — and a record of which rung each child needed. Each
    rung JOINS what it signalled; a child that survives even the kill is
    reported rather than pretended away."""
    deaf = join_survivors(children, grace_s)
    for _, child in deaf:
        child.terminate()
    wedged = join_survivors(children, signal_grace_s) if deaf else ()
    for _, child in wedged:
        child.kill()
    lost = join_survivors(children, signal_grace_s) if wedged else ()
    return Teardown(deaf=tuple(i for i, _ in deaf),
                    wedged=tuple(i for i, _ in wedged),
                    lost=tuple(i for i, _ in lost))


# ---------------------------------------------------------------------------
# the resident, as the metal process holds it
# ---------------------------------------------------------------------------

class Resident:
    """One process wearing one regime of one host, as its parent holds it:
    the birth, the door (a Transport), the hello it reported, the process
    handle (None for an in-process resident), and the watcher that turns an
    unbidden exit into the metal's decarve."""

    def __init__(self, birth: ResidentBirth, transport: Transport, hello: dict,
                 process: BaseProcess | None) -> None:
        self.birth = birth
        self.transport = transport
        self.hello = hello
        self.process = process
        self.stopping = False

    @property
    def label(self) -> str:
        return self.birth.label

    @property
    def regime(self) -> Regime:
        return self.birth.regime

    # ---- two ways to be born ------------------------------------------------

    @classmethod
    def spawn(cls, birth: ResidentBirth, *,
              hello_timeout_s: float = 1800.0) -> "Resident":
        """A child process: spawned (never forked — it builds its own CUDA
        context), pinned before torch loads, awaited through its hello, and
        refused when the devices it sees are not its partition's. Returns
        with the door open. Not daemonic, because a learner resident has
        children of its own (the chorus) — so ending it is `stop`'s job."""
        context = multiprocessing.get_context("spawn")
        parent_end, child_end = context.Pipe()
        process = context.Process(target=resident_main,
                                  args=(birth.row(), child_end),
                                  name=birth.label)
        process.start()
        child_end.close()
        transport = PipeTransport(parent_end)
        try:
            hello = transport.hello(hello_timeout_s)
            check_devices_seen(hello, birth.partition)
        except BaseException:
            escalate((process,), grace_s=SIGNAL_GRACE_S,
                     signal_grace_s=SIGNAL_GRACE_S)
            transport.close()
            raise
        return cls(birth, transport, hello, process)

    @classmethod
    def in_process(cls, birth: ResidentBirth,
                   obj: Engine | Learner) -> "Resident":
        """An already-built object behind the same door, in this process:
        what the fakes suite drives and what makes the child path a
        transport choice rather than a semantics change."""
        service = (EngineService(obj) if birth.regime.capability == "inference"
                   else LearnerService(obj))
        hello = hello_of(birth, obj, None)
        return cls(birth, DoorTransport(Door(obj, service, hello)), hello, None)

    # ---- what the host and the metal ask of it ------------------------------

    def alive(self) -> bool:
        return self.process is None or self.process.is_alive()

    def pid(self) -> int:
        return self.process.pid if self.process is not None else os.getpid()

    async def sleep(self) -> None:
        await self.transport.call("sleep", {})

    async def wake(self) -> None:
        await self.transport.call("wake", {})

    def heartbeat(self) -> dict:
        """One round trip to the child and back — the host's watchdog asks
        this every RESIDENT_HEARTBEAT_S (ADR 0008, F1). Synchronous, on the
        ask path, because the point is to learn whether the child's own loop
        is still turning; the host runs it off its loop, at most one in
        flight, so a resident that never answers holds up nothing but its own
        verdict."""
        return self.transport.ask("heartbeat", {})

    def row(self) -> dict:
        """The resident as status() and describe() report it, and as the
        host-up event journals it (label + pid)."""
        return {"label": self.label, "kind": self.regime.capability,
                "pid": self.pid(), "alive": self.alive(),
                "devices": list(self.birth.partition.devices),
                "memory": self.birth.partition.memory,
                "sleeps": bool(self.hello.get("sleeps", False))}

    def watch(self, on_exit: Callable[["Resident"], None]) -> None:
        """Start the watcher: a thread on the child's join that reports an
        exit nobody asked for. An in-process resident has no exit to watch.
        Started by the metal once the host is in its books, so the report
        always finds a host to decarve — and a child already dead by then is
        reported at once."""
        if self.process is None:
            return

        def wait_for_exit() -> None:
            # the SENTINEL, never join(): join reaps, and two threads reaping
            # one child leaves the loser reading ECHILD as "still alive" for
            # the rest of the ladder. Waiting on the fd observes the exit and
            # leaves the reap to whoever asks is_alive() next, on one thread.
            multiprocessing.connection.wait([self.process.sentinel])
            if not self.stopping:
                on_exit(self)

        threading.Thread(target=wait_for_exit, daemon=True,
                         name=f"resident-watch:{self.label}").start()

    def stop(self, *, grace_s: float = GRACE_S,
             signal_grace_s: float = SIGNAL_GRACE_S) -> Teardown:
        """End the resident: the stop frame (the child releases its object —
        a learner's own chorus ladder runs inside this grace — and exits),
        then SIGTERM, then SIGKILL, each rung joining what it signalled.
        Idempotent; bounded by grace_s + 2 x signal_grace_s."""
        self.stopping = True
        if self.process is None:
            self.transport.ask("stop", {})
            return Teardown()
        if not self.process.is_alive():
            return Teardown()
        heard = self._say_stop(grace_s)
        teardown = escalate((self.process,), grace_s=grace_s if heard else 0.0,
                            signal_grace_s=signal_grace_s)
        self.transport.close()
        return replace(teardown, heard_the_farewell=heard)

    def _say_stop(self, timeout_s: float) -> bool:
        """The polite word, on a thread with a deadline: a child wedged in a
        collective never answers, and the ladder below is what ends it."""
        spoken: list[bool] = []

        def say_it() -> None:
            try:
                self.transport.ask("stop", {})
                spoken.append(True)
            except Exception:       # noqa: BLE001 — past hearing is the verdict
                pass

        thread = threading.Thread(target=say_it, daemon=True)
        thread.start()
        thread.join(timeout_s)
        return bool(spoken)
