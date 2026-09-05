"""The wire: pools served by ANOTHER host, behind the same Engine protocol.

HostService is the host-side end — it executes pool verbs on its own metal
under its own arbiter, because admission stays with the partition, and engines
are addressed by CAPABILITY (base, tp), never by pool name. A Transport carries
JSON-safe dict frames; RemotePool implements the whole Engine protocol over it,
so the runner cannot tell remote from local. The verb split is the contract
every transport implements against:

    call (async)   sample_tokens / score_tokens, and at a HOST door every
                   learner verb — they occupy the metal, so the service wraps
                   each in the owning host's arbiter.admit.
    ask  (sync)    add_bundle / reachability / tokenize — additive
                   registration and build facts, which by the tenancy
                   invariant never disturb traffic, so they need no admission
                   and may run from sync call sites — and, at a RESIDENT's
                   door, that resident's learner verbs, which the runner
                   beside it admitted before the first frame.

THE LEARNER IS REACHED EXACTLY AS AN ENGINE IS (ADR 0006 Part A). Since ADR
0002 it is a process INSIDE its host — every resident, engine or learner, is a
child of the metal, so `EngineService`/`LearnerService` are the resident's end
of that door and `RemoteLearner` is the Host's end for its learner
(`RemotePool` already was for an engine). Since Part A its verbs also cross
the HOST door: `HostService.serve` admits `install` / `uninstall` /
`forward_backward` / `optim_step` / `emit` / `load` at the serving host's own
arbiter and executes them through that host's learner, so a run anchored
anywhere may join a standing learner and the learner's alternation is honored
where the learner lives. The two hops are unchanged: admission at the host,
then the resident.

ADDRESSES ARE READ HERE AND NOWHERE ELSE (ADR 0007). `parse_address` is the
grammar's one reader and `transport_for` the one factory: an address names its
venue (`modal://<app>/<cls>[#<host>]`) or the in-process wire (`local://<host>`),
and every resolver in the fleet — a desk's `host_for`/`metal_for`, a host's
`transport_for` — is that single function. Real substrates live one per file
under runner/transports/ and are imported inside the branch that names them;
LocalTransport stays here because it is the contract's enforcement, not a
substrate.

Costs, stated: sample replies are non-streamed (one reply carries the whole
event list), add_bundle ships payload bytes as base64, and a routed learner
puts one frame on the wire per microbatch and per update (the TokenBatch out,
the emitted payloads back). LocalTransport round-trips every frame through
json in both directions, so anything that works over it works over a real
transport.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol

from rlstack.data.flatten import TokenBatch
from rlstack.data.trajectory import Message, Role
from rlstack.policy.adapters.base import Directive, Mechanism
from rlstack.policy.compile import Bundle
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.interfaces import (
    Emitted, Engine, EntryInstall, FinishEvent, Learner, OptimSettings,
    Parameterization, TokenEvent, TrainStats,
)
from rlstack.runner.meters import TrafficMeter
from rlstack.spec.canonical import TYPE_KEY
from rlstack.spec.specs import SamplingSpec

if TYPE_CHECKING:
    from rlstack.runner.host import Host


# ---------------------------------------------------------------------------
# codecs — every record that crosses the wire, encoded to JSON-safe dicts
# ---------------------------------------------------------------------------

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def encode_messages(messages: Sequence[Message]) -> list[dict]:
    return [{"role": m.role.value, "content": m.content} for m in messages]


def decode_messages(rows: Sequence[dict]) -> tuple[Message, ...]:
    return tuple(Message(Role(r["role"]), r["content"]) for r in rows)


def encode_sampling(sampling: SamplingSpec) -> dict:
    return {"temperature": sampling.temperature, "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens}


def decode_sampling(row: dict) -> SamplingSpec:
    return SamplingSpec(**row)


def encode_events(events: Sequence[TokenEvent | FinishEvent]) -> list[dict]:
    """Extras/turn_extras must already be JSON-safe — the same discipline the
    membrane demands of them at seal time, so the wire adds no new rule."""
    out: list[dict] = []
    for event in events:
        if isinstance(event, TokenEvent):
            out.append({"kind": "token", "token_id": event.token_id,
                        "logprob": event.logprob,
                        "text_delta": event.text_delta,
                        "extras": dict(event.extras)})
        else:
            out.append({"kind": "finish", "finish": event.finish,
                        "stop_hit": event.stop_hit,
                        "turn_extras": dict(event.turn_extras)})
    return out


def decode_events(rows: Sequence[dict]) -> tuple[TokenEvent | FinishEvent, ...]:
    return tuple(
        TokenEvent(token_id=r["token_id"], logprob=r["logprob"],
                   text_delta=r["text_delta"], extras=r["extras"])
        if r["kind"] == "token"
        else FinishEvent(finish=r["finish"], stop_hit=r["stop_hit"],
                         turn_extras=r["turn_extras"])
        for r in rows)


def encode_directives(directives: Sequence[Directive]) -> list[dict]:
    """A directive crosses as its adapter type's name plus its own fields —
    the adapter type is the closed table the other end decodes against."""
    return [{"adapter_type": d.adapter_type, **asdict(d)} for d in directives]


def decode_directives(rows: Sequence[dict]) -> tuple[Directive, ...]:
    """Dispatch by adapter type: the registered class declares the record it
    accepts (AdapterType.directive), and a row naming an adapter type that
    accepts none is refused rather than guessed at."""
    out = []
    for row in rows:
        fields = dict(row)
        name = fields.pop("adapter_type")
        record = ADAPTER_TYPES.get(name).instance.directive
        if record is None:
            raise ValueError(
                f"adapter type {name!r} accepts no directive, yet one crossed "
                f"the wire addressed to it")
        out.append(record(**fields))
    return tuple(out)


def encode_bundle(bundle: Bundle) -> dict:
    return {"bundle_id": bundle.bundle_id,
            "policy_version": dict(bundle.policy_version),
            "payloads": {name: _b64(data)
                         for name, data in bundle.payloads.items()},
            "adapter_types": dict(bundle.adapter_types)}


def decode_bundle(row: dict) -> Bundle:
    return Bundle(bundle_id=row["bundle_id"],
                  policy_version=row["policy_version"],
                  payloads={name: _unb64(text)
                            for name, text in row["payloads"].items()},
                  adapter_types=row["adapter_types"])


def _spec_classes() -> dict[str, type]:
    """THE closed table of types a canonical spec tree may name.

    canonical_json tags every dataclass with its class name, so decoding is
    dispatch over this table and nothing else — an unknown tag is refused,
    never duck-typed into whatever fits (we own both sides of the wire). The
    decoder lives HERE with the other wire codecs because canonical.py is a
    pure-value module: the encode side needs no class names, only the decode
    side does, and only the wire decodes.
    """
    from rlstack.spec import specs

    classes = (
        specs.SamplingSpec, specs.GenSpec, specs.AdapterSpec,
        specs.PolicySpec, specs.Plans, specs.OptimSpec, specs.Schedule,
        specs.AlgoSpec, specs.PoolMember, specs.LearnerMember,
        specs.HostSpec, specs.Topology, specs.Seeds, specs.WarmStart,
        specs.ExperimentSpec,
    )
    return {cls.__name__: cls for cls in classes}


def _wants_tuple(hint: object) -> bool:
    """Does this declared field type hold a tuple? JSON has only lists, so a
    decoded list becomes a tuple exactly where the dataclass declares one —
    including through an optional (`tuple[str, ...] | None`)."""
    import typing

    origin = typing.get_origin(hint)
    if origin is tuple:
        return True
    if origin is typing.Union or type(hint).__name__ == "UnionType":
        return any(_wants_tuple(arg) for arg in typing.get_args(hint))
    return False


def from_canonical(tree: object) -> object:
    """canonical_json's typed inverse: tagged dicts become their spec
    dataclasses (fields recursed, lists re-tupled where the field declares a
    tuple); untagged dicts and scalars pass through — they were plain data
    going in."""
    if isinstance(tree, dict):
        if TYPE_KEY in tree:
            return _decode_dataclass(tree)
        return {key: from_canonical(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [from_canonical(item) for item in tree]
    return tree


def _decode_dataclass(tree: dict) -> object:
    import dataclasses
    import typing

    classes = _spec_classes()
    tag = tree[TYPE_KEY]
    if tag not in classes:
        raise TypeError(
            f"from_canonical: unknown spec type {tag!r} — the decode table "
            f"holds {sorted(classes)}, and an untabled tag is refused rather "
            f"than guessed at")
    cls = classes[tag]
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for spec_field in dataclasses.fields(cls):
        value = from_canonical(tree[spec_field.name])
        if isinstance(value, list) and _wants_tuple(hints[spec_field.name]):
            value = tuple(value)
        kwargs[spec_field.name] = value
    return cls(**kwargs)


def spec_from_json(row: Mapping | str) -> object:
    """An ExperimentSpec (or any spec value) back from its canonical form —
    what the adoption door decodes. Roundtrip law, pinned by test:
    spec_from_json(json.loads(canonical_json(spec))) == spec, and therefore
    the two hash to one identity."""
    tree = json.loads(row) if isinstance(row, str) else dict(row)
    return from_canonical(tree)


def encode_sites(sites: Sequence[SiteMeta]) -> list[dict]:
    return [{"name": s.name, "path": s.path, "has_weight": s.has_weight,
             "shape": list(s.shape) if s.shape is not None else None,
             "is_boundary": s.is_boundary} for s in sites]


def decode_sites(rows: Sequence[dict]) -> tuple[SiteMeta, ...]:
    return tuple(
        SiteMeta(name=r["name"], path=r["path"], has_weight=r["has_weight"],
                 shape=tuple(r["shape"]) if r["shape"] is not None else None,
                 is_boundary=r["is_boundary"]) for r in rows)


def encode_token_batch(batch: TokenBatch) -> dict:
    """Tuples become lists; floats ride as Python's repr, which JSON keeps
    exact; extras must already be JSON-safe — the membrane's own rule at
    seal time (waves/<update>.jsonl.gz), so the wire adds none."""
    return {"token_ids": list(batch.token_ids),
            "loss_mask": list(batch.loss_mask),
            "behavior_logprobs": list(batch.behavior_logprobs),
            "segment_ids": list(batch.segment_ids),
            "doc_starts": list(batch.doc_starts),
            "postdata": {k: list(v) for k, v in batch.postdata.items()},
            "token_extras": {k: list(v) for k, v in batch.token_extras.items()},
            "doc_turn_extras": [[dict(m) for m in doc]
                                for doc in batch.doc_turn_extras],
            "microbatches_in_update": batch.microbatches_in_update,
            "documents_in_update": batch.documents_in_update}


def decode_token_batch(row: Mapping) -> TokenBatch:
    return TokenBatch(
        token_ids=tuple(row["token_ids"]), loss_mask=tuple(row["loss_mask"]),
        behavior_logprobs=tuple(row["behavior_logprobs"]),
        segment_ids=tuple(row["segment_ids"]),
        doc_starts=tuple(row["doc_starts"]),
        postdata={k: tuple(v) for k, v in row["postdata"].items()},
        token_extras={k: tuple(v) for k, v in row["token_extras"].items()},
        doc_turn_extras=tuple(tuple(doc) for doc in row["doc_turn_extras"]),
        microbatches_in_update=int(row["microbatches_in_update"]),
        documents_in_update=int(row["documents_in_update"]))


def encode_train_stats(stats: TrainStats) -> dict:
    return {"loss": stats.loss, "mean_ratio": stats.mean_ratio,
            "logprob_gap": stats.logprob_gap, "grad_norm": stats.grad_norm,
            "tokens": stats.tokens, "provided": dict(stats.provided)}


def decode_train_stats(row: Mapping) -> TrainStats:
    return TrainStats(loss=row["loss"], mean_ratio=row["mean_ratio"],
                      logprob_gap=row["logprob_gap"], grad_norm=row["grad_norm"],
                      tokens=int(row["tokens"]), provided=dict(row["provided"]))


def encode_emitted(emitted: Emitted) -> dict:
    """Bytes as base64, the way bundles already cross. ONE codec on purpose:
    a by-reference carriage later (payloads moving resident-to-resident over
    NCCL, ADR 0002 Q3) replaces this function and leaves every verb alone."""
    return {"adapters": {k: _b64(v) for k, v in emitted.adapters.items()},
            "optim": {k: _b64(v) for k, v in emitted.optim.items()}}


def decode_emitted(row: Mapping) -> Emitted:
    return Emitted(adapters={k: _unb64(v) for k, v in row["adapters"].items()},
                   optim={k: _unb64(v) for k, v in row["optim"].items()})


def encode_payloads(payloads: Mapping[str, bytes] | None) -> dict | None:
    return None if payloads is None else {k: _b64(v) for k, v in payloads.items()}


def decode_payloads(row: Mapping | None) -> dict[str, bytes] | None:
    return None if row is None else {k: _unb64(v) for k, v in row.items()}


def encode_parameterization(p: Parameterization) -> dict:
    return {"base": p.base, "loss": p.loss,
            "entries": [{"name": e.name, "adapter_type": e.adapter_type,
                         "init": dict(e.init), "trainable": e.trainable,
                         "sites": encode_sites(e.sites)} for e in p.entries],
            "optim": {"name": p.optim.name, "lr": p.optim.lr,
                      "betas": list(p.optim.betas),
                      "weight_decay": p.optim.weight_decay,
                      "overrides": {k: dict(v)
                                    for k, v in p.optim.overrides.items()}}}


def decode_parameterization(row: Mapping) -> Parameterization:
    optim = row["optim"]
    return Parameterization(
        base=row["base"], loss=row["loss"],
        entries=tuple(EntryInstall(
            name=e["name"], adapter_type=e["adapter_type"], init=dict(e["init"]),
            trainable=bool(e["trainable"]), sites=decode_sites(e["sites"]))
            for e in row["entries"]),
        optim=OptimSettings(name=optim["name"], lr=optim["lr"],
                            betas=tuple(optim["betas"]),
                            weight_decay=optim["weight_decay"],
                            overrides={k: dict(v)
                                       for k, v in optim["overrides"].items()}))


# ---------------------------------------------------------------------------
# the transport
# ---------------------------------------------------------------------------

class Undeclared:
    """The absence of a declaration, as a value.

    A metal registration's `idle_s` is THREE-VALUED (ADR 0003): a number is
    that metal's own idle limit, None PINS it (never released, however long
    it sits), and DESK_DEFAULT — the argument not given, the key absent from
    the frame — leaves the desk's own default to decide. None cannot mean
    both "pinned" and "unsaid", so the third value is named. It lives in the
    wire module because the fact is a wire fact (the `metal` frame either
    carries the key or does not) and because desk.py imports this module,
    never the reverse."""

    def __repr__(self) -> str:
        return "DESK_DEFAULT"


DESK_DEFAULT = Undeclared()


class Unreachable(RuntimeError):
    """A WIRE VERB THAT DID NOT ANSWER IN TIME (ADR 0008, F3).

    Not "it failed" and not "it will never answer" — only that the caller
    stopped waiting, which is the whole of what a deadline can tell you. The
    caller journals `unreachable` on the row it was about and carries on: the
    placement ladder passes that metal over, the reaper concludes that
    listing. An unbounded wait under the placement lock wedged the desk for
    an hour on 2026-09-04 — status, reap and every submit queued behind one
    silent metal, twice."""


DEADLINE_S = 60.0
"""THE DEFAULT BOUND on a wire verb: long enough for a busy container to get
round to a frame, short enough that a caller finds out inside a lease."""

BUILD_DEADLINE_S = 1800.0
"""The bound on a verb that BUILDS: a carve boots an engine and an adopt runs
Phase 0 and Phase 1, and both are minutes on a 32B. Still finite — a build
that has not answered in half an hour is not building."""


async def bounded(work, deadline_s: float, what: str):
    """One awaited frame with a DEADLINE (F3), and one place the refusal is
    worded. Every transport's `call` and `ask` goes through here, so "every
    wire verb has a deadline" is a property of the protocol rather than a
    habit of its implementations."""
    try:
        return await asyncio.wait_for(work, deadline_s)
    except (TimeoutError, asyncio.TimeoutError) as expired:
        raise Unreachable(
            f"{what} did not answer within {deadline_s:g}s") from expired


class Blocking:
    """THE ONE BRIDGE from a synchronous call site onto the async wire.

    Since ADR 0008 (Q4) a Transport's verbs are both coroutines — a door that
    can be cancelled without taking its container down has to be — but two
    protocols this library owns are synchronous by contract and stay that
    way: an Engine's registration and build facts (`add_bundle`,
    `reachability`, `tokenize`), and every Learner verb (ADR 0002, Q6). Those
    call sites run one coroutine on THIS loop — one loop, on one daemon
    thread, for the process's life — and block for the reply, which is
    exactly what they did before, one thread hop later.

    ONE loop and never one per call: a waiter, a future and an admission are
    all bound to the loop they were first awaited on, so a caller that
    changed loops between frames would find its own door bound to a loop that
    no longer turns. Daemonic because nothing durable lives here."""

    _loop: asyncio.AbstractEventLoop | None = None
    _minting = threading.Lock()

    @classmethod
    def loop(cls) -> asyncio.AbstractEventLoop:
        with cls._minting:
            if cls._loop is None:
                cls._loop = asyncio.new_event_loop()
                threading.Thread(target=cls._loop.run_forever, daemon=True,
                                 name="blocking-wire").start()
            return cls._loop

    @classmethod
    def run(cls, work):
        """One coroutine, driven to its reply, from a thread that owns
        nothing. The caller blocks; the wire's own deadline is what bounds
        the block."""
        return asyncio.run_coroutine_threadsafe(work, cls.loop()).result()


class Transport(Protocol):
    """Carries dict frames to one host's service. Frames are JSON-safe by
    contract; LocalTransport enforces it, real transports inherit it free.

    A transport built from an address that names an EPOCH stamps that epoch
    into every frame it carries (F2), so the receiving container can refuse a
    frame meant for an instance that no longer exists.

    BOTH VERBS ARE COROUTINES AND BOTH CARRY A DEADLINE (ADR 0008, F3/Q4). A
    cancelled SYNCHRONOUS input on a concurrent Modal container shuts the
    container down — observed three times on 2026-09-04 — where a cancelled
    async one is a task the loop drops; and a wait with no bound is how one
    unreachable metal wedges a desk. Synchronous call sites bridge through
    `Blocking`."""

    async def call(self, verb: str, payload: dict, *,
                   deadline_s: float = DEADLINE_S) -> dict:
        """An admitted verb: the serving host wraps it in its arbiter."""
        ...

    async def ask(self, verb: str, payload: dict, *,
                  deadline_s: float = DEADLINE_S) -> dict:
        """An admission-free verb: registration and build facts."""
        ...


# ---------------------------------------------------------------------------
# the epoch: a frame names the instance it means (ADR 0008, F2)
# ---------------------------------------------------------------------------

EPOCH_KEY = "@epoch"
"""THE RESERVED PAYLOAD KEY the epoch rides under.

Every rlstack door has the same three-argument shape (host, verb, payload),
so the instance a frame is addressed to travels IN the payload rather than
beside it — under a key beginning with "@", which no verb's own argument name
may be, so it can never collide with one."""


class WrongEpoch(RuntimeError):
    """A frame addressed to an instance this container is not (F2).

    A name is not an instance: a released container, a redeployed venue and a
    reborn metal all answer at the same address, and a frame minted for the
    one that is gone must fail BY NAME rather than be served by its
    successor."""


def stamped(payload: Mapping, epoch: str) -> dict:
    """`payload` addressed to one instance — what every transport built from
    an `@epoch` address sends. An epoch-free transport stamps nothing, which
    is how a registration (the frame that ANNOUNCES an epoch) reaches a
    container that has not told anyone its epoch yet."""
    return dict(payload) if not epoch else {**payload, EPOCH_KEY: epoch}


def check_epoch(payload: Mapping, mine: str, who: str) -> None:
    """THE REFUSAL, one named function and one home for the rule (F2): a
    frame naming an epoch that is not this instance's is refused by name.

    Both silences are deliberate. A frame naming NO epoch is served — the
    grammar's suffix is optional, and a registration is exactly the frame
    that cannot name one yet. A receiver with no epoch of its own serves
    anything — a hand-built host in a test process is one instance forever,
    and there is nothing for the rule to be about."""
    asked = payload.get(EPOCH_KEY)
    if not asked or not mine or asked == mine:
        return
    raise WrongEpoch(
        f"{who} is epoch {mine!r}; this frame is addressed to epoch "
        f"{asked!r}, which is an instance that no longer answers here — "
        f"re-resolve the address (the desk relists a metal at its new epoch "
        f"on its next registration)")


LEARNER_VERBS = ("install", "uninstall", "forward_backward", "optim_step",
                 "emit", "load")
"""The Learner protocol as frames — the whole of what crosses a host door to
a learner. Host-addressed: a host wears at most one learner, so unlike an
engine verb a learner verb carries no capability address, only its tenant."""


class HostService:
    """The host-side end of the wire.

    Executes pool verbs on this host's engines — and learner verbs on this
    host's learner — under this host's arbiter: the physical resource owns
    admission, so a remote experiment is just one more source of admitted work
    and never gets a vote. Engines are addressed by CAPABILITY (base, tp), the
    fleet's demand vocabulary, never by pool name — pool names are an
    experiment's private routing."""

    def __init__(self, host: Host) -> None:
        self.host = host

    def describe(self) -> dict:
        """What this host serves — the advertisement the fleet matches on."""
        return {"host": self.host.name,
                "serves": [{"base": engine.base, "tp": engine.tp}
                           for engine in self.host.engines]}

    def _engine(self, base: str | None, tp: int) -> Engine:
        """Resolution delegates to the host's own shape-matched lookup
        (Host.engine_for — the one home of the rule); the wire's only
        addition is turning "not served" into a refusal."""
        engine = self.host.engine_for(base, tp)
        if engine is None:
            raise KeyError(
                f"host {self.host.name!r} serves no ({base!r}, tp={tp}); it "
                f"serves {[(e.base, e.tp) for e in self.host.engines]}")
        return engine

    async def serve(self, verb: str, payload: dict) -> dict:
        """One admitted verb, admission included: enter the owning host's
        arbiter, run, leave. A regime-host's engines are attached at birth; a
        bare host's attach here on first remote use, at zero footprint — a
        joiner never re-counts a size the partition already owns.

        `adopt` and `stop` ride this async path but are NOT admitted: adopt
        registers a tenancy whose daemons admit their own work, stop cancels
        one, so neither door occupies anything — and both are host-addressed,
        so they resolve no engine. The learner verbs ride it BECAUSE they are
        admitted (serve_learner): they occupy this host's training metal, so
        they are exactly the shape sample_tokens has.

        THE EPOCH IS CHECKED FIRST (F2): a frame minted for a host that died
        with its container is refused before anything is admitted, because a
        carve name recycles and the successor must not serve its corpse's
        mail."""
        check_epoch(payload, self.host.epoch, f"host {self.host.name!r}")
        if verb == "adopt":
            return await self.host.adopt(payload["spec"],
                                         payload.get("routes", {}),
                                         payload.get("code"),
                                         payload.get("subdir"))
        if verb == "stop":
            return await self.host.stop(payload["run_id"])
        if verb in LEARNER_VERBS:
            return await self.serve_learner(verb, payload)
        engine = self._engine(payload["base"], payload["tp"])
        if not self.host.arbiter.is_attached(engine):
            self.host.arbiter.attach(
                engine, label=f"{self.host.name}:{payload['base'] or '*'}")
        if verb not in ("sample_tokens", "score_tokens"):
            raise ValueError(f"unknown admitted verb {verb!r}")
        async with self.host.arbiter.admit(engine):
            return await EngineService(engine).serve(verb, payload)

    async def serve_learner(self, verb: str, payload: dict) -> dict:
        """One learner verb from ANOTHER host, admitted here (ADR 0006 A).

        The learner is reached exactly as an engine is: this host owns the
        metal, so THIS host's arbiter admits every frame — which is what makes
        the learner's alternation group (a multi-member HostSpec) honored
        where the learner lives, whoever is driving it. A bare host's learner
        attaches on first remote use, at zero footprint, exactly as its
        engines do. The verb itself is synchronous — a Learner's protocol is —
        so the admitted frame executes through the learner's own service, and
        a frame that raises is a refusal to the caller, never a state here."""
        learner = self._learner()
        if not self.host.arbiter.is_attached(learner):
            self.host.arbiter.attach(learner, label=f"{self.host.name}:learner")
        async with self.host.arbiter.admit(learner):
            reply = LearnerService(learner).answer(verb, payload)
        self.journal_custody(verb, payload["tenant"])
        return reply

    def _learner(self) -> Learner:
        """Resolution's learner half — the wire's only addition to Host's own
        custody rule is turning "wears no learner" into a refusal, exactly as
        `_engine` turns "does not serve that address" into one."""
        if self.host.learner is None:
            raise KeyError(
                f"host {self.host.name!r} wears no learner; it serves "
                f"{[(e.base, e.tp) for e in self.host.engines]}")
        return self.host.learner

    def journal_custody(self, verb: str, tenant: str) -> None:
        """WHO IS ON THIS LEARNER, journaled at its two ends: a foreign
        `install` writes `learner-attach`, a foreign `uninstall` writes
        `learner-detach` (run_id, t). A tenant of this learner may be a run
        anchored on another host, whose attach/detach rows live in THAT
        host's journal — so without these lines nothing here would say what
        the training metal is holding. Observability only: the observer reads
        it, correctness never does, and no run directory sees it."""
        if verb not in ("install", "uninstall"):
            return
        self.host.store.append_host_event(self.host.name, {
            "event": "learner-attach" if verb == "install" else "learner-detach",
            "t": time.time(), "run_id": tenant})

    def answer(self, verb: str, payload: dict) -> dict:
        """One admission-free verb: additive registration (add_bundle never
        disturbs traffic — the multi-tenancy invariant) and build facts
        (reachability, tokenize), all callable from sync call sites. `status`
        is host-addressed (the roster, the partition, the adoptions' fates)
        and resolves no engine. The epoch is checked here too (F2): a probe
        is a frame like any other, and a corpse's address answering `status`
        is exactly the deaf-metal hazard."""
        check_epoch(payload, self.host.epoch, f"host {self.host.name!r}")
        if verb == "status":
            return self.host.status()
        engine = self._engine(payload["base"], payload["tp"])
        return EngineService(engine).answer(verb, payload)


class EngineService:
    """ONE engine's verbs, admission-free: the resident's end of its door.

    HostService admits and resolves, then forwards here — so this is the
    engine-verb half of the host wire with the admission left behind, and
    the whole of what an engine resident (runner/residents.py) serves. A
    frame reaching this class has already been admitted by the host that
    owns the partition, or came from that host's own runner."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    async def serve(self, verb: str, payload: dict) -> dict:
        if verb == "sample_tokens":
            events = [event async for event in self.engine.sample_tokens(
                decode_messages(payload["messages"]),
                decode_sampling(payload["sampling"]),
                tuple(payload["stop"]), payload["bundle_id"],
                payload["seed"],
                decode_directives(payload.get("directives", ())))]
            return {"events": encode_events(events)}
        if verb == "score_tokens":
            scores = await self.engine.score_tokens(
                decode_messages(payload["messages"]),
                tuple(payload["token_ids"]), payload["bundle_id"],
                decode_directives(payload.get("directives", ())))
            return {"logprobs": list(scores)}
        raise ValueError(f"unknown engine verb {verb!r}")

    def answer(self, verb: str, payload: dict) -> dict:
        if verb == "add_bundle":
            self.engine.add_bundle(decode_bundle(payload["bundle"]))
            return {}
        if verb == "knows_bundle":
            return {"known": self.engine.knows_bundle(payload["bundle_id"])}
        if verb == "reachability":
            reach = self.engine.reachability(decode_sites(payload["sites"]))
            return {"mechanisms": {name: mech.name
                                   for name, mech in reach.items()}}
        if verb == "tokenize":
            return {"token_ids": list(self.engine.tokenize(payload["text"]))}
        if verb == "traffic":
            # THE RESIDENT'S OWN WINDOW, drained here and nowhere else. An
            # engine counts its tokens in ITS process (ADR 0002), so the host
            # that journals `traffic` asks each resident on its stats tick
            # and adds the answer to its own door's counts (host.traffic_row).
            # Found on the venue: 71 windows of a 32B generating at ~200
            # tok/s, every one of them zero, because the host drained a
            # meter nothing across the pipe had ever fed.
            meter = getattr(self.engine, "meter", None)
            return {} if meter is None else meter.drain(time.time()).row()
        raise ValueError(f"unknown admission-free verb {verb!r}")


class LearnerService:
    """ONE learner's verbs as frames: the learner resident's end of its door.

    Every verb is synchronous and pins a tenant (I8), so at the RESIDENT's
    door they all ride the `ask` path and arrive in the order the Trainer
    issued them; nothing here admits. Who admitted depends on which door the
    frame came through: for the host's own learner the Trainer admitted at its
    own arbiter before the first frame, and for a learner reached from another
    host `HostService.serve_learner` admitted at the serving host's arbiter
    and then called straight into `answer`."""

    def __init__(self, learner: Learner) -> None:
        self.learner = learner

    async def serve(self, verb: str, payload: dict) -> dict:
        raise ValueError(
            f"{verb!r}: a learner's verbs are synchronous — at a RESIDENT's "
            f"door they ride ask, never call; the admitted async path is the "
            f"HOST door's (HostService.serve_learner), which admits and then "
            f"answers here")

    def answer(self, verb: str, payload: dict) -> dict:
        tenant = payload["tenant"]
        if verb == "install":
            self.learner.install(
                tenant, decode_parameterization(payload["parameterization"]))
            return {}
        if verb == "uninstall":
            self.learner.uninstall(tenant)
            return {}
        if verb == "forward_backward":
            return encode_train_stats(self.learner.forward_backward(
                tenant, decode_token_batch(payload["batch"])))
        if verb == "optim_step":
            self.learner.optim_step(tenant)
            return {}
        if verb == "emit":
            return encode_emitted(self.learner.emit(tenant))
        if verb == "load":
            self.learner.load(tenant, decode_payloads(payload["adapters"]) or {},
                              decode_payloads(payload["optim"]))
            return {}
        raise ValueError(f"unknown learner verb {verb!r}")


def json_roundtrip(frame: dict) -> dict:
    """The honesty gate: a frame that survives this survives any real wire."""
    return json.loads(json.dumps(frame))


_json_roundtrip = json_roundtrip


class Service(Protocol):
    """The SERVER end of the wire: what a transport carries frames to.

    Two verbs, mirroring the Transport's own: `serve` for the admitted ones
    and `answer` for the admission-free ones. HostService, EngineService and
    LearnerService wear it here; Desk, Campaigns and MetalService wear it on
    the fleet plane, which is why one LocalTransport and one address grammar
    reach all five."""

    async def serve(self, verb: str, payload: dict) -> dict:
        ...

    def answer(self, verb: str, payload: dict) -> dict:
        ...


class LocalTransport:
    """Same-process transport that still crosses the serialization boundary
    (json round-trip both ways), so a fleet whose hosts share one process is
    indistinguishable, from above, from one whose hosts do not.

    `epoch` is the instance the frames are addressed to (F2), stamped exactly
    as a real transport stamps it — so the epoch refusal is exercisable with
    no container anywhere, which is what the fakes suite needs it to be."""

    def __init__(self, service: Service, epoch: str = "") -> None:
        self.service = service
        self.epoch = epoch

    async def call(self, verb: str, payload: dict, *,
                   deadline_s: float = DEADLINE_S) -> dict:
        return await bounded(self.served(verb, payload), deadline_s,
                             f"{type(self.service).__name__}.{verb}")

    async def ask(self, verb: str, payload: dict, *,
                  deadline_s: float = DEADLINE_S) -> dict:
        return await bounded(self.answered(verb, payload), deadline_s,
                             f"{type(self.service).__name__}.{verb}")

    async def served(self, verb: str, payload: dict) -> dict:
        return _json_roundtrip(
            await self.service.serve(
                verb, _json_roundtrip(stamped(payload, self.epoch))))

    async def answered(self, verb: str, payload: dict) -> dict:
        """The admission-free half, ON A THREAD. A `Service.answer` is
        synchronous by contract, and running it inline would hold the loop
        that is meant to be bounding it — so the deadline above could never
        fire, and a slow in-process answer would starve every other frame on
        that loop. The hop is what makes "LocalTransport honours the
        deadline" true rather than nominal (ADR 0008, F3)."""
        frame = _json_roundtrip(stamped(payload, self.epoch))
        return _json_roundtrip(
            await asyncio.to_thread(self.service.answer, verb, frame))


# ---------------------------------------------------------------------------
# the address grammar, and the one factory that reads it
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Address:
    """A wire address in parts. The grammar, stated once (ADR 0007, Q3;
    the epoch is ADR 0008, Q2):

        modal://<app>/<cls>#<host>   one host's door inside a Modal class
        modal://<app>/<cls>          that container's own plane (a desk, a
                                     metal)
        local://<host>               the IN-PROCESS wire: a service standing
                                     in this very process under that name
        ...@<epoch>                  any of the above, addressed to ONE
                                     INSTANCE of it

    THE VENUE IS IN THE ADDRESS. One desk serves metal in many apps, so the
    app a frame goes to cannot be baked into a transport class the way each
    venue's own transports baked it before ADR 0007 — the journal's `address`
    fields become self-describing, and a desk rebuilt from them can reach
    every metal it ever registered.

    AND THE INSTANCE IS IN THE ADDRESS (F2). A name is not an instance: a
    container mints an EPOCH at bring-up, and a frame that names an epoch is
    refused by any container wearing another one. The suffix is optional —
    an address without it addresses whoever answers, which is what a
    registration frame must do, because the epoch is what it is announcing."""

    scheme: str
    app: str = ""
    cls: str = ""
    host: str = ""
    epoch: str = ""


def parse_address(address: str) -> Address:
    """An address string as its parts — the grammar's ONE reader.

    A `modal://` address that names no class is refused here, because a
    half-address is a venue bug and the frame that would ride it is a lost
    hour. An unfamiliar scheme parses (everything after `://` is its host)
    and is refused by `transport_for`, which is where the closed set of
    substrates actually lives. A trailing `@<epoch>` is read off first,
    whatever the scheme: it names the INSTANCE, never the route."""
    scheme, sep, rest = address.partition("://")
    if not sep or not scheme or not rest:
        raise ValueError(
            f"{address!r} is not an address: the grammar is "
            f"modal://<app>/<cls>[#<host>][@<epoch>] or "
            f"local://<host>[@<epoch>]")
    rest, at, epoch = rest.rpartition("@")
    if not at:
        rest, epoch = epoch, ""
    if scheme != "modal":
        return Address(scheme=scheme, host=rest, epoch=epoch)
    path, _, host = rest.partition("#")
    app, slash, cls = path.partition("/")
    if not slash or not app or not cls:
        raise ValueError(
            f"{address!r} names no Modal class: a modal address is "
            f"modal://<app>/<cls>[#<host>] — the app and the class are what "
            f"one desk needs to reach metal in another venue's app")
    return Address(scheme="modal", app=app, cls=cls, host=host, epoch=epoch)


def with_epoch(address: str, epoch: str) -> str:
    """An address addressed to ONE INSTANCE of what it names (F2).

    The desk journals a metal's PLANE address without an epoch — the address
    is the container's stable name, and a re-registration at a new epoch must
    read as the same metal, not as a second deploy colliding on it — and
    composes the two here when it builds the remote it will actually send
    frames through. A host's address, minted by the venue at carve, carries
    its epoch already: carve names recycle when a container is reborn, and
    the epoch is what tells the corpse from the newborn."""
    return f"{address}@{epoch}" if epoch else address


def without_epoch(address: str) -> str:
    """`with_epoch`'s inverse: what an address ROUTES to.

    The epoch is never part of the route — a container answers at its name
    whatever life it is on, and the refusal happens at the door, not in the
    dial. So every transport dials the route and STAMPS the epoch, and the
    in-process switchboard is keyed by the route for the same reason."""
    head, at, _ = address.rpartition("@")
    return head if at else address


IN_PROCESS: dict[str, Service] = {}
"""THE IN-PROCESS SWITCHBOARD: address -> the service standing HERE.

An address answered inside this very process is answered IN-PROCESS, never
over the wire. On Modal that is not an optimization but the difference
between working and wedging: a host adopts on the container's event loop and
asks reachability through the transport's SYNC verb, so a Modal self-call
would wait on the loop it is itself blocking (#77, an hour of silence found
on the venue). A metal publishes each host it carves here (MetalService.route)
and a venue publishes its own plane; the factory below is the one reader."""


def serve_in_process(address: str, service: Service) -> None:
    """Publish `service` as the answer to `address` in this process."""
    IN_PROCESS[address] = service


def stop_serving_in_process(address: str) -> None:
    """The inverse: nothing answers here any more (a decarve, a release)."""
    IN_PROCESS.pop(address, None)


def transport_for(address: str) -> Transport:
    """THE ONE FACTORY: an address in, the transport that reaches it out.

    Every resolver in the fleet is this function — a desk's `host_for` and
    `metal_for`, a host's `transport_for`, a campaign's door — so a venue
    never constructs a transport by class and never writes a scheme rule of
    its own. The in-process switchboard is consulted FIRST (#77's rule, said
    once), then the scheme names the substrate; the heavy region is imported
    inside its branch, so `import rlstack` never imports a venue SDK
    (STYLE rule 7)."""
    parsed = parse_address(address)
    standing = IN_PROCESS.get(address) or IN_PROCESS.get(without_epoch(address))
    if standing is not None:
        return LocalTransport(standing, parsed.epoch)
    if parsed.scheme == "modal":
        from rlstack.runner.transports.modal_cls import ModalClsTransport

        return ModalClsTransport(parsed.app, parsed.cls, parsed.host,
                                 parsed.epoch)
    if parsed.scheme == "local":
        raise ValueError(
            f"nothing answers {address!r} in this process: a local:// address "
            f"IS the in-process wire, so it is reachable only where the "
            f"service stands. Serving here: {sorted(IN_PROCESS)}")
    raise ValueError(
        f"no transport for scheme {parsed.scheme!r} ({address!r}): the wire "
        f"substrates are one file each under rlstack/runner/transports/, and "
        f"this fleet speaks 'modal' and 'local'")


# ---------------------------------------------------------------------------
# the client end
# ---------------------------------------------------------------------------

class RemotePool:
    """An Engine served by another host, reached through a Transport.

    Implements the full Engine protocol; base and tp are this pool's
    CAPABILITY ADDRESS on the serving host (the fleet's demand vocabulary),
    and double as the build facts the submit gate attests — the serving host
    refuses an address it does not serve, so the attestation is real.

    TRAFFIC IS COUNTED WHERE IT IS SERVED, never here: every verb below lands
    in HostService, which admits it through the SERVING host's arbiter and
    runs it on that host's engine — both already wired to that host's meter.
    A client-side count would attribute another partition's load to this one,
    so this meter exists only to satisfy the protocol and stays at zero."""

    def __init__(self, transport: Transport, *, base: str | None = None,
                 tp: int = 1, deadline_s: float = BUILD_DEADLINE_S) -> None:
        self.base = base
        self.tp = tp
        # THE BOUND ON THIS POOL'S FRAMES (ADR 0008, F3). Generous, because a
        # sampling request behind a full batch legitimately waits — and
        # finite, because the Engine protocol has no deadline argument of its
        # own and every wire verb must have one.
        self.deadline_s = deadline_s
        self.meter = TrafficMeter()
        self._transport = transport

    def _address(self) -> dict:
        return {"base": self.base, "tp": self.tp}

    def _asked(self, verb: str, payload: dict) -> dict:
        """One admission-free frame from a SYNCHRONOUS call site: the Engine
        protocol's registration and build facts are sync by contract (they
        are called from Phase 0, off any loop of their own), so they bridge
        through `Blocking` onto the async wire."""
        return Blocking.run(self._transport.ask(
            verb, payload, deadline_s=self.deadline_s))

    async def sample_tokens(self, messages, sampling, stop, bundle_id,
                            seed, directives: Sequence[Directive] = (),
                            ) -> AsyncIterator[TokenEvent | FinishEvent]:
        reply = await self._transport.call("sample_tokens", {
            **self._address(), "messages": encode_messages(messages),
            "sampling": encode_sampling(sampling), "stop": list(stop),
            "bundle_id": bundle_id, "seed": seed,
            "directives": encode_directives(directives)},
            deadline_s=self.deadline_s)
        for event in decode_events(reply["events"]):
            yield event

    async def score_tokens(self, messages, token_ids, bundle_id,
                           directives: Sequence[Directive] = (),
                           ) -> tuple[float, ...]:
        reply = await self._transport.call("score_tokens", {
            **self._address(), "messages": encode_messages(messages),
            "token_ids": list(token_ids), "bundle_id": bundle_id,
            "directives": encode_directives(directives)},
            deadline_s=self.deadline_s)
        return tuple(reply["logprobs"])

    def add_bundle(self, bundle: Bundle) -> None:
        self._asked("add_bundle", {
            **self._address(), "bundle": encode_bundle(bundle)})

    def knows_bundle(self, bundle_id: str) -> bool:
        """Admission-free, like the registration it guards: asking what a pool
        holds never disturbs traffic, and the answer is about the SERVING
        engine's residency — which is why it has to cross the wire rather than
        be remembered here."""
        reply = self._asked("knows_bundle", {
            **self._address(), "bundle_id": bundle_id})
        return bool(reply["known"])

    def reachability(self, sites: Sequence[SiteMeta]) -> Mapping[str, Mechanism]:
        reply = self._asked("reachability", {
            **self._address(), "sites": encode_sites(sites)})
        return {name: Mechanism[mech]
                for name, mech in reply["mechanisms"].items()}

    def tokenize(self, text: str) -> tuple[int, ...]:
        reply = self._asked("tokenize", {
            **self._address(), "text": text})
        return tuple(reply["token_ids"])

    def drain_traffic(self) -> dict:
        """The resident's traffic window since its last drain, as a row — what
        the host adds to its own on each stats tick, because the engine
        behind this door counts in its own process (ADR 0002). Empty when
        that engine keeps no meter. Only a host's OWN engines are asked:
        a pool reached as another host's remote is that host's to drain."""
        return self._asked("traffic", self._address())


class RemoteLearner:
    """A Learner served by a resident or by another HOST, reached through a
    Transport — RemotePool's twin, and the Host's end of a learner's door.

    Implements the whole Learner protocol; `fsdp` is the build fact this
    learner was declared at, which the submit gate attests exactly as it did
    against an in-process learner. For a resident's own proxy the number comes
    from the hello; for a routed one it comes from the LearnerMember, and it
    is real for the same reason RemotePool's (base, tp) is: the desk places a
    training demand only on a listing whose training regime has that shape,
    and that host attested its learner against the regime at birth.

    TWO DOORS, ONE CLIENT, because admission lives with the metal:

    `admitted=False` is THE RESIDENT'S door — the Host's own learner, a
    process beside the runner (ADR 0002). The Trainer already admitted it at
    its own arbiter before the first frame, and the resident runs asks inline
    in arrival order, so every verb is one synchronous `ask`.

    `admitted=True` is ANOTHER HOST'S door (ADR 0006 Part A). Every frame must
    be admitted at the SERVING host's arbiter, and admission is the async path
    there (`HostService.serve`, the one `sample_tokens` rides), so the verbs
    ride `call` — driven to completion on this proxy's own thread, because the
    Learner protocol is synchronous (ADR 0002 Q6, kept) and the caller is
    usually an event loop that must not be re-entered.
    """

    def __init__(self, transport: Transport, *, fsdp: int = 1,
                 admitted: bool = False,
                 deadline_s: float = BUILD_DEADLINE_S) -> None:
        self.fsdp = fsdp
        self.admitted = admitted
        # THE BOUND ON A LEARNER FRAME (ADR 0008, F3): a 32B's install and one
        # microbatch's forward are both minutes, so the bound is the build's
        # — but it is a bound, and an expired one is `Unreachable` rather than
        # a Trainer waiting for a process that is never coming back.
        self.deadline_s = deadline_s
        self._transport = transport
        self._frames: asyncio.AbstractEventLoop | None = None

    # ---- the two doors, one named method each -------------------------------

    def frame(self, verb: str, payload: dict) -> dict:
        """One verb through whichever door this proxy speaks (see the class
        docstring): the resident's takes it admission-free, another host's
        takes it admitted. Both bridge from this synchronous call site onto
        the async wire (ADR 0008, Q4) — a Learner verb may not become async
        (ADR 0002, Q6), so the hop is here."""
        if not self.admitted:
            return Blocking.run(self._transport.ask(
                verb, payload, deadline_s=self.deadline_s))
        return self.admitted_frame(verb, payload)

    def admitted_frame(self, verb: str, payload: dict) -> dict:
        """An admitted frame driven from a SYNCHRONOUS call site: the
        coroutine runs on this proxy's OWN loop, on its own thread, and the
        caller blocks for the reply — because the Trainer issuing it is
        usually already inside an event loop, which may be neither re-entered
        nor made to wait, and a Learner verb may not become async (ADR 0002
        Q6, kept). The shape is every venue's own `blocking_ask`
        (deploy/steer_l4.py: one blocking call on its own thread).

        Costs and limits, stated: one thread hop beside a wire hop; a frame
        in flight cannot be cancelled, exactly as an in-process
        forward_backward cannot; and over a LOCAL transport — a fleet sharing
        one process — the serving host's arbiter is entered from this loop
        while the caller's loop is blocked, so a same-process serving host
        that ALTERNATES must not also be carrying engine work admitted from
        the caller's loop. Over a real transport the serving host admits
        every frame on its own loop and the question does not arise."""
        return asyncio.run_coroutine_threadsafe(
            self._transport.call(verb, payload, deadline_s=self.deadline_s),
            self.frames_loop()).result()

    def frames_loop(self) -> asyncio.AbstractEventLoop:
        """This proxy's own loop, on its own daemon thread, alive for the
        proxy's life. ONE loop, never one per frame: the serving host's
        admission is built out of asyncio primitives that belong to the loop
        they were first awaited on, so a caller that changed loops between
        frames would find its own alternation door bound to a loop that no
        longer turns. Daemonic because this thread holds no state — the
        tenant's state is at the learner — so an exiting process may drop
        it."""
        if self._frames is None:
            self._frames = asyncio.new_event_loop()
            threading.Thread(target=self._frames.run_forever, daemon=True,
                             name="learner-frames").start()
        return self._frames

    # ---- the Learner protocol -----------------------------------------------

    def install(self, tenant: str, parameterization: Parameterization) -> None:
        self.frame("install", {
            "tenant": tenant,
            "parameterization": encode_parameterization(parameterization)})

    def uninstall(self, tenant: str) -> None:
        self.frame("uninstall", {"tenant": tenant})

    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats:
        return decode_train_stats(self.frame("forward_backward", {
            "tenant": tenant, "batch": encode_token_batch(batch)}))

    def optim_step(self, tenant: str) -> None:
        self.frame("optim_step", {"tenant": tenant})

    def emit(self, tenant: str) -> Emitted:
        return decode_emitted(self.frame("emit", {"tenant": tenant}))

    def load(self, tenant: str, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None:
        self.frame("load", {
            "tenant": tenant, "adapters": encode_payloads(adapters),
            "optim": encode_payloads(optim)})


class RemoteHost:
    """The client end of ADOPTION: hand a standing host an experiment.

    `adopt` ships the spec's canonical JSON plus placement's pool routes and
    returns the host's acceptance — {run_id, state} or a refusal — never a
    result: the ledger is the result channel, and `status` (the roster over
    the wire) is how a client watches its tenancy without a second channel
    existing. A campaign that wants to WAIT tails the run's own store."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def adopt(self, spec: object,
                    routes: Mapping[str, str] | None = None,
                    code: Mapping[str, str] | None = None,
                    subdir: str | None = None,
                    deadline_s: float = BUILD_DEADLINE_S) -> dict:
        """`spec` may be a live ExperimentSpec (encoded here, hashes computed
        here) or an already-canonical row (forwarded as-is — the DESK's case,
        relaying a client's frame with the CLIENT's claimed hashes)."""
        from rlstack.spec.canonical import canonical_json

        if isinstance(spec, Mapping):
            row = dict(spec)
        else:
            row = json.loads(canonical_json(spec))
            if code is None:
                from rlstack.registry import code_hashes
                code = code_hashes(spec)
        return await self._transport.call("adopt", {
            "spec": row, "routes": dict(routes or {}),
            "code": dict(code or {}), "subdir": subdir},
            deadline_s=deadline_s)

    async def stop(self, run_id: str, deadline_s: float = DEADLINE_S) -> dict:
        """Tell the host to stop a tenancy — cancellation awaited host-side,
        so the reply means the death is complete and the run_id is free to
        adopt again, here or elsewhere (a reroute's first half)."""
        return await self._transport.call("stop", {"run_id": run_id},
                                          deadline_s=deadline_s)

    async def status(self, deadline_s: float = DEADLINE_S) -> dict:
        """The roster over the wire, BOUNDED (ADR 0008, F3): a probe that
        never returns is how one wedged host stalls a whole reaper tick, so
        the caller stops waiting and journals `unreachable` instead."""
        return await self._transport.ask("status", {}, deadline_s=deadline_s)


class RemoteMetal:
    """The desk's end of the METAL PLANE: deduce, then command.

    `residual` is the deduction feed — per-device free GB from the
    container that owns the device, counting built partitions AND in-flight
    bookings. `carve` is the command: the metal books synchronously at its
    own door, so a deduction gone stale between the ask and the command
    costs a refusal, never a double-book. `decarve` frees a host's GB on a
    still-living container — the reaper's second half. `release` is the
    wholesale one: every host down and the shift ended, so the venue takes
    the container back (ADR 0003)."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def residual(self, deadline_s: float = DEADLINE_S) -> list[float]:
        """Free GB per device, LIVE and BOUNDED (ADR 0008, Q3 as amended). A
        live read is the stronger check — it proves the metal is reachable
        NOW and sees a decarve a cached number would miss — and the whole of
        what makes it safe is that it is bounded and taken OUTSIDE the
        placement lock."""
        reply = await self._transport.ask("residual", {},
                                          deadline_s=deadline_s)
        return list(reply["residual"])

    async def describe(self, deadline_s: float = DEADLINE_S) -> dict:
        return await self._transport.ask("describe", {},
                                         deadline_s=deadline_s)

    async def carve(self, request: Mapping,
                    deadline_s: float = BUILD_DEADLINE_S) -> dict:
        return await self._transport.call("carve", dict(request),
                                          deadline_s=deadline_s)

    async def decarve(self, name: str,
                      deadline_s: float = DEADLINE_S) -> dict:
        return await self._transport.call("decarve", {"host": name},
                                          deadline_s=deadline_s)

    async def release(self, deadline_s: float = DEADLINE_S) -> dict:
        """The acquire rung inverted at the metal (ADR 0003): every resident
        down the ladder, the books emptied, and the SHIFT ENDED so the venue
        reclaims the container. Idempotent — a bare or already-released
        metal answers released just the same, so a desk unsure whether its
        release landed may simply say it again."""
        return await self._transport.call("release", {}, deadline_s=deadline_s)


class RemoteDesk:
    """The client end of the standing fleet: a campaign's whole surface.

    The desk is workload-blind, so the SHAPING happens here, client-side:
    `submit` turns a spec into demand rows (one of them the anchor) plus an
    opaque frame via the campaign layer, and one frame carries both to the
    desk — which places, delivers to the anchor, and answers with where
    everything landed (or what to boot). `resolve` is the pure client's
    verb: demands in, addresses out, no delivery. After either, a client
    watches the store — the desk holds no results, exactly as no host does."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def submit(self, spec: object, subdir: str | None = None,
                     anchor: str | None = None, solo: bool = False) -> dict:
        """`anchor` names the member the frame lands on — and therefore where
        the run's Trainer sits (campaign.anchor_demand's rule): unasked, the
        learner's host when the spec declares one, `main` otherwise."""
        from rlstack.runner.campaign import demands_of, frame_for
        from rlstack.runner.desk import demand_rows

        return await self._transport.call("submit", {
            "demands": demand_rows(demands_of(spec, anchor)),
            "frame": frame_for(spec, subdir), "solo": solo})

    async def resolve(self, demands: Sequence) -> dict:
        """Demands in, addresses out — placement without a workload: the
        door for an evaluator, a scorer, any client that wants a pool."""
        from rlstack.runner.desk import demand_rows

        return await self._transport.call("place",
                                          {"demands": demand_rows(demands)})

    async def status(self, deadline_s: float = DEADLINE_S) -> dict:
        return await self._transport.ask("status", {}, deadline_s=deadline_s)

    async def liveness(self, deadline_s: float = DEADLINE_S) -> dict:
        """{host: alive} for every listing, probed by the desk just now.

        It rides the ADMITTED path since ADR 0008: probing every listing is a
        fan of wire calls, and a verb that goes to the wire belongs where the
        desk can bound and cancel it, not on the door that is supposed to
        answer off memory alone."""
        return await self._transport.call("liveness", {},
                                          deadline_s=deadline_s)

    async def heartbeat(self, name: str, epoch: str,
                        residual: Sequence[float] | None = None) -> dict:
        """ONE RENEWAL (ADR 0008, F1): `name` — a metal, or a host carved on
        one — is still there, and it is still `epoch`. A metal's duty sends
        one of these per name every `heartbeat_s`; the residual rides along
        for the row the observer shows. The reply says whether the desk
        believed it, and a refusal names why (an unknown name, a replaced
        epoch) so the container can register again instead of heartbeating
        into a desk that has forgotten it."""
        payload: dict = {"name": name, "epoch": epoch}
        if residual is not None:
            payload["residual"] = [float(gb) for gb in residual]
        return await self._transport.call("heartbeat", payload)

    async def list_host(self, name: str, regimes: Sequence,
                        address: str, solo: bool = False,
                        partition: Mapping | None = None,
                        metal: str = "", epoch: str = "") -> dict:
        """A booted host enters the standing fleet over the wire — the
        phone-home half of the deploy contract: the container that stood the
        host tells the desk what it wears, where it answers, (the capacity
        view) the partition row it was born onto and the metal it lives on,
        and (ADR 0008) the EPOCH it is — which opens its lease."""
        return await self._transport.call("list", {
            "host": name, "address": address, "solo": solo,
            "partition": dict(partition) if partition else None,
            "metal": metal, "epoch": epoch,
            "regimes": [{"name": r.name, "capability": r.capability,
                         "base": r.base, "shape": r.shape} for r in regimes]})

    async def delist(self, name: str, reason: str = "") -> dict:
        """The listing's retirement, over the same wire it entered by —
        bookkeeping only, the metal untouched. For teardown, decommission."""
        return await self._transport.call("delist", {"host": name,
                                                     "reason": reason})

    async def decommission(self, name: str, force: bool = False,
                           reroute: bool = False) -> dict:
        """Carve's inverse at the desk, one frame: decarve at the host's
        metal (engine down, its GB back to residual) plus delist. Refused
        with the running work NAMED when anything lives on or routes through
        the host; `force` tears it down anyway. `reroute` MOVES the running
        work first — each dependent replayed onto a fresh placement with
        this host off the table, or parked (stopped, journaled, waiting in
        the store) when nothing else covers it."""
        return await self._transport.call("decommission",
                                          {"host": name, "force": force,
                                           "reroute": reroute})

    async def reroute(self, run_id: str, avoiding: str = "",
                      park: bool = False) -> dict:
        """Move a delivered workload: the desk replays its archived delivery
        onto a fresh placement (skipping `avoiding`), stopping the old
        tenancy only once there is somewhere to go. `park` stops it
        regardless and journals the run parked — decommission's mode, when
        the host is dying either way."""
        return await self._transport.call("reroute", {
            "run_id": run_id, "avoiding": avoiding, "park": park})

    async def placements(self) -> dict:
        """The desk's current-binding table: the latest delivered placement
        per run_id — pools to listings, plus the archived demand rows and
        frame where the delivery carried them."""
        reply = await self._transport.ask("placements", {},
                                          deadline_s=DEADLINE_S)
        return reply["placements"]

    async def register_metal(self, name: str, gpu: str, devices: int,
                             vram_gb: float, address: str,
                             builds: Mapping | None = None,
                             idle_s: float | None | Undeclared = DESK_DEFAULT,
                             container: str | None = None,
                             epoch: str = "") -> dict:
        """A metal container phones home its OWN existence — the other half
        of the deploy contract: after this the desk can deduce (residual)
        and command (carve/decarve) against it at `address`. The facts are
        MEASURED (MetalService.measure). `builds` is the metal's recipe row
        (Builds.row()), the desk's canon from here on, journaled so the
        record of HOW a host was built is durable (ADR 0002, Q4a). A known
        name at the same address is a re-registration — the container
        generation turned over: the desk updates the row, reaps that metal's
        corpses and retries the parked queue (ADR 0001, Q5); the reply says
        what it reaped and retried. A registration also RE-ACQUIRES a metal
        the desk had released (ADR 0003) — the row goes back on the
        carve-able set. `idle_s` is this metal's own idle limit: unsaid, the
        desk's default decides; None PINS it, never released.

        `epoch` is this CONTAINER's boot identity (ADR 0008, F2), minted at
        bring-up: it opens the metal's lease, rides every frame the desk sends
        back, and is what lets the desk address two lives of one metal name as
        two instances. The reply carries the lease constants the desk holds,
        so the container heartbeats at the desk's cadence and not its own."""
        payload = {"name": name, "gpu": gpu, "devices": devices,
                   "vram_gb": vram_gb, "address": address, "epoch": epoch,
                   "builds": dict(builds) if builds else None,
                   "container": container}
        if not isinstance(idle_s, Undeclared):
            payload["idle_s"] = idle_s
        return await self._transport.call("metal", payload)

    async def recipe(self, metal: str, builds: Mapping) -> dict:
        """DECLARE WHAT A METAL BUILDS (ADR 0007, Q4). `builds` is a
        `Builds.row()`: the desk journals it as a `recipe` event, the latest
        per metal wins, and the next carve to that metal carries it — which is
        how a container that booted bare learns what it is for. Through the
        desk rather than into the journal directly, because the fleet journal
        has one writer (I10)."""
        return await self._transport.call("recipe", {"metal": metal,
                                                     "builds": dict(builds)})

    async def release(self, name: str, reason: str = "released",
                      force: bool = False) -> dict:
        """Hand a metal back BY HAND — the acquire rung inverted at the desk
        (ADR 0003): its listings delisted, its residents down the ladder,
        its shift ended so the venue reclaims the container, and the row
        kept as inventory the next placement may knock awake. The desk's
        idle sweep issues this same verb on its own clock; this is the door
        for an operator who knows the metal is done sooner.

        REFUSED WHEN SOMEONE ELSE IS STILL WORKING THERE (ADR 0007, Q6):
        under one desk a venue tearing down the metal it acquired would take
        every other experiment on it too, so running work is named back and
        nothing comes down. `force` overrides and is the desk operator's
        verb; a venue door leaves it unsaid."""
        payload = {"metal": name, "reason": reason}
        if force:
            payload["force"] = True
        return await self._transport.call("release", payload)

    async def reap(self, probes: int = 3, wait: float = 0.0) -> dict:
        """The janitor's sweep, run by the desk now: probe every listing,
        retry the silent (on a lazy venue the knock is the restart), reap
        what stays silent — decarve at its metal, delist with the reason
        journaled — then RECONTINUE: strand the reaped hosts' runs, knock
        their metals, retry the parked queue (ADR 0001, Q5d) — and, FIRST,
        sweep for idle metal: observe every carve-able metal and release
        what has sat past its limit (ADR 0003). Returns
        {"listings": {host: alive | recovered | reaped}, "knocked":
        {metal: answered}, "runs": {run_id: rerouted | parked},
        "released": [metal, ...]}."""
        return await self._transport.call("reap", {"probes": probes,
                                                   "wait": wait})

    async def migrate(self, run_ids: Sequence[str], *, optim: str = "load",
                      remaining_only: bool = False) -> dict:
        """Warm-fork each run onto the current code from its ledger tail —
        the code-refresh pass, one frame. The desk does everything; the reply
        maps parent run_id -> its child's acceptance (or refusal)."""
        return await self._transport.call("migrate", {
            "run_ids": list(run_ids), "optim": optim,
            "remaining_only": remaining_only})
