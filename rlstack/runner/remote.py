"""The wire: pools served by ANOTHER host, behind the same Engine protocol.

HostService is the host-side end — it executes pool verbs on its own metal
under its own arbiter, because admission stays with the partition, and engines
are addressed by CAPABILITY (base, tp), never by pool name. A Transport carries
JSON-safe dict frames; RemotePool implements the whole Engine protocol over it,
so the runner cannot tell remote from local. The verb split is the contract
every transport implements against:

    call (async)   sample_tokens / score_tokens — they occupy the metal, so
                   the service wraps each in the owning host's arbiter.admit.
    ask  (sync)    add_bundle / reachability / tokenize — additive
                   registration and build facts, which by the tenancy
                   invariant never disturb traffic, so they need no admission
                   and may run from sync call sites.

Costs, stated: sample replies are non-streamed (one reply carries the whole
event list), add_bundle ships payload bytes as base64, and the learner is never
remote ACROSS HOSTS — the runner goes to it. Since ADR 0002 the learner IS
behind a wire INSIDE its host: every resident (engine or learner) is a child
process of the metal, so `EngineService`/`LearnerService` are the resident's
end of that door and `RemoteLearner` is the Host's end for its learner
(`RemotePool` already was for an engine). `HostService` keeps admission and
forwards through the proxy, so the two hops are: admission at the host, then
the resident. LocalTransport round-trips every frame through json in both
directions, so anything that works over it works over a real transport.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict
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
            "microbatches_in_update": batch.microbatches_in_update}


def decode_token_batch(row: Mapping) -> TokenBatch:
    return TokenBatch(
        token_ids=tuple(row["token_ids"]), loss_mask=tuple(row["loss_mask"]),
        behavior_logprobs=tuple(row["behavior_logprobs"]),
        segment_ids=tuple(row["segment_ids"]),
        doc_starts=tuple(row["doc_starts"]),
        postdata={k: tuple(v) for k, v in row["postdata"].items()},
        token_extras={k: tuple(v) for k, v in row["token_extras"].items()},
        doc_turn_extras=tuple(tuple(doc) for doc in row["doc_turn_extras"]),
        microbatches_in_update=int(row["microbatches_in_update"]))


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


class Transport(Protocol):
    """Carries dict frames to one host's service. Frames are JSON-safe by
    contract; LocalTransport enforces it, real transports inherit it free."""

    async def call(self, verb: str, payload: dict) -> dict:
        """An admitted verb: the serving host wraps it in its arbiter."""
        ...

    def ask(self, verb: str, payload: dict) -> dict:
        """An admission-free verb: registration and build facts."""
        ...


class HostService:
    """The host-side end of the wire.

    Executes pool verbs on this host's engines under this host's arbiter — the
    physical resource owns admission, so a remote experiment is just one more
    source of admitted work and never gets a vote. Engines are addressed by
    CAPABILITY (base, tp), the fleet's demand vocabulary, never by pool name —
    pool names are an experiment's private routing."""

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
        so they resolve no engine."""
        if verb == "adopt":
            return await self.host.adopt(payload["spec"],
                                         payload.get("routes", {}),
                                         payload.get("code"),
                                         payload.get("subdir"))
        if verb == "stop":
            return await self.host.stop(payload["run_id"])
        engine = self._engine(payload["base"], payload["tp"])
        if not self.host.arbiter.is_attached(engine):
            self.host.arbiter.attach(
                engine, label=f"{self.host.name}:{payload['base'] or '*'}")
        if verb not in ("sample_tokens", "score_tokens"):
            raise ValueError(f"unknown admitted verb {verb!r}")
        async with self.host.arbiter.admit(engine):
            return await EngineService(engine).serve(verb, payload)

    def answer(self, verb: str, payload: dict) -> dict:
        """One admission-free verb: additive registration (add_bundle never
        disturbs traffic — the multi-tenancy invariant) and build facts
        (reachability, tokenize), all callable from sync call sites. `status`
        is host-addressed (the roster, the partition, the adoptions' fates)
        and resolves no engine."""
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
        raise ValueError(f"unknown admission-free verb {verb!r}")


class LearnerService:
    """ONE learner's five verbs as frames: the learner resident's end of its
    door. Every verb is synchronous and pins a tenant (I8), so all five ride
    the `ask` path and arrive in the order the Trainer issued them; nothing
    here admits — the Trainer admitted itself at its host's arbiter before
    the first frame."""

    def __init__(self, learner: Learner) -> None:
        self.learner = learner

    async def serve(self, verb: str, payload: dict) -> dict:
        raise ValueError(
            f"{verb!r}: a learner's verbs are synchronous — they ride ask, "
            f"never call")

    def answer(self, verb: str, payload: dict) -> dict:
        tenant = payload["tenant"]
        if verb == "install":
            self.learner.install(
                tenant, decode_parameterization(payload["parameterization"]))
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


class LocalTransport:
    """Same-process transport that still crosses the serialization boundary
    (json round-trip both ways), so a fleet whose hosts share one process is
    indistinguishable, from above, from one whose hosts do not."""

    def __init__(self, service: "HostService | EngineService | LearnerService"
                 ) -> None:
        self.service = service

    async def call(self, verb: str, payload: dict) -> dict:
        return _json_roundtrip(
            await self.service.serve(verb, _json_roundtrip(payload)))

    def ask(self, verb: str, payload: dict) -> dict:
        return _json_roundtrip(
            self.service.answer(verb, _json_roundtrip(payload)))


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
                 tp: int = 1) -> None:
        self.base = base
        self.tp = tp
        self.meter = TrafficMeter()
        self._transport = transport

    def _address(self) -> dict:
        return {"base": self.base, "tp": self.tp}

    async def sample_tokens(self, messages, sampling, stop, bundle_id,
                            seed, directives: Sequence[Directive] = (),
                            ) -> AsyncIterator[TokenEvent | FinishEvent]:
        reply = await self._transport.call("sample_tokens", {
            **self._address(), "messages": encode_messages(messages),
            "sampling": encode_sampling(sampling), "stop": list(stop),
            "bundle_id": bundle_id, "seed": seed,
            "directives": encode_directives(directives)})
        for event in decode_events(reply["events"]):
            yield event

    async def score_tokens(self, messages, token_ids, bundle_id,
                           directives: Sequence[Directive] = (),
                           ) -> tuple[float, ...]:
        reply = await self._transport.call("score_tokens", {
            **self._address(), "messages": encode_messages(messages),
            "token_ids": list(token_ids), "bundle_id": bundle_id,
            "directives": encode_directives(directives)})
        return tuple(reply["logprobs"])

    def add_bundle(self, bundle: Bundle) -> None:
        self._transport.ask("add_bundle", {
            **self._address(), "bundle": encode_bundle(bundle)})

    def knows_bundle(self, bundle_id: str) -> bool:
        """Admission-free, like the registration it guards: asking what a pool
        holds never disturbs traffic, and the answer is about the SERVING
        engine's residency — which is why it has to cross the wire rather than
        be remembered here."""
        reply = self._transport.ask("knows_bundle", {
            **self._address(), "bundle_id": bundle_id})
        return bool(reply["known"])

    def reachability(self, sites: Sequence[SiteMeta]) -> Mapping[str, Mechanism]:
        reply = self._transport.ask("reachability", {
            **self._address(), "sites": encode_sites(sites)})
        return {name: Mechanism[mech]
                for name, mech in reply["mechanisms"].items()}

    def tokenize(self, text: str) -> tuple[int, ...]:
        reply = self._transport.ask("tokenize", {
            **self._address(), "text": text})
        return tuple(reply["token_ids"])


class RemoteLearner:
    """A Learner served by a resident, reached through a Transport — the
    Host's end of its learner's door, and RemotePool's twin.

    Implements the whole Learner protocol; `fsdp` is the build fact the
    resident reported in its hello, which the host attests against its
    training regime exactly as it did against an in-process learner. Every
    verb is one `ask` frame, synchronous and blocking like the in-process
    call it replaces (ADR 0002, Q6): the wire adds custody, never
    scheduling."""

    def __init__(self, transport: Transport, *, fsdp: int = 1) -> None:
        self.fsdp = fsdp
        self._transport = transport

    def install(self, tenant: str, parameterization: Parameterization) -> None:
        self._transport.ask("install", {
            "tenant": tenant,
            "parameterization": encode_parameterization(parameterization)})

    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats:
        return decode_train_stats(self._transport.ask("forward_backward", {
            "tenant": tenant, "batch": encode_token_batch(batch)}))

    def optim_step(self, tenant: str) -> None:
        self._transport.ask("optim_step", {"tenant": tenant})

    def emit(self, tenant: str) -> Emitted:
        return decode_emitted(self._transport.ask("emit", {"tenant": tenant}))

    def load(self, tenant: str, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None:
        self._transport.ask("load", {
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
                    subdir: str | None = None) -> dict:
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
            "code": dict(code or {}), "subdir": subdir})

    async def stop(self, run_id: str) -> dict:
        """Tell the host to stop a tenancy — cancellation awaited host-side,
        so the reply means the death is complete and the run_id is free to
        adopt again, here or elsewhere (a reroute's first half)."""
        return await self._transport.call("stop", {"run_id": run_id})

    def status(self) -> dict:
        return self._transport.ask("status", {})


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

    def residual(self) -> list[float]:
        return list(self._transport.ask("residual", {})["residual"])

    def describe(self) -> dict:
        return self._transport.ask("describe", {})

    async def carve(self, request: Mapping) -> dict:
        return await self._transport.call("carve", dict(request))

    async def decarve(self, name: str) -> dict:
        return await self._transport.call("decarve", {"host": name})

    async def release(self) -> dict:
        """The acquire rung inverted at the metal (ADR 0003): every resident
        down the ladder, the books emptied, and the SHIFT ENDED so the venue
        reclaims the container. Idempotent — a bare or already-released
        metal answers released just the same, so a desk unsure whether its
        release landed may simply say it again."""
        return await self._transport.call("release", {})


class RemoteDesk:
    """The client end of the standing fleet: a campaign's whole surface.

    The desk is workload-blind, so the SHAPING happens here, client-side:
    `submit` turns a spec into demand rows (anchor on the learner) plus an
    opaque frame via the campaign layer, and one frame carries both to the
    desk — which places, delivers to the anchor, and answers with where
    everything landed (or what to boot). `resolve` is the pure client's
    verb: demands in, addresses out, no delivery. After either, a client
    watches the store — the desk holds no results, exactly as no host does."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def submit(self, spec: object,
                     subdir: str | None = None) -> dict:
        from rlstack.runner.campaign import demands_of, frame_for
        from rlstack.runner.desk import demand_rows

        return await self._transport.call("submit", {
            "demands": demand_rows(demands_of(spec)),
            "frame": frame_for(spec, subdir)})

    async def resolve(self, demands: Sequence) -> dict:
        """Demands in, addresses out — placement without a workload: the
        door for an evaluator, a scorer, any client that wants a pool."""
        from rlstack.runner.desk import demand_rows

        return await self._transport.call("place",
                                          {"demands": demand_rows(demands)})

    def status(self) -> dict:
        return self._transport.ask("status", {})

    def liveness(self) -> dict:
        """{host: alive} for every listing, probed by the desk just now."""
        return self._transport.ask("liveness", {})

    async def list_host(self, name: str, regimes: Sequence,
                        address: str, solo: bool = False,
                        partition: Mapping | None = None,
                        metal: str = "") -> dict:
        """A booted host enters the standing fleet over the wire — the
        phone-home half of the deploy contract: the container that stood the
        host tells the desk what it wears, where it answers, and (the
        capacity view) the partition row it was born onto and the metal it
        lives on."""
        return await self._transport.call("list", {
            "host": name, "address": address, "solo": solo,
            "partition": dict(partition) if partition else None,
            "metal": metal,
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

    def placements(self) -> dict:
        """The desk's current-binding table: the latest delivered placement
        per run_id — pools to listings, plus the archived demand rows and
        frame where the delivery carried them."""
        return self._transport.ask("placements", {})["placements"]

    async def register_metal(self, name: str, gpu: str, devices: int,
                             vram_gb: float, address: str,
                             builds: Mapping | None = None,
                             idle_s: float | None | Undeclared = DESK_DEFAULT,
                             ) -> dict:
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
        desk's default decides; None PINS it, never released."""
        payload = {"name": name, "gpu": gpu, "devices": devices,
                   "vram_gb": vram_gb, "address": address,
                   "builds": dict(builds) if builds else None}
        if not isinstance(idle_s, Undeclared):
            payload["idle_s"] = idle_s
        return await self._transport.call("metal", payload)

    async def release(self, name: str, reason: str = "released") -> dict:
        """Hand a metal back BY HAND — the acquire rung inverted at the desk
        (ADR 0003): its listings delisted, its residents down the ladder,
        its shift ended so the venue reclaims the container, and the row
        kept as inventory the next placement may knock awake. The desk's
        idle sweep issues this same verb on its own clock; this is the door
        for an operator who knows the metal is done sooner."""
        return await self._transport.call("release", {"metal": name,
                                                      "reason": reason})

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
