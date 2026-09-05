"""Fake metal: a deterministic Engine and Learner behind the real protocols.

Stand-ins a caller hands to a Host in place of real metal. They make the whole
runner — identity, seeds, waves, the commit protocol, resume — executable and
byte-for-byte reproducible with no GPU, and they dry-run an experiment in
seconds before it touches hardware. The contract they keep: every output is a
pure function of (seed, inputs) or of the digest state threaded through the
learner, never of wall clock, global RNG or scheduling order. Resume-
equivalence depends on it.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from collections.abc import AsyncIterator, Mapping, Sequence

from rlstack.data.flatten import TokenBatch
from rlstack.data.trajectory import Message
from rlstack.policy.adapters.base import (
    AdapterType, Directive, Mechanism, adapter_type,
)
from rlstack.policy.adapters.rollout import Request
from rlstack.policy.compile import Bundle
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.interfaces import (
    Emitted, FinishEvent, Parameterization, TokenEvent, TrainStats,
)
from rlstack.runner.meters import TrafficMeter
from rlstack.runner.seeds import derive
from rlstack.spec.canonical import content_hash
from rlstack.spec.specs import SamplingSpec

_ARITH = re.compile(r"(\d+)\s*\+\s*(\d+)")


class FakeEngine:
    """Char-level deterministic engine.

    A prompt containing "a+b" is answered correctly with probability
    `p_correct` (a per-seed coin), wrongly otherwise — enough reward variance
    to make group normalization nontrivial. `record_draws=True` deposits a
    fake per-token "adapter_draw" into TokenEvent.extras and
    `record_latent=True` deposits fake PER-REQUEST facts into the stream's
    FinishEvent — the two granularities of the recording channel, both
    exercisable with no GPU. `plugins` is this fake build's set of installed
    plugin mechanisms — reachability honestly reports NONE for a plugin
    mechanism that is not in it.
    """

    def __init__(self, p_correct: float = 0.5, record_draws: bool = False,
                 record_latent: bool = False,
                 plugins: frozenset[Mechanism] = frozenset(),
                 base: str | None = None, tp: int = 1,
                 sleeps: bool = False) -> None:
        self.base = base            # None: fake metal serves any base
        self.tp = tp                # build fact: a fake TP-2 engine is tp=2
        # build fact, as on the real engine: can this build hand the device
        # back? A resident's hello reports it and the host wires the
        # alternation hooks only when it says yes. `naps` records the calls,
        # so a test can prove the hooks crossed the door.
        self.sleeps = sleeps
        self.naps: list[str] = []
        self.down = False
        self.p_correct = p_correct
        self.record_draws = record_draws
        self.record_latent = record_latent
        self.plugins = plugins
        # how many requests this engine has answered — what makes
        # `first_contact` empty until there has BEEN a first contact
        self.served = 0
        self.bundle_log: list[str] = []       # every add_bundle, in order
        self._known: set[str] = set()
        # bundle -> the adapter types its payloads carry: what a request's
        # directives are recorded against (record_directives below)
        self._carries: dict[str, tuple[str, ...]] = {}
        self.directives_seen: list[tuple[Directive, ...]] = []
        # the same traffic seam the real engine has, so the whole emission
        # plane is exercisable without a GPU. Token counts stay a pure
        # function of the inputs; only the TTFT gap is wall clock, and it
        # reaches the host journal alone — never a run directory.
        self.meter = TrafficMeter()

    # ---- the sleep seam (a door verb, not an Engine verb) --------------------

    async def sleep(self) -> None:
        self.naps.append("sleep")

    async def wake(self) -> None:
        self.naps.append("wake")

    def shutdown(self) -> None:
        """A resident's last verb; nothing to release on a fake, but the fact
        that it was said is what a teardown test reads."""
        self.down = True

    # ---- Engine protocol ----------------------------------------------------

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(ord(c) for c in text)

    def first_contact(self) -> dict:
        """The fake's own measurement (ADR 0008, F5): nothing until it has
        answered a request, and then numbers that are a pure function of what
        it served — so the whole first-contact path (the door verb, the
        host's journal line, the comparison against the declared GB) is
        exercisable with no GPU, which is where every OOM would have been
        caught cheaply."""
        if not self.served:
            return {}
        return {"weights_gb": 0.0, "kv_tokens": self.served * 16,
                "requests": self.served}

    def reachability(self, sites: Sequence[SiteMeta]) -> Mapping[str, Mechanism]:
        """A realistic fake build: punica reaches weighted matrices, prompt
        rows enter at the embedding boundary, logits have a processor, and the
        two plugin levers — the side-attention rectangle, the residual
        boundaries — are reachable only when their plugin is installed.
        Everything else is honestly NONE."""
        def reach(meta: SiteMeta) -> Mechanism:
            if meta.has_weight:
                return Mechanism.PUNICA
            if meta.path == "model.embed_tokens":
                return Mechanism.PROMPT_EMBEDS
            if meta.name == "logits":
                return Mechanism.LOGITS
            if meta.path == "attn_scores":
                return (Mechanism.SIDE_ATTENTION
                        if Mechanism.SIDE_ATTENTION in self.plugins
                        else Mechanism.NONE)
            if meta.is_boundary and (meta.path.startswith("model.layers.")
                                     or meta.path == "model.norm"):
                return (Mechanism.RESIDUAL
                        if Mechanism.RESIDUAL in self.plugins
                        else Mechanism.NONE)
            return Mechanism.NONE
        return {meta.name: reach(meta) for meta in sites}

    def add_bundle(self, bundle: Bundle) -> None:
        if bundle.bundle_id in self._known:
            return  # additive AND idempotent (the invariant)
        self.bundle_log.append(bundle.bundle_id)
        self._known.add(bundle.bundle_id)
        self._carries[bundle.bundle_id] = tuple(
            dict.fromkeys(bundle.adapter_types[name]
                          for name in sorted(bundle.adapter_types)))

    def record_directives(self, bundle_id: str, token_ids: tuple[int, ...],
                          seed: int | None,
                          directives: Sequence[Directive]) -> dict:
        """The recording half of the real bus's apply, adapter-blind: every
        adapter type the bundle carries is asked what it records about this
        request's directive (its own, or none), exactly as VllmEngine asks
        each attached lowering — so a directive round-trips to the seal with
        no GPU, and the fact replay reads is the real adapter type's."""
        request = Request(token_ids=token_ids, seed=seed, occupied=0,
                          directives=tuple(directives))
        facts: dict = {}
        for name in self._carries.get(bundle_id, ()):
            adapter_type = ADAPTER_TYPES.get(name).instance
            facts.update(adapter_type.record_directive(
                adapter_type.directive_for(request), request))
        return facts

    def knows_bundle(self, bundle_id: str) -> bool:
        """The fake never evicts — it has no residency to bound — so this is
        exactly "was it ever registered here"."""
        return bundle_id in self._known

    async def score_tokens(self, messages: Sequence[Message],
                           token_ids: Sequence[int],
                           bundle_id: str,
                           directives: Sequence[Directive] = (),
                           ) -> tuple[float, ...]:
        """Deterministic scoring: a pure function of (bundle, context, token,
        position) — no RNG state, so scoring never perturbs sampling. A
        directive is validated (one per adapter type) and records nothing:
        score traffic seals no turn."""
        if bundle_id not in self._known:
            raise RuntimeError(f"bundle {bundle_id!r} was never registered")
        self.directives_seen.append(tuple(directives))
        context = "".join(m.content for m in messages)
        self.record_directives(bundle_id, tuple(ord(c) for c in context),
                               None, directives)
        self.served += 1
        self.meter.opened_request(len(context) + len(token_ids))
        return tuple(
            -0.2 - 0.5 * (int(content_hash({
                "bundle": bundle_id, "ctx": context,
                "tok": int(tok), "pos": pos})[:6], 16) / 16 ** 6)
            for pos, tok in enumerate(token_ids))

    async def sample_tokens(
        self,
        messages: Sequence[Message],
        sampling: SamplingSpec,
        stop: tuple[str, ...],
        bundle_id: str,
        seed: int,
        directives: Sequence[Directive] = (),
    ) -> AsyncIterator[TokenEvent | FinishEvent]:
        if bundle_id not in self._known:
            raise RuntimeError(f"bundle {bundle_id!r} was never registered")
        self.directives_seen.append(tuple(directives))
        rng = random.Random(seed)
        text = self._completion(messages[-1].content, rng)
        # char-level metal: one token per character, so the prompt's token
        # count is its length — known before the first token, as on real metal
        prompt_ids = tuple(ord(c) for m in messages for c in m.content)
        self.served += 1
        self.meter.opened_request(len(prompt_ids))
        submitted = time.time()

        turn_extras = {**self.latent_draw(seed),
                       **self.record_directives(bundle_id, prompt_ids, seed,
                                                directives)}

        emitted = ""
        for ch in text:
            if len(emitted) >= sampling.max_tokens:
                yield FinishEvent("length", turn_extras=turn_extras)
                return
            if not emitted:
                self.meter.first_token_after(time.time() - submitted)
            emitted += ch
            self.meter.decoded(1)
            extras = {"adapter_draw": rng.randrange(4)} if self.record_draws else {}
            yield TokenEvent(token_id=ord(ch), logprob=-(0.2 + rng.random() / 2),
                             text_delta=ch, extras=extras)
            hit = next((s for s in stop if emitted.endswith(s)), None)
            if hit is not None:
                yield FinishEvent("stop", stop_hit=hit, turn_extras=turn_extras)
                return
        yield FinishEvent("eos", turn_extras=turn_extras)

    def latent_draw(self, seed: int) -> dict:
        """Fake PER-REQUEST facts, the FinishEvent mirror of record_draws.

        A probabilistic adapter type draws once per request and the draw seals
        into Turn.turn_extras, so the membrane needs a stdlib-only way to be
        exercised at that granularity — one vector and one index, the two
        shapes a real one records. Drawn off the SEED TREE rather than the
        token stream's generator, so switching it on shifts no other byte of a
        run.
        """
        if not self.record_latent:
            return {}
        rng = random.Random(derive(seed, "fake-latent"))
        return {"latent_draw": [round(rng.uniform(-1.0, 1.0), 6)
                                for _ in range(4)],
                "member_draw": rng.randrange(4)}

    # ---- generation policy --------------------------------------------------

    def _completion(self, prompt: str, rng: random.Random) -> str:
        match = _ARITH.search(prompt)
        if match is None:
            return str(rng.randrange(100))
        answer = int(match.group(1)) + int(match.group(2))
        if rng.random() < self.p_correct:
            return str(answer)
        return str(answer + 1 + rng.randrange(9))


def fake_initial_payload(sites: Sequence[SiteMeta],
                         init: Mapping[str, object]) -> bytes:
    """THE FAKE WORLD'S INIT FUNCTION: version 0 of one bank entry as bytes, a
    pure digest of the entry's resolved sites and its (seed-carrying) init.

    The real one is `AdapterType.initial_payload`, which needs torch; this is
    the same rule with the numerics taken out, and it is the ONE definition
    both fake sides call — `FakeLearner` for a frozen entry it was asked to
    install, `FakeAdapter.emit` for the same entry built with no learner at
    all. That is what lets the stdlib-only suite pin ADR 0006 Part B's
    promise: a learner-built v0 and a learner-less v0 compile to the same
    bundle id. Entry NAMES do not enter it, exactly as they do not enter a
    real adapter type's params: two entries with the same sites and the same
    seed ARE the same delta.
    """
    return ("fake-init:" + content_hash({
        "sites": [meta.name for meta in sites],
        "init": {key: init[key] for key in sorted(init)},
    })[:16]).encode()


@dataclass
class FakeParams:
    """What FakeAdapter builds: the payload, and nothing else. A fake adapter
    type has no tensors, so its parameterization IS its emitted bytes."""

    payload: bytes


@adapter_type("fake")
class FakeAdapter(AdapterType):
    """A stdlib adapter type: the fake world's delta, servable through punica.

    FakeEngine / FakeLearner make the runner executable with no GPU; this
    makes the POLICY executable with no GPU, which is what a test of the init
    function needs — every real adapter type's `params` builds tensors. It
    lowers nothing (a fake bank is never really served) and trains nothing;
    what it is for is the seam: `initial_payload` here and `FakeLearner`'s
    frozen payload are one function (ADR 0006 Part B).
    """

    serving = Mechanism.PUNICA

    def site_ok(self, meta: SiteMeta) -> bool:
        """Wherever a weight is — the punica lever's own rule."""
        return meta.has_weight

    def params(self, sites: tuple[SiteMeta, ...], init: dict) -> FakeParams:
        return FakeParams(fake_initial_payload(sites, init))

    def emit(self, params: FakeParams) -> bytes:
        return params.payload

    def load(self, params: FakeParams, payload: bytes) -> None:
        params.payload = payload


@dataclass
class _FakeTenant:
    """One experiment's digest state on this fake learner."""

    trainable: list[str]
    all_names: list[str]
    provides: list[str]                 # every name the bank DECLARES it provides
    frozen: dict[str, bytes]            # v0 payloads, by the fake init function
    init: str
    state: str
    steps: int = 0


class FakeLearner:
    """Digest-state learner: no numerics, honest choreography, multi-tenant.

    Per-tenant state is a content hash folded over every batch and step, so
    emitted payloads change exactly when training happened and `load` restores
    the precise pre-crash state — which is what the resume-equivalence tests
    bite on. The tenant key never enters the digests: a single-tenant run's
    bytes are identical whether or not anyone shares the learner (the tenancy
    invariant, testable). Frozen deltas emit `fake_initial_payload` — the fake
    world's init function, which is also what a run with NO learner builds its
    v0 with, so the two agree by construction (ADR 0006 Part B, Q6).
    """

    def __init__(self, fsdp: int = 1, sleeps: bool = False,
                 sleep_refusal: str = "") -> None:
        self.fsdp = fsdp            # build fact: a fake 2-shard learner is fsdp=2
        self.sleeps = sleeps        # build fact: see FakeEngine.sleeps
        # and why not, when not: the real sharded learner's is a fact about
        # its torch (fsdp_torch.probe_sharded_sleep), so a fake carries the
        # string rather than deriving one
        self.sleep_refusal = sleep_refusal
        # how many forwards this learner has run — what makes `first_contact`
        # empty until there has BEEN a first forward
        self.forwards = 0
        self.naps: list[str] = []
        self.down = False
        self._tenants: dict[str, _FakeTenant] = {}

    async def sleep(self) -> None:
        self.naps.append("sleep")

    async def wake(self) -> None:
        self.naps.append("wake")

    def shutdown(self) -> None:
        """A resident's last verb; see FakeEngine.shutdown."""
        self.down = True

    def _tenant(self, tenant: str) -> _FakeTenant:
        if tenant not in self._tenants:
            raise KeyError(f"tenant {tenant!r} was never installed")
        return self._tenants[tenant]

    # ---- Learner protocol ---------------------------------------------------

    def install(self, tenant: str, parameterization: Parameterization) -> None:
        entries = {entry.name: entry for entry in parameterization.entries}
        all_names = sorted(entries)
        # the init digest folds the DERIVED per-entry seeds: a different
        # master seed is a different init, exactly as before, and the master
        # itself never reaches a learner (ADR 0002, Q2)
        init = content_hash({
            "seeds": {name: entries[name].init.get("seed") for name in all_names},
            "bank": {name: entries[name].adapter_type for name in all_names},
            "sites": {name: [m.name for m in entries[name].sites]
                      for name in all_names},
        })
        self._tenants[tenant] = _FakeTenant(
            trainable=sorted(name for name, e in entries.items() if e.trainable),
            all_names=all_names,
            provides=sorted({
                name for e in entries.values()
                for name in ADAPTER_TYPES.get(e.adapter_type).instance.provides}),
            # a frozen entry is built at install and never moves, so its
            # payload is the fake init function's — the same bytes a run with
            # no learner builds for it (ADR 0006 Part B)
            frozen={name: fake_initial_payload(entry.sites, entry.init)
                    for name, entry in entries.items() if not entry.trainable},
            init=init, state=init)

    def uninstall(self, tenant: str) -> None:
        """The tenancy ends and its digest state goes with it; a tenant
        nobody installed is already uninstalled (the protocol's rule)."""
        self._tenants.pop(tenant, None)

    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats:
        state = self._tenant(tenant)
        self.forwards += 1
        state.state = content_hash({
            "state": state.state,
            "ids": batch.token_ids,
            "mask": batch.loss_mask,
            "post": {k: batch.postdata[k] for k in sorted(batch.postdata)},
            "blp": batch.behavior_logprobs,
        })
        return TrainStats(
            loss=int(state.state[:8], 16) / 16 ** 8,
            mean_ratio=1.0,
            logprob_gap=0.0,
            grad_norm=int(state.state[8:16], 16) / 16 ** 8,
            tokens=len(batch),
            provided=self.provided_digest(state),
        )

    def provided_digest(self, state: _FakeTenant) -> dict[str, float]:
        """One digest-derived float per DECLARED provided name.

        No numerics, honest choreography — the fake's charter. What it exercises
        is the emission path: a bank that declares a provided tensor makes that
        name reach TrainStats and then the ledger's train block, so the whole
        route from `provides` to a run's own dictionary is testable with no GPU.
        The values are a pure function of the digest state, like everything else
        here, so resume-equivalence is untouched.
        """
        return {name: int(content_hash({"state": state.state,
                                        "provided": name})[:8], 16) / 16 ** 8
                for name in state.provides}

    def optim_step(self, tenant: str) -> None:
        state = self._tenant(tenant)
        state.steps += 1
        state.state = content_hash({"state": state.state, "step": state.steps})

    def emit(self, tenant: str) -> Emitted:
        state = self._tenant(tenant)
        adapters = {
            name: (f"fake-delta:{name}:{state.state}".encode()
                   if name in state.trainable else state.frozen[name])
            for name in state.all_names
        }
        optim = {name: f"fake-optim:{name}:{state.steps}:{state.state}".encode()
                 for name in state.trainable}
        return Emitted(adapters=adapters, optim=optim)

    def first_contact(self) -> dict:
        """The fake learner's own measurement (ADR 0008, F5): empty until it
        has run a forward, then a per-rank peak that is a pure function of
        the widths it was built at — enough to exercise the journal line the
        real learner's `torch.cuda.max_memory_allocated` fills in."""
        if not self.forwards:
            return {}
        return {"peak_gb": [round(0.25 * self.forwards, 4)] * self.fsdp,
                "forwards": self.forwards}

    def load(self, tenant: str, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None:
        state = self._tenant(tenant)
        states = {payload.decode().rsplit(":", 1)[1] for name, payload
                  in adapters.items() if name in state.trainable}
        if len(states) > 1:
            raise ValueError(f"inconsistent adapter payloads: {sorted(states)}")
        if states:
            state.state = states.pop()
        if optim:
            steps = {int(p.decode().split(":")[2]) for p in optim.values()}
            if len(steps) != 1:
                raise ValueError(f"inconsistent optim payloads: {sorted(steps)}")
            state.steps = steps.pop()
        else:
            state.steps = 0
