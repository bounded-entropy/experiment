"""Fake metal: a deterministic Engine and Learner behind the real protocols.

Two jobs. In tests, they make the entire runner — identity, seeds, waves, the
commit protocol, resume — executable and byte-for-byte reproducible with no
GPU. As `--fake` metal, they dry-run any experiment in seconds before it
touches real hardware.

Determinism contract: every output is a pure function of (seed, inputs) or of
the digest state threaded through the learner — never of wall clock, global
RNG, or scheduling order. The resume-equivalence tests depend on this.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from collections.abc import AsyncIterator, Mapping, Sequence

from rlstack.data.flatten import TokenBatch
from rlstack.data.trajectory import Message
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.compile import Bundle
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.interfaces import Emitted, FinishEvent, TokenEvent, TrainStats
from rlstack.spec.canonical import content_hash
from rlstack.spec.specs import ExperimentSpec, SamplingSpec

_ARITH = re.compile(r"(\d+)\s*\+\s*(\d+)")


class FakeEngine:
    """Char-level deterministic engine.

    A prompt containing "a+b" is answered correctly with probability
    `p_correct` (a per-seed coin), wrongly otherwise — enough reward variance
    to make group normalization nontrivial. `record_draws=True` deposits a
    fake per-token "adapter_draw" into TokenEvent.extras, exercising the
    recording channel end to end. `plugins` is this fake build's set of
    installed plugin mechanisms — reachability honestly reports NONE for a
    plugin mechanism that is not in it.
    """

    def __init__(self, p_correct: float = 0.5, record_draws: bool = False,
                 plugins: frozenset[Mechanism] = frozenset()) -> None:
        self.p_correct = p_correct
        self.record_draws = record_draws
        self.plugins = plugins
        self.bundle_log: list[str] = []       # every add_bundle, in order
        self._known: set[str] = set()

    # ---- Engine protocol ----------------------------------------------------

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(ord(c) for c in text)

    def reachability(self, sites: Sequence[SiteMeta]) -> Mapping[str, Mechanism]:
        """A realistic fake build: punica reaches weighted matrices, prompt
        rows enter at the embedding boundary, logits have a processor, and the
        side-attention rectangle is reachable only when the plugin is
        installed. Everything else is honestly NONE."""
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
            return Mechanism.NONE
        return {meta.name: reach(meta) for meta in sites}

    def add_bundle(self, bundle: Bundle) -> None:
        if bundle.bundle_id in self._known:
            return  # additive AND idempotent (the invariant)
        self.bundle_log.append(bundle.bundle_id)
        self._known.add(bundle.bundle_id)

    async def sample_tokens(
        self,
        messages: Sequence[Message],
        sampling: SamplingSpec,
        stop: tuple[str, ...],
        bundle_id: str,
        seed: int,
    ) -> AsyncIterator[TokenEvent | FinishEvent]:
        if bundle_id not in self._known:
            raise RuntimeError(f"bundle {bundle_id!r} was never registered")
        rng = random.Random(seed)
        text = self._completion(messages[-1].content, rng)

        emitted = ""
        for ch in text:
            if len(emitted) >= sampling.max_tokens:
                yield FinishEvent("length")
                return
            emitted += ch
            extras = {"adapter_draw": rng.randrange(4)} if self.record_draws else {}
            yield TokenEvent(token_id=ord(ch), logprob=-(0.2 + rng.random() / 2),
                             text_delta=ch, extras=extras)
            hit = next((s for s in stop if emitted.endswith(s)), None)
            if hit is not None:
                yield FinishEvent("stop", stop_hit=hit)
                return
        yield FinishEvent("eos")

    # ---- generation policy --------------------------------------------------

    def _completion(self, prompt: str, rng: random.Random) -> str:
        match = _ARITH.search(prompt)
        if match is None:
            return str(rng.randrange(100))
        answer = int(match.group(1)) + int(match.group(2))
        if rng.random() < self.p_correct:
            return str(answer)
        return str(answer + 1 + rng.randrange(9))


@dataclass
class _FakeTenant:
    """One experiment's digest state on this fake learner."""

    trainable: list[str]
    all_names: list[str]
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
    invariant, testable). Frozen deltas emit a constant init-derived payload.
    """

    def __init__(self) -> None:
        self._tenants: dict[str, _FakeTenant] = {}

    def _tenant(self, tenant: str) -> _FakeTenant:
        if tenant not in self._tenants:
            raise KeyError(f"tenant {tenant!r} was never installed")
        return self._tenants[tenant]

    # ---- Learner protocol ---------------------------------------------------

    def install(self, tenant: str, spec: ExperimentSpec,
                resolved_sites: Mapping[str, tuple[SiteMeta, ...]]) -> None:
        all_names = sorted(spec.policy.bank)
        init = content_hash({
            "master": spec.seeds.master,
            "bank": {name: spec.policy.bank[name].kind for name in all_names},
            "sites": {name: [m.name for m in resolved_sites[name]]
                      for name in all_names},
        })
        self._tenants[tenant] = _FakeTenant(
            trainable=sorted(name for name, a in spec.policy.bank.items()
                             if a.trainable),
            all_names=all_names, init=init, state=init)

    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats:
        state = self._tenant(tenant)
        state.state = content_hash({
            "state": state.state,
            "ids": batch.token_ids,
            "mask": batch.loss_mask,
            "post": {k: batch.post[k] for k in sorted(batch.post)},
            "blp": batch.behavior_logprobs,
        })
        return TrainStats(
            loss=int(state.state[:8], 16) / 16 ** 8,
            mean_ratio=1.0,
            logprob_gap=0.0,
            grad_norm=int(state.state[8:16], 16) / 16 ** 8,
            tokens=len(batch),
        )

    def optim_step(self, tenant: str) -> None:
        state = self._tenant(tenant)
        state.steps += 1
        state.state = content_hash({"state": state.state, "step": state.steps})

    def emit(self, tenant: str) -> Emitted:
        state = self._tenant(tenant)
        adapters = {
            name: (f"fake-delta:{name}:"
                   f"{state.state if name in state.trainable else state.init}"
                   ).encode()
            for name in state.all_names
        }
        optim = {name: f"fake-optim:{name}:{state.steps}:{state.state}".encode()
                 for name in state.trainable}
        return Emitted(adapters=adapters, optim=optim)

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
