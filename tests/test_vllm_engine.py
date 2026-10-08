"""VllmEngine's own logic over stub `transformers` and `vllm` modules: the
AUTO eager rule (CUDA graphs unless a served adapter type demands eager, an
eager rebuild if the graphed build fails, an explicit choice never changed)
and the DELTA stream (each step's new tokens, logprobs and text read once,
reassembled exactly)."""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from rlstack import Message, Role
from rlstack.policy.adapters.rollout import BuildDemands
from rlstack.policy.compile import Bundle
from rlstack.runner.interfaces import FinishEvent, TokenEvent
from rlstack.spec.specs import SamplingSpec

BUILT: list[dict] = []
FAIL_GRAPHED = [False]


class StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


@dataclass
class StubCompletion:
    token_ids: list
    logprobs: list
    text: str
    finish_reason: str | None = None
    stop_reason: object = None


class StubLLM:
    """Streams a scripted completion in DELTA steps of two tokens."""

    script = [(65, "A"), (66, "B"), (67, "C"), (68, "D"), (69, "E")]

    def __init__(self, args):
        self.args = args
        self.params = None

    async def generate(self, prompt, params, request_id, **kwargs):
        self.params = params
        for start in range(0, len(self.script), 2):
            chunk = self.script[start:start + 2]
            last = start + 2 >= len(self.script)
            yield SimpleNamespace(outputs=[StubCompletion(
                token_ids=[t for t, _ in chunk],
                logprobs=[{t: SimpleNamespace(logprob=-0.5 * (start + k))}
                          for k, (t, _) in enumerate(chunk)],
                text="".join(c for _, c in chunk),
                finish_reason="stop" if last else None)])


def stub_modules() -> dict[str, types.ModuleType]:
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda base: StubTokenizer())
    transformers.AutoConfig = SimpleNamespace(
        from_pretrained=lambda base: SimpleNamespace(vision_config=None))
    vllm = types.ModuleType("vllm")

    def from_engine_args(args):
        BUILT.append(dict(args))
        if FAIL_GRAPHED[0] and not args["enforce_eager"]:
            raise RuntimeError("graph capture failed")
        return StubLLM(args)

    vllm.AsyncEngineArgs = lambda **kwargs: kwargs
    vllm.AsyncLLMEngine = SimpleNamespace(from_engine_args=from_engine_args)
    vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    vllm.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.RequestOutputKind = SimpleNamespace(DELTA="delta", CUMULATIVE="cumulative")
    return {"transformers": transformers, "vllm": vllm, "vllm.sampling_params": sampling_params}


class EagerDemand:
    """A served adapter type whose lever cannot ride a captured graph."""

    adapter_type = "eager_kind"

    def demands(self):
        return BuildDemands(engine_args={"enforce_eager": True})


def engine(demanding: bool = False, **kwargs):
    from rlstack.runner.engines.vllm_engine import VllmEngine

    class Engine(VllmEngine):
        def _serving_adapter_types(self):
            return {"eager_kind": EagerDemand()} if demanding else {}

    return Engine("stub/base", serves=(), **kwargs)


def drain(eng, bundle_id="bundle:base"):
    async def go():
        return [event async for event in eng.sample_tokens(
            [Message(Role.USER, "hi")], SamplingSpec(temperature=1.0, top_p=1.0, max_tokens=8),
            (), bundle_id, 7)]
    return asyncio.run(go())


class VllmEngineTest(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, stub_modules())
        self.modules.start()
        sys.modules.pop("rlstack.runner.engines.vllm_engine", None)
        BUILT.clear()
        FAIL_GRAPHED[0] = False

    def tearDown(self):
        self.modules.stop()
        sys.modules.pop("rlstack.runner.engines.vllm_engine", None)

    def test_auto_builds_cuda_graphs_unless_a_served_type_demands_eager(self):
        self.assertFalse(engine()._engine_args["enforce_eager"])
        self.assertTrue(engine().eager_auto)
        demanded = engine(demanding=True)
        self.assertTrue(demanded._engine_args["enforce_eager"])
        self.assertFalse(demanded.eager_auto)
        self.assertTrue(engine(enforce_eager=True)._engine_args["enforce_eager"])
        self.assertFalse(engine(enforce_eager=False)._engine_args["enforce_eager"])
        with self.assertRaises(NotImplementedError):     # a demand never overrides a build fact
            engine(demanding=True, enforce_eager=False)

    def test_a_failed_graphed_auto_build_is_rebuilt_eager_once(self):
        FAIL_GRAPHED[0] = True
        eng = engine()
        eng.add_bundle(Bundle("bundle:base", {}))
        drain(eng)
        self.assertEqual([args["enforce_eager"] for args in BUILT], [False, True])
        self.assertIn("graph capture failed", eng.eager_fallback)
        self.assertTrue(eng.first_contact()["enforce_eager"])

    def test_an_explicit_graphed_build_that_fails_is_not_changed(self):
        FAIL_GRAPHED[0] = True
        eng = engine(enforce_eager=False)
        eng.add_bundle(Bundle("bundle:base", {}))
        with self.assertRaises(RuntimeError):
            drain(eng)
        self.assertEqual(len(BUILT), 1)

    def test_the_delta_stream_reassembles_every_token_logprob_and_the_text(self):
        eng = engine(log_stats=True)
        self.assertFalse(eng._engine_args["disable_log_stats"])
        eng.add_bundle(Bundle("bundle:base", {}))
        events = drain(eng)
        tokens = [e for e in events if isinstance(e, TokenEvent)]
        self.assertEqual([e.token_id for e in tokens], [65, 66, 67, 68, 69])
        self.assertEqual([e.logprob for e in tokens], [0.0, -0.5, -1.0, -1.5, -2.0])
        self.assertEqual("".join(e.text_delta for e in tokens), "ABCDE")
        self.assertIsInstance(events[-1], FinishEvent)
        self.assertEqual(events[-1].finish, "eos")
        self.assertEqual(eng._llm.params.output_kind, "delta")
        self.assertEqual(eng.meter.decode_tokens, 5)


if __name__ == "__main__":
    unittest.main()
