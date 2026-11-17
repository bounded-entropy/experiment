"""The engine's memory ask: a whole-device partition must still start vLLM.

vLLM refuses to start when the device's free memory is below the
gpu_memory_utilization it is asked for, and the engine process's own CUDA
context already holds part of the card, so a partition of the whole device
(memory 1.0: a PoolMember without vram_gb) failed every boot on an RTX PRO
6000 (2026-11-17). build_engine now caps the ask below the whole device.
The real VllmEngine is stood in by a recorder, so this runs without vLLM.
"""

from __future__ import annotations

import sys
import types
import unittest

from rlstack import Partition, Regime
from rlstack.runner.residents import (
    ENGINE_WHOLE_DEVICE_UTILIZATION, EngineBuild, build_engine, engine_memory_utilization,
)


class RecordingEngine:
    seen: list[dict] = []

    def __init__(self, base, **kwargs):
        RecordingEngine.seen.append({"base": base, **kwargs})


class EngineMemoryAsk(unittest.TestCase):
    def test_a_partition_keeps_its_fraction_below_the_cap(self):
        self.assertEqual(engine_memory_utilization(0.4), 0.4)
        self.assertEqual(engine_memory_utilization(ENGINE_WHOLE_DEVICE_UTILIZATION),
                         ENGINE_WHOLE_DEVICE_UTILIZATION)

    def test_a_whole_device_asks_for_less_than_the_whole_device(self):
        self.assertLess(engine_memory_utilization(1.0), 1.0)
        self.assertEqual(engine_memory_utilization(1.0), ENGINE_WHOLE_DEVICE_UTILIZATION)

    def test_build_engine_hands_vllm_the_capped_fraction(self):
        stub = types.ModuleType("rlstack.runner.engines.vllm_engine")
        stub.VllmEngine = RecordingEngine
        saved = sys.modules.get(stub.__name__)
        sys.modules[stub.__name__] = stub
        RecordingEngine.seen.clear()
        try:
            regime = Regime("main-tp1", "inference", "Qwen/Qwen3-0.6B")
            for memory, expected in ((1.0, ENGINE_WHOLE_DEVICE_UTILIZATION), (0.375, 0.375)):
                build_engine(regime, Partition("m", (0,), memory, "L4"), EngineBuild(), store=types.SimpleNamespace(cas_get=None))
                self.assertEqual(RecordingEngine.seen[-1]["gpu_memory_utilization"], expected)
        finally:
            if saved is None:
                sys.modules.pop(stub.__name__, None)
            else:
                sys.modules[stub.__name__] = saved


if __name__ == "__main__":
    unittest.main()
