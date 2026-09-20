"""ADR 0019, the engine side: the Library, `add_library` on every Engine, and
library / stacked routes — on the fakes, across the wire, and (torch-gated,
vLLM-free) the concatenation a stacked route is served as."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

from common import arith_store
from rlstack import Bundle, FakeEngine, FakeLearner, Message, Role, SamplingSpec
from rlstack.policy.adapters.dream_bank import Route
from rlstack.policy.adapters.rollout import (
    Levers, Library, Request, ServingBuild, check_rank_fits,
)
from rlstack.runner.fakes import library_names
from rlstack.runner.host import Host, Partition, Regime
from rlstack.runner.remote import HostService, LocalTransport, RemotePool
from rlstack.runner.residents import (
    EngineBuild, FakeEngineBuild, Resident, ResidentBirth, build_engine,
    decode_build, encode_build,
)

try:
    import torch
except ImportError:
    torch = None

BUNDLE = Bundle(bundle_id="bundle:dreams000", policy_version={"pi": 0},
                adapter_types={"pi": "dream_bank"})
PROMPT = [Message(Role.USER, "Question: 2+3?\nAnswer:")]
STACKED = Route("lib:mem/001+dreamer")


def go(coro):
    return asyncio.run(coro)


def sampled(engine, directives=(), seed: int = 7) -> list:
    async def collect():
        return [event async for event in engine.sample_tokens(
            PROMPT, SamplingSpec(temperature=1.0), (), BUNDLE.bundle_id, seed, directives)]
    return go(collect())


def scored(engine, directives=()) -> tuple[float, ...]:
    return go(engine.score_tokens(PROMPT, (53, 54), BUNDLE.bundle_id, directives))


def engine_with(*names: str, max_library: int = 32) -> FakeEngine:
    engine = FakeEngine(max_library=max_library)
    engine.add_bundle(BUNDLE)
    for name in names:
        engine.add_library(name, f"payload of {name}".encode())
    return engine


class LibraryTest(unittest.TestCase):
    def test_add_is_idempotent_and_a_name_is_write_once(self):
        library = Library(4)
        library.add("a", b"one")
        library.add("a", b"one")
        self.assertEqual(library.log, ["a"])
        with self.assertRaisesRegex(ValueError, "write-once"):
            library.add("a", b"two")
        self.assertEqual(library.get("a"), b"one")

    def test_the_least_recently_used_name_is_evicted_and_forgotten(self):
        forgotten: list[str] = []
        library = Library(2, forgotten.append)
        library.add("a", b"a")
        library.add("b", b"b")
        library.get("a")                                  # a is now the recent one
        library.add("c", b"c")
        self.assertEqual(forgotten, ["b"])
        self.assertEqual(library.names(), ("a", "c"))
        library.add("b", b"b")                            # back from the store: not a loss
        self.assertEqual(forgotten, ["b", "a"])

    def test_a_pinned_name_is_immune_and_does_not_shield_the_stale(self):
        forgotten: list[str] = []
        library = Library(2, forgotten.append)
        library.add("a", b"a")
        library.add("b", b"b")
        with library.pinned(("a",)):
            library.add("c", b"c")
            self.assertEqual(forgotten, ["b"])
            library.add("d", b"d")                        # soft by the in-flight set
            self.assertEqual(forgotten, ["b", "c"])
            self.assertTrue(library.knows("a"))
        library.add("e", b"e")
        self.assertEqual(forgotten, ["b", "c", "a"])

    def test_a_name_nobody_handed_in_is_refused_by_read_and_by_pin(self):
        library = Library(2)
        with self.assertRaisesRegex(RuntimeError, "add_library"):
            library.get("ghost")
        with self.assertRaisesRegex(RuntimeError, "'ghost'"):
            with library.pinned(("ghost",)):
                pass

    def test_levers_join_the_names_they_read(self):
        merged = Levers(library=("a",)).merged_with(Levers(library=("b",)))
        self.assertEqual(merged.library, ("a", "b"))

    def test_a_delta_wider_than_the_build_is_refused_with_both_numbers(self):
        build = ServingBuild(base="b", config=None, workdir=Path("."), max_bundles=2, max_rank=32)
        check_rank_fits("adapter 'x'", 32, build)
        with self.assertRaisesRegex(ValueError, r"rank 64.*max_rank 32"):
            check_rank_fits("adapter 'x'", 64, build)


class FakeEngineLibraryTest(unittest.TestCase):
    def test_the_route_grammar_names_its_library_parts(self):
        self.assertEqual(library_names((STACKED,)), ("mem/001",))
        self.assertEqual(library_names((Route("lib:a"),)), ("a",))
        self.assertEqual(library_names((Route("memory:03"),)), ())
        self.assertEqual(library_names(()), ())

    def test_add_library_is_idempotent_and_recorded(self):
        engine = engine_with("mem/001")
        engine.add_library("mem/001", b"payload of mem/001")
        self.assertEqual(engine.library.log, ["mem/001"])

    def test_a_route_naming_a_library_adapter_never_added_is_refused(self):
        engine = engine_with()
        for route in (Route("lib:mem/001"), STACKED):
            with self.assertRaisesRegex(RuntimeError, "mem/001.*add_library"):
                sampled(engine, (route,))
            with self.assertRaisesRegex(RuntimeError, "mem/001.*add_library"):
                scored(engine, (route,))

    def test_an_evicted_name_is_refused_until_its_bytes_come_back(self):
        engine = engine_with("a", "b", max_library=2)
        scored(engine, (Route("lib:a"),))                 # a is the recent one
        engine.add_library("c", b"payload of c")
        with self.assertRaisesRegex(RuntimeError, "'b'"):
            scored(engine, (Route("lib:b+dreamer"),))
        engine.add_library("b", b"payload of b")
        scored(engine, (Route("lib:b+dreamer"),))

    def test_sampling_under_a_stacked_route_is_deterministic_and_records_it(self):
        first, second = (sampled(engine_with("mem/001"), (STACKED,)) for _ in range(2))
        self.assertEqual(first, second)
        self.assertEqual(first[-1].turn_extras["route"], "lib:mem/001+dreamer")
        alone = sampled(engine_with("mem/001"), (Route("lib:mem/001"),))
        self.assertEqual(alone[-1].turn_extras["route"], "lib:mem/001")

    def test_a_score_is_a_function_of_the_named_payload(self):
        engine = engine_with("mem/001", "mem/002")
        under = scored(engine, (STACKED,))
        self.assertEqual(under, scored(engine_with("mem/001"), (STACKED,)))
        self.assertNotEqual(under, scored(engine, (Route("dreamer"),)))
        self.assertNotEqual(under, scored(engine, (Route("lib:mem/002+dreamer"),)))
        other = FakeEngine()
        other.add_bundle(BUNDLE)
        other.add_library("mem/001", b"another fit's bytes")
        self.assertNotEqual(under, scored(other, (STACKED,)))

    def test_traffic_naming_no_library_adapter_is_answered_as_before(self):
        bare, holding = engine_with(), engine_with("mem/001")
        for directives in ((), (Route("memory:02"),), (Route("base"),)):
            self.assertEqual(sampled(bare, directives), sampled(holding, directives))
            self.assertEqual(scored(bare, directives), scored(holding, directives))


class WireTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, _, _ = arith_store(tmp.name)
        self.payload = bytes(range(256)) * 3              # not text: base64 or bust

    def test_add_library_crosses_the_host_wire_as_bytes(self):
        engine = FakeEngine()
        host = Host("srv", engines=(engine,), learner=FakeLearner(), store=self.store)
        remote = RemotePool(LocalTransport(HostService(host)))
        remote.add_bundle(BUNDLE)
        with self.assertRaisesRegex(Exception, "add_library"):
            scored(remote, (STACKED,))
        remote.add_library("mem/001", self.payload)
        remote.add_library("mem/001", self.payload)
        self.assertEqual(engine.library.log, ["mem/001"])
        self.assertEqual(engine.library.get("mem/001"), self.payload)
        local = FakeEngine()
        local.add_bundle(BUNDLE)
        local.add_library("mem/001", self.payload)
        self.assertEqual(scored(remote, (STACKED,)), scored(local, (STACKED,)))
        self.assertEqual(sampled(remote, (STACKED,)), sampled(local, (STACKED,)))

    def test_add_library_crosses_a_resident_door(self):
        engine = FakeEngine(base="Qwen/Qwen3-0.6B")
        serve = Regime("serve", "inference", "Qwen/Qwen3-0.6B", 1)
        birth = ResidentBirth(label="alt:serve", partition=Partition("fake-metal", (0,), 1.0, "L4"),
                              regime=serve, build=FakeEngineBuild(), store=self.store.address())
        pool = RemotePool(Resident.in_process(birth, engine).transport, base="Qwen/Qwen3-0.6B")
        pool.add_library("mem/001", self.payload)
        self.assertEqual(engine.library.get("mem/001"), self.payload)

    def test_a_build_record_carries_max_library(self):
        self.assertEqual(EngineBuild().max_library, 32)
        for build in (EngineBuild(max_library=5, serves=("dream_bank",)),
                      FakeEngineBuild(max_library=5)):
            row = json.loads(json.dumps(encode_build(build)))
            self.assertEqual(row["max_library"], 5)
            self.assertEqual(decode_build(row), build)
        old_row = {k: v for k, v in encode_build(EngineBuild()).items() if k != "max_library"}
        self.assertEqual(decode_build(old_row).max_library, 32)
        engine = build_engine(Regime("serve", "inference", "b", 1),
                              Partition("m", (0,), 1.0, "L4"),
                              FakeEngineBuild(max_library=5), self.store)
        self.assertEqual(engine.library.capacity, 5)


# ---------------------------------------------------------------------------
# torch-gated, vLLM-free: the concatenation, the adapter dir, the lowering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StubLoRARequest:
    """vLLM's LoRARequest, as far as the lowering uses it."""

    lora_name: str
    lora_int_id: int
    lora_path: str


def dream_bank_vllm_module():
    """dream_bank's rollout lowering, imported over a stub `vllm.lora.request`
    where vLLM is not installed: the file's own logic is torch and files."""
    try:
        import vllm.lora.request  # noqa: F401
    except ImportError:
        for name in ("vllm", "vllm.lora", "vllm.lora.request"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["vllm.lora.request"].LoRARequest = StubLoRARequest
    from rlstack.policy.adapters import dream_bank_vllm
    return dream_bank_vllm


def forget_stub_vllm() -> None:
    if getattr(sys.modules.get("vllm.lora.request"), "LoRARequest", None) is StubLoRARequest:
        for name in ("vllm", "vllm.lora", "vllm.lora.request",
                     "rlstack.policy.adapters.dream_bank_vllm"):
            sys.modules.pop(name, None)


PATHS = ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj")
D_IN, D_OUT = 12, 10


def named_payload(r: int, seed: int, paths=PATHS) -> bytes:
    """One LoRA set as `lora_torch.emit` writes it, with B off zero."""
    from rlstack.policy.adapters import lora_torch
    from rlstack.policy.siteschema import SiteMeta
    sites = tuple(SiteMeta(p, p, True, (D_IN, D_OUT), False) for p in paths)
    state = lora_torch.build(sites, {"r": r, "seed": seed})
    generator = torch.Generator().manual_seed(seed + 100)
    for path in paths:
        state.b[path].data = torch.randn(D_OUT, r, generator=generator)
    return lora_torch.emit(state)


def delta_of(payload: bytes, path: str, x):
    from safetensors.torch import load as st_load
    tensors = st_load(payload)
    a = tensors[f"base_model.model.{path}.lora_A.weight"]
    b = tensors[f"base_model.model.{path}.lora_B.weight"]
    return x @ a.T @ b.T


@unittest.skipUnless(torch is not None, "torch required")
class StackFragmentsTest(unittest.TestCase):
    def test_the_concatenation_is_the_sum_of_the_two_deltas(self):
        from rlstack.policy.adapters import lora_torch
        one, two = named_payload(4, seed=1), named_payload(6, seed=2)
        stacked = lora_torch.stack_fragments(one, two)
        x = torch.randn(5, 3, D_IN, generator=torch.Generator().manual_seed(9))
        for path in PATHS:
            torch.testing.assert_close(
                delta_of(stacked, path, x),
                delta_of(one, path, x) + delta_of(two, path, x), rtol=1e-5, atol=1e-5)

    def test_the_adapter_dir_is_one_scaling_1_adapter_of_the_summed_rank(self):
        from safetensors.torch import load_file
        from rlstack.policy.adapters import lora_torch
        stacked = lora_torch.stack_fragments(named_payload(4, 1), named_payload(6, 2))
        with tempfile.TemporaryDirectory() as tmp:
            rank = lora_torch.write_adapter_dir(Path(tmp) / "stack", "Qwen/x", {"s": stacked})
            config = json.loads((Path(tmp) / "stack" / "adapter_config.json").read_text())
            tensors = load_file(Path(tmp) / "stack" / "adapter_model.safetensors")
        self.assertEqual((rank, config["r"], config["lora_alpha"]), (10, 10, 10))
        self.assertEqual(config["target_modules"], ["q_proj", "v_proj"])
        for path in PATHS:
            self.assertEqual(tuple(tensors[f"base_model.model.{path}.lora_A.weight"].shape), (10, D_IN))
            self.assertEqual(tuple(tensors[f"base_model.model.{path}.lora_B.weight"].shape), (D_OUT, 10))

    def test_a_site_only_one_part_covers_carries_that_part_alone(self):
        from safetensors.torch import load as st_load
        from rlstack.policy.adapters import lora_torch
        one, two = named_payload(4, 1), named_payload(6, 2, paths=PATHS[:1])
        stacked = st_load(lora_torch.stack_fragments(one, two))
        self.assertEqual(stacked[f"base_model.model.{PATHS[0]}.lora_A.weight"].shape[0], 10)
        self.assertEqual(stacked[f"base_model.model.{PATHS[1]}.lora_A.weight"].shape[0], 4)
        x = torch.randn(2, D_IN)
        torch.testing.assert_close(delta_of(st_save_of(stacked), PATHS[1], x),
                                   delta_of(one, PATHS[1], x))


def st_save_of(tensors) -> bytes:
    from safetensors.torch import save as st_save
    return st_save(tensors)


@unittest.skipUnless(torch is not None, "torch required")
class DreamBankRolloutLibraryTest(unittest.TestCase):
    R = 4

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = dream_bank_vllm_module()

    @classmethod
    def tearDownClass(cls) -> None:
        forget_stub_vllm()

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.library = Library(2, lambda name: self.lowering.forget_library(name))
        self.build = ServingBuild(base="Qwen/x", config=None, workdir=Path(tmp.name),
                                  max_bundles=2, max_rank=2 * self.R, max_members=2,
                                  library=self.library)
        self.lowering = self.module.DreamBankRollout(self.build)

    def bank_payload(self, seed: int = 0, memories: int = 1) -> bytes:
        from rlstack.policy.adapters import dream_bank_torch
        from rlstack.policy.siteschema import SiteMeta
        sites = tuple(SiteMeta(p, p, True, (D_IN, D_OUT), False) for p in PATHS)
        state = dream_bank_torch.build(sites, {"r": self.R, "memories": memories, "seed": seed})
        for lora in state.sets.values():
            for path in PATHS:
                lora.b[path].data = torch.randn(D_OUT, self.R)
        return dream_bank_torch.emit(state)

    def levers(self, attached, route: str) -> Levers:
        return self.lowering.apply(attached, Request(token_ids=(1, 2), directives=(Route(route),)))

    def adapter_payload(self, levers: Levers) -> bytes:
        return (Path(levers.kwargs["lora_request"].lora_path) / "adapter_model.safetensors").read_bytes()

    def test_plain_routes_are_served_as_before(self):
        attached = self.lowering.attach("bundle:v1", {"pi": self.bank_payload()})
        self.assertEqual(sorted(attached.sets), ["dreamer", "memory:00"])
        levers = self.levers(attached, "memory:00")
        self.assertIs(levers.kwargs["lora_request"], attached.sets["memory:00"])
        self.assertEqual((levers.turn_extras, levers.library), ({"route": "memory:00"}, ()))
        self.assertEqual(self.levers(attached, "base").kwargs, {})
        self.assertEqual(Path(attached.sets["memory:00"].lora_path).name, "memory_00")

    def test_a_library_route_is_its_own_adapter_shared_by_every_bundle(self):
        payload = named_payload(self.R, seed=5)
        self.library.add("mem/001", payload)
        v1 = self.lowering.attach("bundle:v1", {"pi": self.bank_payload(1)})
        v2 = self.lowering.attach("bundle:v2", {"pi": self.bank_payload(2)})
        levers = self.levers(v1, "lib:mem/001")
        self.assertEqual((levers.turn_extras, levers.library),
                         ({"route": "lib:mem/001"}, ("mem/001",)))
        self.assertIs(levers.kwargs["lora_request"],
                      self.levers(v2, "lib:mem/001").kwargs["lora_request"])
        x = torch.randn(3, D_IN)
        torch.testing.assert_close(delta_of(self.adapter_payload(levers), PATHS[0], x),
                                   delta_of(payload, PATHS[0], x))

    def test_a_stacked_route_is_one_adapter_per_bundle_and_is_the_sum(self):
        from safetensors.torch import save as st_save
        from rlstack.policy.adapters.dream_bank_torch import split_sets
        payload, bank = named_payload(self.R, seed=5), self.bank_payload(1)
        self.library.add("mem/001", payload)
        v1 = self.lowering.attach("bundle:v1", {"pi": bank})
        v2 = self.lowering.attach("bundle:v2", {"pi": self.bank_payload(2)})
        levers = self.levers(v1, "lib:mem/001+dreamer")
        stack = levers.kwargs["lora_request"]
        self.assertEqual(levers.library, ("mem/001",))
        self.assertEqual(levers.turn_extras, {"route": "lib:mem/001+dreamer"})
        self.assertIs(stack, self.levers(v1, "lib:mem/001+dreamer").kwargs["lora_request"])
        other = self.levers(v2, "lib:mem/001+dreamer").kwargs["lora_request"]
        self.assertNotEqual(stack.lora_int_id, other.lora_int_id)       # the dreamer moved
        config = json.loads((Path(stack.lora_path) / "adapter_config.json").read_text())
        self.assertEqual((config["r"], config["lora_alpha"]), (2 * self.R, 2 * self.R))
        dreamer = st_save(split_sets(bank)[1]["dreamer"])
        x = torch.randn(3, D_IN)
        for path in PATHS:
            torch.testing.assert_close(
                delta_of(self.adapter_payload(levers), path, x),
                delta_of(payload, path, x) + delta_of(dreamer, path, x), rtol=1e-5, atol=1e-5)

    def test_every_lora_id_on_the_build_is_distinct(self):
        self.library.add("a", named_payload(self.R, 5))
        attached = self.lowering.attach("bundle:v1", {"pi": self.bank_payload()})
        ids = [request.lora_int_id for request in attached.sets.values()]
        ids += [self.levers(attached, route).kwargs["lora_request"].lora_int_id
                for route in ("lib:a", "lib:a+dreamer", "lib:a+memory:00")]
        self.assertEqual(len(set(ids)), 5)

    def test_a_stack_dies_with_its_library_entry_and_with_its_bundle(self):
        for name in ("a", "b"):
            self.library.add(name, named_payload(self.R, seed=len(name) + ord(name)))
        attached = self.lowering.attach("bundle:v1", {"pi": self.bank_payload()})
        alone = Path(self.levers(attached, "lib:a").kwargs["lora_request"].lora_path)
        stack = Path(self.levers(attached, "lib:a+dreamer").kwargs["lora_request"].lora_path)
        kept = Path(self.levers(attached, "lib:b+dreamer").kwargs["lora_request"].lora_path)
        self.library.get("b")
        self.library.add("c", named_payload(self.R, 9))                # evicts a
        self.assertFalse(alone.exists() or stack.exists())
        self.assertTrue(kept.exists())
        with self.assertRaisesRegex(RuntimeError, "'a'.*add_library"):
            self.levers(attached, "lib:a+dreamer")
        self.lowering.detach(attached)
        self.assertFalse(kept.exists())
        self.assertFalse(Path(attached.sets["dreamer"].lora_path).exists())

    def test_a_stack_wider_than_the_build_is_refused_with_both_numbers(self):
        self.library.add("wide", named_payload(self.R + 1, 5))
        attached = self.lowering.attach("bundle:v1", {"pi": self.bank_payload()})
        self.levers(attached, "lib:wide")                              # alone it fits
        with self.assertRaisesRegex(ValueError, rf"rank {2 * self.R + 1}.*max_rank {2 * self.R}"):
            self.levers(attached, "lib:wide+dreamer")
        narrow = self.module.DreamBankRollout(ServingBuild(
            base="Qwen/x", config=None, workdir=self.build.workdir / "narrow",
            max_bundles=2, max_rank=self.R - 1))
        with self.assertRaisesRegex(ValueError, rf"rank {self.R}.*max_rank {self.R - 1}"):
            narrow.attach("bundle:v1", {"pi": self.bank_payload()})

    def test_routes_outside_the_grammar_are_refused(self):
        self.library.add("a", named_payload(self.R, 5))
        attached = self.lowering.attach("bundle:v1", {"pi": self.bank_payload()})
        for route in ("lib:a+base", "lib:a+dreamer+memory:00", "dreamer+dreamer",
                      "lib:", "lib:a+memory:07"):
            with self.assertRaises(ValueError, msg=route):
                self.levers(attached, route)


if __name__ == "__main__":
    unittest.main()
