"""Explicit source closure and worker installation, without a recipe dependency."""

import asyncio
import importlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

from common import arith_spec
from rlstack.code import load_code, require_declared_function, source_manifest, verify_code
from rlstack.policy.compile import Bundle, compile_bundle, compile_tensor_pin
from rlstack.policy.state import StateLayout
from rlstack.registry import code_hashes
from rlstack.runner.fakes import FakeEngine, FakeLearner
from rlstack.runner.host import Host, HostError
from rlstack.runner.interfaces import OptimSettings, Parameterization
from rlstack.runner.remote import decode_bundle, decode_parameterization, encode_bundle, encode_parameterization
from rlstack.runner.venues.strangeloop.provider import source_bundle
from rlstack.spec.canonical import canonical_json


class ExternalCodeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.package = self.root / "owned_experiment"
        self.package.mkdir()
        (self.package / "__init__.py").write_text("")
        (self.package / "helper.py").write_text('PROMPT = "first"\n')
        (self.package / "interaction.py").write_text(
            'from .helper import PROMPT\nasync def episode(client, task):\n    return PROMPT\n')
        (self.package / "constructor.py").write_text(
            'def learner(fsdp, sleeps, dp):\n    from rlstack.runner.fakes import FakeLearner\n    learner = FakeLearner(fsdp=fsdp, sleeps=sleeps, dp=dp)\n    learner.from_experiment = True\n    return learner\n')
        sys.path.insert(0, self.tmp.name)
        importlib.invalidate_caches()

    def tearDown(self):
        sys.path.remove(self.tmp.name)
        for name in list(sys.modules):
            if name == "owned_experiment" or name.startswith("owned_experiment."):
                del sys.modules[name]
        self.tmp.cleanup()
        importlib.invalidate_caches()

    def test_helper_and_constant_changes_enter_the_full_source_identity(self):
        first = load_code(("owned_experiment",))
        (self.package / "helper.py").write_text('PROMPT = "second"\n')
        second = source_manifest(("owned_experiment",))
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(ValueError, "changed after import"):
            load_code(("owned_experiment",))
        del sys.modules["owned_experiment"]
        load_code(("owned_experiment",))
        spec = arith_spec("cas://train")
        with_code = replace(spec, code=(*spec.code, "owned_experiment"))
        self.assertEqual(code_hashes(with_code)["module:owned_experiment"], second["owned_experiment"])

    def test_subpackage_identity_includes_executed_parent_initializer(self):
        nested = self.package / "nested"
        nested.mkdir()
        (nested / "__init__.py").write_text("")
        (nested / "operation.py").write_text("value = 1\n")
        first = source_manifest(("owned_experiment.nested",))
        (self.package / "__init__.py").write_text("parent_constant = 7\n")
        self.assertNotEqual(first, source_manifest(("owned_experiment.nested",)))

    def test_imported_function_still_requires_an_explicit_root(self):
        importlib.import_module("owned_experiment.interaction")
        with self.assertRaisesRegex(ValueError, "not covered"):
            require_declared_function("owned_experiment.interaction:episode", ())
        self.assertTrue(asyncio.iscoroutinefunction(require_declared_function(
            "owned_experiment.interaction:episode", ("owned_experiment",))))

    def test_missing_or_changed_worker_code_is_refused_before_install(self):
        code = load_code(("owned_experiment",))
        verify_code(code)
        changed = {"owned_experiment": "0" * 64}
        learner = FakeLearner()
        parameterization = Parameterization("base", "nll", (), OptimSettings("adamw", 1e-4, (0.9, 0.95), 0.0, {}), changed)
        with self.assertRaisesRegex(ValueError, "code mismatch"):
            learner.install("tenant", parameterization)
        self.assertEqual(learner._tenants, {})
        engine = FakeEngine()
        with self.assertRaisesRegex(ValueError, "code mismatch"):
            engine.add_bundle(Bundle("untrusted", {}, code=changed))
        self.assertFalse(engine.knows_bundle("untrusted"))
        with self.assertRaises((ValueError, ModuleNotFoundError)):
            verify_code({"missing_experiment_package": "0" * 64})

    def test_bundle_identity_and_wires_include_the_code_manifest(self):
        code = load_code(("owned_experiment",))
        plain = compile_bundle({"a": b"weights"}, {"a": 1}, ("a",), {"a": "lora"})
        owned = compile_bundle({"a": b"weights"}, {"a": 1}, ("a",), {"a": "lora"}, code)
        self.assertNotEqual(plain.bundle_id, owned.bundle_id)
        self.assertEqual(decode_bundle(encode_bundle(owned)), owned)
        parameterization = Parameterization("base", "nll", (), OptimSettings("adamw", 1e-4, (0.9, 0.95), 0.0, {}), code)
        self.assertEqual(decode_parameterization(encode_parameterization(parameterization)), parameterization)
        layout = StateLayout("", None, ())
        self.assertNotEqual(compile_tensor_pin({"a": layout}, {"a": 1}, {"a": "lora"}).bundle_id,
                            compile_tensor_pin({"a": layout}, {"a": 1}, {"a": "lora"}, code).bundle_id)

    def test_empty_code_preserves_the_prior_canonical_form(self):
        spec = replace(arith_spec("cas://train"), gen=None, code=())
        self.assertNotIn('"code"', canonical_json(spec))

    def test_host_refuses_missing_code_claims_and_partial_registered_claims(self):
        spec = arith_spec("cas://train")
        claimed = code_hashes(spec)
        host = object.__new__(Host)
        host.check_code_agreement(spec, claimed)
        incomplete = dict(claimed)
        incomplete.pop("module:boundary_fixtures")
        with self.assertRaisesRegex(HostError, "code skew"):
            host.check_code_agreement(spec, incomplete)
        with self.assertRaisesRegex(HostError, "code skew"):
            host.check_code_agreement(spec, None)

    def test_named_state_discovery_uses_adapter_rules_and_explicit_plan_dependencies(self):
        from rlstack.data.plan import GroupPlan, Sample, WavePlan
        from rlstack.data.trajectory import Task
        from rlstack.policy.adapters.base import AdapterType
        from rlstack.runner.names import lib_names, row_lib_names, wave_lib_names

        class OwnedGrammar(AdapterType):
            def named_dependencies(self, routes):
                return tuple(route.removeprefix("frozen/") for route in routes if route.startswith("frozen/"))

        adapter = OwnedGrammar()
        self.assertEqual(lib_names(("frozen/parent",), (adapter,)), ("parent",))
        plan = WavePlan((GroupPlan("g", (Sample("t", "example:episode",
            role="frozen/parent", dependencies=("explicit",)),)),))
        self.assertEqual(wave_lib_names(plan, (adapter,)), ("explicit", "parent"))
        rows = [{"turns": [{"turn_extras": {"route": "frozen/parent"}}]}]
        self.assertEqual(row_lib_names(rows, (adapter,)), ("parent",))
        self.assertEqual(lib_names(("lib:old-recipe+dreamer",)), ())

    def test_external_fake_constructor_is_verified_and_uses_the_normal_build_door(self):
        from rlstack.runner.host import Partition, Regime
        from rlstack.runner.residents import FakeLearnerBuild, build_learner, decode_build, encode_build
        code = load_code(("owned_experiment",))
        build = FakeLearnerBuild(sleeps=True, factory="owned_experiment.constructor:learner", code=code)
        self.assertEqual(decode_build(json.loads(json.dumps(encode_build(build)))), build)
        learner = build_learner(Regime("learner", "training", "base", 2),
                                Partition("metal", (0, 1), 1.0, "fake"), build)
        self.assertTrue(learner.from_experiment)
        self.assertEqual(learner.fsdp, 2)
        self.assertTrue(learner.sleeps)
        with self.assertRaisesRegex(ValueError, "code mismatch"):
            build_learner(Regime("learner", "training", "base", 1),
                          Partition("metal", (0,), 1.0, "fake"),
                          replace(build, code={"owned_experiment": "0" * 64}))

    def test_source_archive_ships_only_explicit_external_packages(self):
        (self.root / "rlstack").mkdir()
        (self.root / "rlstack" / "base.py").write_text("value = 1\n")
        with zipfile.ZipFile(io.BytesIO(source_bundle(self.root))) as archive:
            self.assertEqual(archive.namelist(), ["rlstack/base.py"])
        with zipfile.ZipFile(io.BytesIO(source_bundle(self.root, ("owned_experiment",)))) as archive:
            self.assertIn("owned_experiment/helper.py", archive.namelist())
            self.assertEqual(archive.read("owned_experiment/helper.py"), b'PROMPT = "first"\n')
