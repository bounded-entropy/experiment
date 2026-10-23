"""Publication is independent of cadence, adapter names and Python codecs."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch, Mock

from common import arith_spec, arith_store, generation_spec, cas_uri
from rlstack import Plans, RunPlan, WavePlan, GroupPlan, Replay, Topology, HostSpec, learner, encode
from rlstack.runner.weight_transfer import WeightTransfer
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.campaign import frame_for
from rlstack.runner.fakes import FakeEngine, FakeLearner
from rlstack.runner.loop import run_experiment
from rlstack.policy.siteschema import fake_qwen_schema
from rlstack.policy.state import StateLayout, TensorLayout, tensor_bundle_id
from rlstack.runner.publication import publish_tensors
from rlstack.runner.pool import Pool, Replicas
import asyncio


class Exports(FakeLearner):
    def __init__(self):
        super().__init__()
        self.adapter_exports = self.optimizer_exports = 0

    def emit_adapters(self, tenant, versions=None):
        self.adapter_exports += 1
        return super().emit_adapters(tenant, versions)

    def emit_optim(self, tenant):
        self.optimizer_exports += 1
        return super().emit_optim(tenant)


class PublicationPolicy(unittest.TestCase):
    def replay_student(self, store, train):
        schema = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
        teacher = run_experiment(generation_spec(train), schema, store,
            FakeEngine(), None, checkpointing=Checkpointing(1))
        rows = store.peek_rollout(teacher.run_id, 1)
        plan = RunPlan(tuple(WavePlan((GroupPlan("sealed", tuple(
            Replay(f"store://{teacher.run_id}/rollouts/1#{i}") for i in range(len(rows)))),))
            for _ in range(4)))
        store.cas_put(encode(plan))
        spec = arith_spec(train, gen=None, plans=Plans(train=cas_uri(encode(plan))),
                          topology=Topology((HostSpec((learner(),)),)))
        return replace(spec, algo=replace(spec.algo, loss="sft", post=()))

    def test_choice_is_on_submission_not_scientific_spec(self):
        with tempfile.TemporaryDirectory() as root:
            _, train, _ = arith_store(root)
            spec = arith_spec(train)
            base = frame_for(spec, checkpointing=Checkpointing(3))
            nccl = frame_for(spec, checkpointing=Checkpointing(3),
                             weight_transfer=WeightTransfer("nccl"))
            self.assertEqual(base["spec"], nccl["spec"])
            self.assertEqual(base["code"], nccl["code"])
            self.assertEqual(nccl["weight_transfer"]["backend"], "nccl")
            self.assertNotIn("adapter_type", nccl["weight_transfer"])

    def test_invalid_choices_are_refused(self):
        for kwargs in ({"backend": "lora_nccl"}, {"timeout_s": 0},
                       {"timeout_s": float("inf")}, {"rendezvous_host": ""}):
            with self.assertRaises(ValueError):
                WeightTransfer(**kwargs)

    def test_nccl_is_refused_on_fake_metal_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as root:
            store, train, _ = arith_store(root)
            spec = arith_spec(train)
            with self.assertRaisesRegex(ValueError, "no CUDA tensors"):
                run_experiment(spec, fake_qwen_schema(4, base=spec.policy.base),
                    store, FakeEngine(), FakeLearner(), checkpointing=Checkpointing(3),
                    weight_transfer=WeightTransfer("nccl"))

    def test_optimizer_exports_follow_checkpoint_frequency(self):
        with tempfile.TemporaryDirectory() as root:
            store, train, _ = arith_store(root)
            learner = Exports()
            run_experiment(arith_spec(train), fake_qwen_schema(4, base="Qwen/Qwen3-0.6B"), store,
                           FakeEngine(), learner, checkpointing=Checkpointing(3))
            # Initial state, update 3, final update 4; serving weights still
            # publish on all four updates, without four Adam serializations.
            self.assertEqual(learner.optimizer_exports, 3)
            self.assertEqual(learner.adapter_exports, 5)

    def test_sealed_sft_does_not_export_intermediate_weights(self):
        with tempfile.TemporaryDirectory() as root:
            store, train, _ = arith_store(root)
            spec = self.replay_student(store, train)
            metal = Exports()
            run_experiment(spec, fake_qwen_schema(4, base=spec.policy.base), store,
                           {}, metal, checkpointing=Checkpointing(7))
            self.assertEqual(metal.adapter_exports, 2)     # initial and final only
            self.assertEqual(metal.optimizer_exports, 2)

    def test_sealed_sft_ledger_is_independent_of_checkpoint_cadence(self):
        ledgers = []
        for every in (1, 3, 7):
            with tempfile.TemporaryDirectory() as root:
                store, train, _ = arith_store(root)
                spec = self.replay_student(store, train)
                report = run_experiment(spec, fake_qwen_schema(4, base=spec.policy.base),
                    store, {}, FakeLearner(), checkpointing=Checkpointing(every))
                ledgers.append(store.peek_ledger(report.run_id))
        self.assertEqual(ledgers[0], ledgers[1])
        self.assertEqual(ledgers[0], ledgers[2])


class PublicationAcknowledgement(unittest.TestCase):
    def setup_publication(self):
        layout = StateLayout("", None, (TensorLayout("w", "float32", (4,), "digest"),))
        versions = {"external": 1}
        manifest = {"token": "snapshot", "source": "resident-epoch", "versions": versions,
                    "states": {"external": layout.row()}, "adapter_types": {"external": "custom"},
                    "bundle_id": tensor_bundle_id(versions, {"external": layout}), "hash_s": 0.}
        source = Mock(spec=["prepare_transfer", "send_transfer", "release_transfer"])
        source.prepare_transfer.return_value = manifest
        source.send_transfer.return_value = {"bytes": 16, "transfer_s": .01}
        engines = []
        for index in range(2):
            engine = Mock(spec=["prepare_transfer", "finish_transfer", "abort_transfer"])
            engine.prepare_transfer.return_value = {"host": "127.0.0.1", "port": 1,
                                                     "channel": str(index)}
            engine.finish_transfer.return_value = {"bundle_id": manifest["bundle_id"]}
            engines.append(engine)
        pools = {"main": Pool.of("main", Replicas(tuple(engines)))}
        return source, engines, pools, manifest

    def publish(self, source, pools, manifest):
        return asyncio.run(publish_tensors(source, "tenant", manifest["versions"],
            ["external"], manifest["adapter_types"], pools, WeightTransfer("nccl"), None, 1))

    def test_every_replica_acknowledges_one_captured_version(self):
        source, engines, pools, manifest = self.setup_publication()
        pin = self.publish(source, pools, manifest)
        self.assertEqual(pin.bundle_id, manifest["bundle_id"])
        source.prepare_transfer.assert_called_once()
        self.assertEqual(source.send_transfer.call_count, 2)
        source.release_transfer.assert_called_once_with("snapshot")
        for engine in engines:
            engine.finish_transfer.assert_called_once_with("resident-epoch", "snapshot")
            engine.abort_transfer.assert_not_called()

    def test_wrong_ack_cleans_up_without_returning_a_publishable_pin(self):
        source, engines, pools, manifest = self.setup_publication()
        engines[1].finish_transfer.return_value = {"bundle_id": "wrong-version"}
        with self.assertRaisesRegex(ValueError, "different policy version"):
            self.publish(source, pools, manifest)
        source.release_transfer.assert_called_once_with("snapshot")
        for engine in engines:
            engine.abort_transfer.assert_called_once_with("resident-epoch", "snapshot")

    def test_uncertain_send_is_never_replayed(self):
        source, engines, pools, manifest = self.setup_publication()
        source.send_transfer.side_effect = TimeoutError("unknown completion")
        with self.assertRaises(TimeoutError):
            self.publish(source, pools, manifest)
        source.send_transfer.assert_called_once()
        engines[0].abort_transfer.assert_called_once()
        engines[1].prepare_transfer.assert_not_called()
        source.release_transfer.assert_called_once_with("snapshot")


TORCH = importlib.util.find_spec("torch") is not None
SAFE = importlib.util.find_spec("safetensors") is not None


@unittest.skipUnless(TORCH and SAFE, "requires torch and safetensors")
class TensorProtocol(unittest.TestCase):
    def setUp(self):
        import torch
        import torch.distributed as dist
        if not dist.is_gloo_available():
            self.skipTest("CPU distributed Gloo is unavailable")
        from rlstack.runner.transports.tensor import TensorSender, TensorReceiver
        self.torch = torch
        self.received = []
        self.sender = TensorSender("cpu", backend="gloo")
        self.receiver = TensorReceiver(lambda manifest, states: self.received.append(
            (manifest["bundle_id"], states)), device="cpu", backend="gloo")
        self.addCleanup(self.receiver.close)
        self.addCleanup(self.sender.close)

    def states(self):
        from rlstack.policy.adapters.state_torch import TensorState
        torch = self.torch
        header = b'{"recipe":"external","version":7}'
        return {
            "external_one": TensorState({"matrix": torch.arange(12.).reshape(3, 4),
                                         "scalar": torch.tensor(2., dtype=torch.bfloat16)}),
            "external_two": TensorState({"mask": torch.tensor([True, False]),
                                         "empty": torch.empty(0, 2)},
                                         len(header).to_bytes(8, "big") + header,
                                         {"alpha": "0.5"})}

    def transfer(self, states):
        policy = WeightTransfer("nccl", timeout_s=10)
        manifest = self.sender.capture(states, {name: 7 for name in states},
                                       {name: name for name in states})
        endpoint = self.receiver.prepare(manifest, policy)
        sent = self.sender.send(manifest["token"], endpoint, 10)
        ack = self.receiver.finish(manifest["source"], manifest["token"])
        self.sender.release(manifest["token"])
        return manifest, endpoint, sent, ack

    def test_raw_tensors_cross_without_any_archive_or_base64(self):
        import torch.distributed as dist
        from rlstack.policy.adapters.state_torch import TensorState
        before = dist.is_initialized()
        states = self.states()
        with patch("base64.b64encode", side_effect=AssertionError("base64 used")), \
             patch("torch.save", side_effect=AssertionError("checkpoint serialized")), \
             patch.object(TensorState, "encode", side_effect=AssertionError("archive encoded")):
            manifest, _, sent, ack = self.transfer(states)
        self.assertEqual(before, dist.is_initialized(), "default learner process group changed")
        self.assertEqual(manifest["bundle_id"], ack["bundle_id"])
        self.assertEqual(sent["bytes"], ack["bytes"])
        received = self.received[-1][1]
        for name, state in states.items():
            self.assertEqual(state.layout(), received[name].layout())
            self.assertEqual(state.encode(), received[name].encode())

    def test_builtin_state_shapes_share_one_transport(self):
        from test_batched_replay import a_state as lora_state, SITES
        from test_plora import a_state as latent_state
        from test_steer import a_state as steer_state
        from test_spectral import a_trained_state as spectral_state
        from test_spectral_latent import a_trained_state as slatent_state
        from test_soft_prompt import BOUNDARY, WIDTH
        from rlstack.policy.adapters import dream_bank_torch, soft_prompt_torch
        from rlstack.policy.state import payload_layout
        parameters = {
            "lora": lora_state(1), "latent": latent_state(), "steer": steer_state(1),
            "spectral": spectral_state(), "slatent": slatent_state(),
            "bank": dream_bank_torch.build(SITES, {"r": 3, "memories": 2, "seed": 1}),
            "prompt": soft_prompt_torch.build(BOUNDARY, {"n": 3, "d": WIDTH, "seed": 1})}
        states = {name: params.export_state(7) for name, params in parameters.items()}
        self.transfer(states)
        received = self.received[-1][1]
        for name, state in states.items():
            with self.subTest(adapter=name):
                self.assertEqual(state.layout(), received[name].layout())
                self.assertEqual(state.layout(), payload_layout(received[name].encode()))

    def test_checkpoint_reconstructs_the_same_tensor_bundle_id(self):
        from rlstack.policy.state import payload_layout, tensor_bundle_id
        from rlstack.policy.compile import restore_bundle
        states = self.states()
        versions = {name: 7 for name in states}
        layouts = {name: state.layout() for name, state in states.items()}
        for name, state in states.items():
            self.assertEqual(layouts[name], payload_layout(state.encode()))
        bundle_id = tensor_bundle_id(versions, layouts)
        restored = restore_bundle(versions, bundle_id,
            lambda section, name, version: states[name].encode(), states,
            {name: name for name in states})
        self.assertEqual(bundle_id, restored.bundle_id)

    def test_peer_group_is_reused_and_ack_is_idempotent(self):
        one, endpoint, _, ack = self.transfer(self.states())
        self.assertEqual(ack, self.receiver.finish(one["source"], one["token"]))
        two, second, _, _ = self.transfer(self.states())
        self.assertEqual(endpoint, second)
        self.assertNotEqual(one["token"], two["token"])
        self.assertEqual(len(self.sender.groups), 1)

    def test_snapshot_survives_mutation_of_training_tensors(self):
        from rlstack.policy.adapters.state_torch import TensorState
        state = TensorState({"weight": self.torch.tensor([1., 2.])})
        manifest = self.sender.capture({"adapter": state}, {"adapter": 1}, {"adapter": "external"})
        state.tensors["weight"].add_(100)
        endpoint = self.receiver.prepare(manifest, WeightTransfer("nccl", timeout_s=10))
        self.sender.send(manifest["token"], endpoint, 10)
        self.receiver.finish(manifest["source"], manifest["token"])
        self.torch.testing.assert_close(self.received[-1][1]["adapter"].tensors["weight"],
                                        self.torch.tensor([1., 2.]))
        self.sender.release(manifest["token"])

    def test_explicit_ensemble_version_does_not_follow_checkpoint_count(self):
        from test_plora import a_state
        state = a_state(seed=19)
        first = state.export_state(17)
        state.export_state(2)              # taking another checkpoint cannot advance a draw
        again = state.export_state(17)
        self.assertEqual(state.version, 0)
        self.assertEqual(first.encode(), again.encode())
        self.assertNotEqual(first.encode(), state.export_state(18).encode())

    def test_warm_start_and_resume_preserve_the_ensemble_counter_offset(self):
        from test_plora import a_state
        from rlstack.policy.adapters import plora_torch
        state = a_state(seed=19)
        state.version = 40               # inherited checkpoint, new run starts at logical zero
        state.bind_version(0)
        initial = plora_torch.emit(state)
        self.assertEqual(initial, state.export_state(0).encode())
        checkpoint = state.export_state(4).encode()
        restored = a_state(seed=19)
        plora_torch.load(restored, checkpoint)
        restored.bind_version(4)
        self.assertEqual(checkpoint, plora_torch.emit(restored))
        self.assertEqual(state.export_state(5).encode(), restored.export_state(5).encode())

    def test_corrupt_transfer_is_never_installed(self):
        states = self.states()
        manifest = self.sender.capture(states, {n: 7 for n in states}, {n: n for n in states})
        # Corrupt the retained sending buffer after its manifest was signed.
        self.sender.snapshots[manifest["token"]]["external_one"].tensors["matrix"].add_(1)
        endpoint = self.receiver.prepare(manifest, WeightTransfer("nccl", timeout_s=10))
        self.sender.send(manifest["token"], endpoint, 10)
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            self.receiver.finish(manifest["source"], manifest["token"])
        self.assertEqual(self.received, [])
        self.sender.release(manifest["token"])


@unittest.skipUnless(TORCH and SAFE, "requires torch and safetensors")
class LiveNcclTransport(unittest.TestCase):
    def test_two_cuda_devices_transfer_tensor_state(self):
        """Hardware smoke test, not model-serving qualification or a benchmark."""
        import torch
        import torch.distributed as dist
        if not dist.is_nccl_available() or torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA devices with working NCCL")
        from rlstack.policy.adapters.state_torch import TensorState
        from rlstack.runner.transports.tensor import TensorSender, TensorReceiver
        received = []
        sender = TensorSender("cuda:0")
        receiver = TensorReceiver(lambda _, states: received.append(states), "cuda:1")
        self.addCleanup(receiver.close)
        self.addCleanup(sender.close)
        tensor = torch.arange(2048, device="cuda:0", dtype=torch.bfloat16)
        manifest = sender.capture({"external": TensorState({"weights": tensor})},
                                   {"external": 1}, {"external": "external"})
        endpoint = receiver.prepare(manifest, WeightTransfer("nccl"))
        sender.send(manifest["token"], endpoint, 60)
        receiver.finish(manifest["source"], manifest["token"])
        torch.testing.assert_close(received[0]["external"].tensors["weights"].cpu(), tensor.cpu())
        sender.release(manifest["token"])
