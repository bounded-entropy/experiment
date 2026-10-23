"""Direct local clients serialize shared-model work, beyond residency admission."""
from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from unittest.mock import patch

from rlstack.data.stores.local import LocalStore
from test_supervision import row
from rlstack.policy.siteschema import SiteSchema
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.refs import RefReader
from rlstack.runner.training_client import TrainingClient
from rlstack.spec.specs import AdapterSpec
from rlstack.training.supervision import SftStage, supervised_spec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rlstack.runner.learners.torch_learner import TorchLearner
    from test_routed_bank import SITE, TinyModel


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 7 for character in text]


@unittest.skipIf(torch is None, "CPU torch required for real shared-model custody")
class LocalLearnerCustodyTest(unittest.IsolatedAsyncioTestCase):
    async def test_fsdp_announcements_cannot_overtake_an_active_local_operation(self):
        """A CPU chorus stub proves ordering, without claiming GPU collectives."""
        from rlstack.runner.learners.fsdp_torch import FsdpTorchLearner
        from test_learner_sleep import Heard
        from test_routed_bank import installation

        learner = FsdpTorchLearner(Heard())
        entered, attempted, release = threading.Event(), threading.Event(), threading.Event()
        def blocked_install(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test failed to release chorus installation")
        original_uninstall = learner.uninstall
        def record_uninstall(*args):
            attempted.set()
            return original_uninstall(*args)
        with (patch.object(TorchLearner, "install", blocked_install),
              patch.object(TorchLearner, "uninstall", return_value=None),
              patch.object(learner, "uninstall", record_uninstall)):
            installing = asyncio.create_task(asyncio.to_thread(learner.install, "first", installation()))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                removing = asyncio.create_task(asyncio.to_thread(learner.uninstall, "second"))
                self.assertTrue(await asyncio.to_thread(attempted.wait, 3))
                self.assertEqual(learner.ranks.said, ["install"])
            finally:
                release.set()
            await asyncio.gather(installing, removing)
        self.assertEqual(learner.ranks.said, ["install", "uninstall"])

    async def test_concurrent_parent_clients_cannot_install_into_an_active_child_forward(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        store = LocalStore(root.name)
        with torch.random.fork_rng():
            torch.manual_seed(17)
            learner = TorchLearner("cpu", torch.float32, checkpoint_activations=False)
            learner._model = TinyModel().requires_grad_(False).eval()
        learner._base, learner._tokenizer = "route-toy", Tokenizer()
        schema, arbiter = SiteSchema("route-toy", (SITE,)), Arbiter()
        arbiter.attach(learner, label="training")
        study = row(0)
        spec = supervised_spec("route-toy", AdapterSpec("lora", "proj", {"r": 2}),
            [[study]], SftStage(2, 1, 1e-4), 17, store.cas_put)
        clients = [TrainingClient(learner, arbiter, tenant=name,
            refs=RefReader(store, store.open_run(name, {"run_id": name}, subdir="parents")),
            schema=schema, seed=17, microbatch_tokens=128) for name in ("first", "second")]
        entered, attempted_install, release = threading.Event(), threading.Event(), threading.Event()
        original_forward, original_install = learner._batched_logprobs, learner.install
        held = False
        def blocked_forward(*args):
            nonlocal held
            if not held:
                held = True
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test failed to release local learner")
            return original_forward(*args)
        def record_install(*args):
            if entered.is_set():
                attempted_install.set()
            return original_install(*args)
        tasks = []
        with (patch.object(learner, "_batched_logprobs", blocked_forward),
              patch.object(learner, "install", record_install)):
            try:
                tasks.append(asyncio.create_task(clients[0].run(spec, checkpoint_every=2, save_state=False)))
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                first_tenant, = learner._tenants
                tasks.append(asyncio.create_task(clients[1].run(spec, checkpoint_every=2, save_state=False)))
                self.assertTrue(await asyncio.to_thread(attempted_install.wait, 3))
                # Both clients can hold physical admission, while the second
                # install must wait before changing the shared wrapper roster.
                self.assertGreaterEqual(arbiter.in_flight(), 2)
                self.assertEqual(set(learner._tenants), {first_tenant})
                self.assertFalse(tasks[1].done())
            finally:
                release.set()
            results = await asyncio.gather(*tasks)
        self.assertEqual([result.completed for result in results], [2, 2])
        self.assertNotEqual(results[0].run_id, results[1].run_id)
        self.assertEqual(learner._tenants, {})
        self.assertIsInstance(learner._model.proj, torch.nn.Linear)
        self.assertEqual(arbiter.in_flight(), 0)
        for client in clients:
            await client.close()


if __name__ == "__main__":
    unittest.main()
