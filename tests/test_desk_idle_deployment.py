"""The deployed desk restores finite GPU idle limits without replaying work."""
import importlib.util
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from rlstack import LocalStore
from rlstack.runner.desk import Desk, DeskError, Metal
from venue_stub import modal_stubbed


class FiniteIdleDeployment(unittest.TestCase):
    def test_replayed_pins_become_finite_without_registration_or_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalStore(directory)
            previous = Desk(store, host_for=lambda address: None)
            for name, limit in [('pinned', None), ('five-minutes', 300.0), ('short', 90.0)]:
                previous.register_metal(Metal(name, 'NVIDIA L4', 1, 22.03), idle_s=limit)
            store.append_fleet_event({'event': 'parked', 'run_id': 'historical-paused',
                                      'avoiding': '', 'reason': 'review pause'})
            original_journal = store.read_fleet_log()
            with modal_stubbed():
                path = Path(__file__).resolve().parents[1] / 'deploy/desk.py'
                spec = importlib.util.spec_from_file_location('finite_idle_deployment', path)
                module = importlib.util.module_from_spec(spec)
                with patch.object(sys, "path", [str(path.parent), *sys.path]):
                    spec.loader.exec_module(module)
                instance = module.Desk()
                with patch.object(module, 'a_store', return_value=store):
                    instance.bring_up()
            self.assertEqual(instance.desk.idle_limit('pinned'), 300.0)
            self.assertEqual(instance.desk.idle_limit('five-minutes'), 300.0)
            self.assertEqual(instance.desk.idle_limit('short'), 90.0)
            self.assertEqual(instance.desk.parked(), {'historical-paused': ''})
            self.assertEqual(store.read_fleet_log(), original_journal)
            for limit in (None, float("inf"), float("nan")):
                with self.assertRaisesRegex(DeskError, "finite automatic"):
                    instance.desk.register_metal(Metal("new", "L4", 1, 24), idle_s=limit)
            self.assertNotIn("new", instance.desk.metal)
            self.assertEqual(store.read_fleet_log(), original_journal)


if __name__ == '__main__':
    unittest.main()
