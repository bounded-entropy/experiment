"""ModalVolumeStore: the commit discipline over a recorded volume.

The property under test: staged writes persist EXACTLY at the protocol's
durable points — run creation, the ledger line (the commit point, sealing
everything the update staged), and eval output — never per blob or per wave.
"""

from __future__ import annotations

import tempfile
import unittest

from rlstack import ModalVolumeStore


class RecordingVolume:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class ModalVolumeStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.volume = RecordingVolume()
        self.store = ModalVolumeStore(tmp.name, volume=self.volume)

    def test_staged_work_persists_only_at_the_durable_points(self) -> None:
        run = self.store.open_run("r1", manifest={"run_id": "r1"})
        created = self.volume.commits          # manifest + ledger creation
        self.assertGreaterEqual(created, 1)

        # an update's staged work: no commits until the ledger line
        run.write_wave(1, [{"task": "t0"}])
        run.write_postdata(1, {"reward": [1.0]})
        run.write_blob("adapters", "pi", 1, b"delta")
        run.write_blob("optim", "pi", 1, b"moments")
        self.assertEqual(self.volume.commits, created)

        run.append_ledger({"update": 1, "versions": {"pi": 1}})
        self.assertEqual(self.volume.commits, created + 1)   # THE commit point

        run.write_eval(1, "summary.json", "{}")              # firewalled output
        self.assertEqual(self.volume.commits, created + 2)

    def test_without_a_volume_it_is_just_a_local_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ModalVolumeStore(tmp)                    # volume=None
            run = store.open_run("r1", manifest={"run_id": "r1"})
            run.append_ledger({"update": 1, "versions": {}})
            self.assertEqual(run.ledger_tail()["update"], 1)


if __name__ == "__main__":
    unittest.main()
