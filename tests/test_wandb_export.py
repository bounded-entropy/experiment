"""Offline export derives real committed metrics and never repairs the source."""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlstack.data.stores import LedgerError, LocalStore, StoreError
from rlstack.runner.exporters.wandb import (
    LedgerWandbExporter, OfflineWandbRun, await_manifest, committed_entries, ledger_metrics, main,
)
from test_strangeloop_store import MemoryScratch


class RecordingWandbRun(OfflineWandbRun):
    def __init__(self, **config):
        self.config = config
        self.path = str(config["directory"] / ("offline-" + config["attempt_id"]))
        self.rows = []
        self.finished = False
        self.error = None

    def log(self, metrics, *, step):
        self.rows.append((step, dict(metrics)))
        if self.error:
            raise self.error

    def finish(self):
        self.finished = True


class WandbExportTests(unittest.TestCase):
    def test_export_waits_for_exact_manifest_without_creating_a_run(self):
        with patch.object(self.store, "peek_manifest", side_effect=[None, {"run_id": "r"}]) as read, \
                patch("rlstack.runner.exporters.wandb.time.sleep") as sleep:
            await_manifest(self.store, "family/r")
        self.assertEqual([call.args for call in read.call_args_list], [("family/r",), ("family/r",)])
        sleep.assert_called_once()
        with self.assertRaisesRegex(StoreError, "publication wait"):
            await_manifest(self.store, "family/missing", timeout_s=0)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = LocalStore(self.root / "store")
        self.run = self.store.open_run("r", {"run_id": "r"}, subdir="family")
        mocked = patch("rlstack.runner.exporters.wandb.OfflineWandbRun", RecordingWandbRun)
        mocked.start()
        self.addCleanup(mocked.stop)

    def exporter(self, **overrides):
        args = dict(objective="train/loss", directory=self.root / "artifacts", lease_id="lease")
        args.update(overrides)
        return LedgerWandbExporter(self.store, "family/r", **args)

    def commit(self, update, **fields):
        self.run.append_ledger({"update": update, "versions": {}, **fields})

    def test_repeated_polls_log_real_points_once_without_touching_source(self):
        self.commit(0, train={"loss": 2.5}, post={"reward": 0.25})
        self.commit(1, train={"loss": 1.5}, wave={"groups": 2})
        before = {key: self.store._read(key) for key in self.store._list("runs/")}
        exporter = self.exporter()
        result = exporter.poll()
        self.assertEqual((result.logged, result.last_update, result.objective_points), (2, 1, 2))
        self.assertEqual(exporter._wandb.rows, [
            (0, {"update": 0, "train/loss": 2.5, "post/reward": 0.25, "objective": 2.5}),
            (1, {"update": 1, "train/loss": 1.5, "wave/groups": 2, "objective": 1.5}),
        ])
        self.assertEqual(exporter.poll().logged, 0)
        self.assertEqual(before, {key: self.store._read(key) for key in self.store._list("runs/")})
        self.commit(2, train={"loss": 1.0})
        self.assertEqual(exporter.poll().logged, 1)
        self.assertEqual([step for step, _ in exporter._wandb.rows], [0, 1, 2])

    def test_missing_objective_is_reported_and_never_filled_with_zero(self):
        self.commit(1, post={"reward": 3.0})
        self.commit(2, train={"loss": math.nan, "tokens": 123})
        exporter = self.exporter()
        progress = exporter.poll()
        self.assertEqual(progress.missing_objective_updates, (1, 2))
        self.assertEqual(progress.objective_points, 0)
        self.assertTrue(all("objective" not in row for _, row in exporter._wandb.rows))
        self.assertNotIn("train/loss", exporter._wandb.rows[1][1])

    def test_generation_exports_only_sealed_waves_once_without_a_ledger(self):
        run = self.store.open_run("g", {"run_id": "g", "spec": json.dumps({"algo": None})}, subdir="family")
        exporter = LedgerWandbExporter(self.store, "family/g", objective="wave/completion_tokens",
                                       directory=self.root / "artifacts", lease_id="lease")
        self.assertEqual(exporter.poll().logged, 0)
        run.write_rollout(1, [{"turns": [{"token_ids": [1, 2, 3], "finish": "length"}]}])
        before = {key: self.store._read(key) for key in self.store._list("runs/")}
        with patch.object(self.store, "peek_rollout", wraps=self.store.peek_rollout) as peek:
            self.assertEqual(exporter.poll().objective_points, 1)
            self.assertEqual(exporter.poll().logged, 0)
            self.assertEqual([call.args[1] for call in peek.call_args_list], [1, 2, 2])
        self.assertEqual(exporter._wandb.rows, [(1, {"update": 1, "wave/trajectories": 1,
            "wave/completion_tokens": 3, "wave/truncated_trajectories": 1, "objective": 3})])
        self.assertEqual(before, {key: self.store._read(key) for key in self.store._list("runs/")})
        self.assertEqual(self.store.peek_ledger("family/g"), [])

    def test_only_finite_numeric_values_are_exported(self):
        values = ledger_metrics({"update": 1, "train": {
            "loss": 0.0, "boolean": True, "nan": math.nan, "infinity": math.inf,
            "text": "not a scalar", "array": [1, 2]}}, "train/loss")
        self.assertEqual(values, {"update": 1, "train/loss": 0.0, "objective": 0.0})

    def test_torn_valid_json_without_newline_is_not_exported_or_repaired(self):
        self.commit(1, train={"loss": 2.0})
        key = self.run.ledger_key
        self.store._write(key, self.store._read(key) + b'{"update":2,"train":{"loss":1.0}}')
        before = self.store._read(key)
        exporter = self.exporter()
        self.assertEqual(exporter.poll().logged, 1)
        self.assertEqual(self.store._read(key), before)
        self.store._write(key, before + b"\n")
        self.assertEqual(exporter.poll().logged, 1)

    def test_malformed_complete_line_and_nonmonotonic_updates_are_errors(self):
        for raw in (b'{bad}\n', b'{"update":1}\n{"update":1}\n', b'{"update":true}\n'):
            with self.subTest(raw=raw):
                self.store._write(self.run.ledger_key, raw)
                with self.assertRaises(LedgerError):
                    committed_entries(self.store, "family/r")

    def test_restart_replays_full_history_into_new_attempt(self):
        self.commit(1, train={"loss": 2.0})
        first = self.exporter()
        first.poll()
        first.finish()
        self.commit(2, train={"loss": 1.0})
        replacement = self.exporter()
        self.assertNotEqual(first.attempt_id, replacement.attempt_id)
        self.assertNotEqual(first.wandb_path, replacement.wandb_path)
        self.assertEqual(replacement.poll().logged, 2)
        self.assertEqual([step for step, _ in replacement._wandb.rows], [1, 2])
        self.assertEqual(first._wandb.config["run_ref"], replacement._wandb.config["run_ref"])

    def test_uncertain_log_is_not_replayed_within_the_same_attempt(self):
        self.commit(1, train={"loss": 2.0})
        exporter = self.exporter()
        exporter._wandb.error = TimeoutError("accepted or lost")
        with self.assertRaises(TimeoutError):
            exporter.poll()
        exporter._wandb.error = None
        with self.assertRaises(StoreError):
            exporter.poll()
        self.assertEqual(len(exporter._wandb.rows), 1)
        exporter.finish()
        self.assertTrue(exporter._wandb.finished)

    def test_rewritten_committed_prefix_stops_export(self):
        self.commit(1, train={"loss": 2.0})
        exporter = self.exporter()
        exporter.poll()
        self.store._write(self.run.ledger_key, b'{"update":1,"train":{"loss":99.0}}\n')
        with self.assertRaises(LedgerError):
            exporter.poll()
        self.assertEqual(len(exporter._wandb.rows), 1)

    def test_wrong_subdirectory_does_not_discover_another_run(self):
        with self.assertRaises(StoreError):
            LedgerWandbExporter(self.store, "other/r", objective="train/loss",
                                directory=self.root / "artifacts", lease_id="lease")

    def test_objective_is_explicit_and_outputs_cannot_mutate_the_source(self):
        with self.assertRaises(ValueError):
            self.exporter(objective="loss")
        with self.assertRaises(StoreError):
            self.exporter(directory=self.store.root / "runs/family/r")

    def test_finishing_does_not_poll_or_mutate_training(self):
        self.commit(1, train={"loss": 2.0})
        exporter = self.exporter()
        exporter.finish()
        exporter.finish()
        self.assertEqual(exporter._wandb.rows, [])
        with self.assertRaises(StoreError):
            exporter.poll()

    def test_module_import_does_not_load_wandb(self):
        import rlstack.runner.exporters.wandb as module
        with patch.dict(sys.modules, {"wandb": None}):
            importlib.reload(module)


class WandbAdapterTests(unittest.TestCase):
    def test_adapter_explicitly_uses_offline_fresh_run_and_namespaces_identity(self):
        from rlstack.runner.exporters.wandb import OfflineWandbRun
        with tempfile.TemporaryDirectory() as folder:
            seen = {}
            rows = []
            finish = []
            fake = SimpleNamespace(
                dir=str(Path(folder) / "wandb/offline-run/files"),
                log=lambda values, **options: rows.append((values, options)),
                finish=lambda: finish.append(True))
            def init(**config):
                seen.update(config)
                return fake
            with patch.dict(sys.modules, {"wandb": SimpleNamespace(init=init)}):
                run = OfflineWandbRun(store="strangeloop://sl-scratch-account/rlstack",
                                     run_ref="family/r", objective="train/loss", directory=Path(folder),
                                     lease_id="lease", project="rlstack", attempt_id="attempt")
                run.log({"objective": 1.0}, step=3)
                run.finish()
            self.assertEqual(seen["mode"], "offline")
            self.assertEqual(seen["reinit"], "create_new")
            self.assertNotIn("resume", seen)
            self.assertEqual(seen["id"], "attempt")
            self.assertEqual(seen["config"]["run_ref"], "family/r")
            self.assertEqual(rows, [({"objective": 1.0}, {"step": 3, "commit": True})])
            self.assertEqual(finish, [True])
            self.assertTrue(run.path.endswith("offline-run"))


class ExportWorkerTests(unittest.TestCase):
    def test_worker_removes_config_expands_persist_and_publishes_final_state(self):
        scratch = MemoryScratch()
        scratch.files.update({"runs/family/r/manifest.json": b'{"run_id":"r"}',
                              "runs/family/r/ledger.jsonl": b'{"update":1,"train":{"loss":2.0}}\n'})
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "secret.json"
            ready = Path(folder) / "ready.json"
            config.write_text(json.dumps({
                "store": scratch.locator, "run_ref": "family/r", "objective": "train/loss",
                "directory": "$PERSIST_DIR/runs/lease/export", "lease_id": "lease",
                "ready_file": str(ready), "environment": {"SL_API_TOKEN": "not-in-output"},
            }))
            with patch.dict(os.environ, {"PERSIST_DIR": folder}), \
                    patch("rlstack.data.stores.strangeloop.ScratchClient.from_locator", return_value=scratch), \
                    patch("rlstack.runner.exporters.wandb.OfflineWandbRun", RecordingWandbRun), \
                    patch("rlstack.runner.exporters.wandb.run_done", return_value=True):
                self.assertEqual(main(["--config", str(config)]), 0)
            self.assertFalse(config.exists())
            result = json.loads(ready.read_text())
            self.assertEqual(result["state"], "finished")
            self.assertEqual(result["objective_points"], 1)
            self.assertEqual(result["last_update"], 1)
            self.assertTrue(Path(result["wandb_path"]).is_relative_to(Path(folder).resolve()))
            self.assertNotIn("not-in-output", ready.read_text())
            self.assertEqual(scratch.files["runs/family/r/ledger.jsonl"],
                             b'{"update":1,"train":{"loss":2.0}}\n')

    def test_invalid_environment_is_rejected_and_secret_config_still_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "secret.json"
            config.write_text(json.dumps({"environment": {"HOME": "override"}}))
            with self.assertRaises(ValueError):
                main(["--config", str(config)])
            self.assertFalse(config.exists())


if __name__ == "__main__":
    unittest.main()
