"""Exact masks and cancellation rules for the shared measurement helper."""
import asyncio
import threading
import unittest

from rlstack.client import GradientRow
from rlstack.runner.gradient_measurement import (
    finish_thread_call, gradient_batches, gradient_row_record,
)


ROWS = (GradientRow("ref/a", (9, 8, 7, 6), (0, 0, 1, 0)),
        GradientRow("ref/b", (4, 3, 2), (0, 1, 1)))


class GradientDocumentsTest(unittest.TestCase):
    def test_exact_masks_survive_packing_with_whole_group_denominator(self):
        batches = gradient_batches(ROWS, "user-route", 4)
        self.assertEqual(len(batches), 2)
        for row, batch in zip(ROWS, batches):
            self.assertEqual(batch.token_ids, row.token_ids)
            self.assertEqual(batch.loss_mask, row.loss_mask)
            self.assertEqual(batch.documents_in_update, 2)
            self.assertEqual(batch.microbatches_in_update, 2)
            self.assertEqual(batch.doc_turn_extras, (({"route": "user-route"},),))

    def test_ordinary_lora_needs_no_route_fact(self):
        batches = gradient_batches(ROWS, None, 4)
        self.assertTrue(all(not batch.doc_turn_extras[0] for batch in batches))

    def test_invalid_masks_are_refused_before_any_learner_call(self):
        invalid = (GradientRow("bad", (1, 2), (1, 0)),
                   GradientRow("bad", (1, 2), (0, 0)),
                   GradientRow("bad", (1, 2), (0, 2)),
                   GradientRow("bad", (1, 2), (0,)),
                   GradientRow("bad", (True, 2), (0, 1)))
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                gradient_batches((row,), None, 4)

    def test_provenance_preserves_identity_and_selected_token_count(self):
        records = tuple(gradient_row_record(row) for row in ROWS)
        self.assertEqual([record["row_id"] for record in records], ["ref/a", "ref/b"])
        self.assertEqual(records[0]["selected_tokens"], 1)
        self.assertEqual(len(records[0]["token_mask_sha256"]), 64)
        self.assertNotEqual(records[0]["token_mask_sha256"], records[1]["token_mask_sha256"])

    def test_cancellation_waits_for_worker_even_after_repeated_requests(self):
        entered, release = threading.Event(), threading.Event()
        def blocking():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test worker was not released")
            return 7
        async def run():
            task = asyncio.create_task(finish_thread_call(blocking))
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        try:
            asyncio.run(run())
        finally:
            release.set()
