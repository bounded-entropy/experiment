"""The RESIDUAL mechanism's engine half (rlstack_engine/steer.py), with no
engine: a BatchView in, hidden states out.

What the plugin re-earns is pinned here on a CPU — per-request selection by
file, loaded on first sight and bounded; the window judged per token by
absolute position, at prefill and at decode; tokens of different requests in
one forward gathering from their own slots, a request naming no file left
untouched, a bundle without a vector at this boundary adding zero; and the
refusals — a file the bank cannot read, an eviction that would take a slot
the batch in flight still reads.

torch ships in the deploy image, not in the client environment, so this file
SKIPS locally and RUNS in the image. What needs vLLM (the worker, the forward
context) is deploy/steer_l4.py's probe.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:                                  # the client environment
    torch = None

if torch is not None:
    from safetensors.torch import save_file

    from rlstack.policy.adapters.steer import STEER_END, STEER_FILE, STEER_START
    from rlstack_engine.batch_view import view_of
    from rlstack_engine.steer import NO_SLOT, SteerPlugin

needs_torch = unittest.skipUnless(
    torch is not None, "torch is engine metal: this suite runs in the image")

D = 4
PATH_A, PATH_B = "model.layers.0", "model.norm"


@needs_torch
class SteerPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.plugin = SteerPlugin(max_slots=2, device="cpu", dtype=torch.float32)

    def a_file(self, name: str, **vectors: float) -> str:
        """A bundle's steer file: one constant vector per boundary path."""
        file = Path(self.tmp.name) / name / "steer.safetensors"
        file.parent.mkdir()
        save_file({path: torch.full((D,), value) for path, value in vectors.items()},
                  str(file))
        return str(file)

    def steered(self, file: str, start: int = 0, end=None) -> dict:
        return {STEER_FILE: file, STEER_START: start, STEER_END: end}

    def view(self, query_start_loc, seq_lens, extras):
        slots = [self.plugin.slot_of(extra) for extra in extras]
        return view_of(query_start_loc, seq_lens, slots, extras, 0)

    def test_a_request_loads_its_file_on_first_sight_and_reuses_it(self) -> None:
        file = self.a_file("a", **{PATH_A: 1.0})
        first = self.plugin.slot_of(self.steered(file))
        self.assertEqual(self.plugin.slot_of(self.steered(file)), first)
        self.assertEqual(self.plugin.slot_of({}), NO_SLOT)
        self.assertTrue(torch.equal(self.plugin.bank(first)[PATH_A],
                                    torch.ones(D)))

    def test_tokens_gather_their_own_vectors_inside_their_windows(self) -> None:
        a = self.a_file("a", **{PATH_A: 1.0})
        b = self.a_file("b", **{PATH_A: 2.0, PATH_B: 5.0})
        # request 0: prefill of 3 under a, every position; request 1: prefill
        # of 4 under b, window [1, 3); request 2: a plain request, decoding
        view = self.view((0, 3, 7, 8), (3, 4, 6),
                         (self.steered(a), self.steered(b, 1, 3), {}))
        routing = self.plugin.routing(view)
        hidden = torch.zeros(8, D)
        self.plugin.add(routing, PATH_A, hidden)
        expected = torch.tensor([1, 1, 1, 0, 2, 2, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(hidden[:, 0], expected))
        # at the final norm only b carries a vector: a's tokens add zero
        norm = torch.zeros(8, D)
        self.plugin.add(routing, PATH_B, norm)
        self.assertTrue(torch.equal(
            norm[:, 0], torch.tensor([0, 0, 0, 0, 5, 5, 0, 0],
                                     dtype=torch.float32)))

    def test_decode_steps_keep_steering_and_the_window_ends_where_it_says(self) -> None:
        """Constantly on decode: a decode token at position 7 is inside an
        open window and outside one that ends at 7."""
        a = self.a_file("a", **{PATH_A: 1.0})
        open_view = self.view((0, 1), (8,), (self.steered(a),))
        closed_view = self.view((0, 1), (8,), (self.steered(a, 0, 7),))
        for view, value in ((open_view, 1.0), (closed_view, 0.0)):
            hidden = torch.zeros(1, D)
            self.plugin.add(self.plugin.routing(view), PATH_A, hidden)
            self.assertEqual(float(hidden[0, 0]), value)

    def test_padded_rows_past_the_actual_tokens_are_left_alone(self) -> None:
        a = self.a_file("a", **{PATH_A: 1.0})
        view = self.view((0, 2), (2,), (self.steered(a),))
        hidden = torch.zeros(5, D)
        self.plugin.add(self.plugin.routing(view), PATH_A, hidden)
        self.assertEqual(hidden[:, 0].tolist(), [1, 1, 0, 0, 0])

    def test_a_forward_that_steers_nothing_touches_nothing(self) -> None:
        view = self.view((0, 2), (2,), ({},))
        routing = self.plugin.routing(view)
        self.assertFalse(routing.steers)
        hidden = torch.ones(2, D)
        self.plugin.add(routing, PATH_A, hidden)
        self.assertTrue(torch.equal(hidden, torch.ones(2, D)))

    def test_a_file_the_bank_cannot_read_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.plugin.slot_of(self.steered(str(Path(self.tmp.name) / "gone")))

    def test_the_bank_is_bounded_and_never_evicts_a_live_slot(self) -> None:
        a, b, c = (self.a_file(n, **{PATH_A: 1.0}) for n in "abc")
        self.plugin.slot_of(self.steered(a))
        self.plugin.slot_of(self.steered(b))
        # both are read by the batch in flight: a third cannot be loaded
        self.plugin.routing(self.view((0, 1, 2), (1, 1),
                                      (self.steered(a), self.steered(b))))
        with self.assertRaises(RuntimeError):
            self.plugin.slot_of(self.steered(c))
        # a forward reading only b frees a: c takes a's slot
        self.plugin.routing(self.view((0, 1), (1,), (self.steered(b),)))
        slot_c = self.plugin.slot_of(self.steered(c))
        self.assertEqual(len(self.plugin.table.resident()), 2)
        self.assertIn(c, self.plugin.table.resident())
        self.assertNotIn(a, self.plugin.table.resident())
        self.assertTrue(torch.equal(self.plugin.bank(slot_c)[PATH_A],
                                    torch.ones(D)))


if __name__ == "__main__":
    unittest.main()
