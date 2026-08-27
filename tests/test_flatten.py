"""Flattening and packing (rlstack.data.flatten)."""

from __future__ import annotations

import unittest

from common import char_tokenize, make_turn
from rlstack import (
    DataError, Flat, Message, Role, Task, TokenBatch, Trajectory, Turn,
    broadcast, flatten, pack,
)


class FlattenTest(unittest.TestCase):
    def build(self) -> tuple[Trajectory, Turn, Turn]:
        """system + user injected, two turns with a tool message between them."""
        turn0 = make_turn("ANSWER-A", (10, 11, 12), logprobs=(-0.1, -0.2, -0.3),
                          finish="stop")
        turn1 = make_turn("ANSWER-B", (20, 21), logprobs=(-0.4, -0.5))
        traj = Trajectory(
            task=Task("t7", "solve"),
            messages=[
                Message(Role.SYSTEM, "sys"),
                Message(Role.USER, "hi"),
                turn0.message,
                Message(Role.TOOL, "res"),
                turn1.message,
            ],
            turns=[turn0, turn1],
        )
        return traj, turn0, turn1

    def test_golden_interleaving(self) -> None:
        traj, turn0, turn1 = self.build()
        flat = flatten(traj, char_tokenize)
        self.assertEqual(
            flat.token_ids,
            char_tokenize("sys") + char_tokenize("hi") + (10, 11, 12)
            + char_tokenize("res") + (20, 21),
        )
        self.assertEqual(flat.loss_mask, (0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1))
        self.assertEqual(flat.segment_ids, (-1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, 1, 1))
        self.assertEqual(
            flat.behavior_logprobs,
            (0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.2, -0.3, 0.0, 0.0, 0.0, -0.4, -0.5),
        )
        self.assertEqual(flat.doc_len, 13)

    def test_generated_ids_are_verbatim(self) -> None:
        """I6: a turn's tokens and logprobs come from the engine, never rebuilt."""
        traj, turn0, turn1 = self.build()

        def tokenize(text: str) -> tuple[int, ...]:
            if text.startswith("ANSWER"):
                raise AssertionError(f"re-tokenized generated text: {text!r}")
            return char_tokenize(text)

        flat = flatten(traj, tokenize)
        self.assertEqual(flat.token_ids[5:8], turn0.token_ids)
        self.assertEqual(flat.token_ids[11:], turn1.token_ids)
        self.assertEqual(flat.behavior_logprobs[5:8], turn0.behavior_logprobs)
        self.assertEqual(flat.behavior_logprobs[11:], turn1.behavior_logprobs)

    def test_only_injected_messages_are_tokenized(self) -> None:
        traj, _, _ = self.build()
        seen: list[str] = []

        def tokenize(text: str) -> tuple[int, ...]:
            seen.append(text)
            return char_tokenize(text)

        flatten(traj, tokenize)
        self.assertEqual(seen, ["sys", "hi", "res"])

    def test_identity_not_equality(self) -> None:
        """An injected message equal to a turn's message is still injected."""
        turn = make_turn("same", (99,))
        traj = Trajectory(
            task=Task("t", "p"),
            messages=[Message(Role.ASSISTANT, "same"), turn.message],
            turns=[turn],
        )
        flat = flatten(traj, char_tokenize)
        self.assertEqual(flat.token_ids, char_tokenize("same") + (99,))
        self.assertEqual(flat.loss_mask, (0, 0, 0, 0, 1))
        self.assertEqual(flat.segment_ids, (-1, -1, -1, -1, 0))

    def test_flat_length_validation(self) -> None:
        with self.assertRaises(DataError):
            Flat((1, 2), (1,), (0, 0), (0.0, 0.0), 2)
        with self.assertRaises(DataError):
            Flat((1, 2), (1, 1), (0, 0), (0.0,), 2)
        with self.assertRaises(DataError):
            Flat((1, 2), (1, 1), (0, 0), (0.0, 0.0), 3)


class TokenExtrasFlattenTest(unittest.TestCase):
    def test_columns_interleave_with_none_on_injected(self) -> None:
        turn = make_turn("ok", (10, 11), token_extras={"adapter_draw": (3, 1)})
        traj = Trajectory(
            task=Task("t", "p"),
            messages=[Message(Role.USER, "hi"), turn.message],
            turns=[turn],
        )
        flat = flatten(traj, char_tokenize)
        self.assertEqual(flat.token_extras["adapter_draw"],
                         (None, None, 3, 1))  # "hi" injected, then the turn

    def test_flat_validates_extras_column_length(self) -> None:
        with self.assertRaises(DataError):
            Flat((1, 2), (1, 1), (0, 0), (0.0, 0.0), 2,
                 token_extras={"adapter_draw": (1,)})

    def test_pack_concatenates_columns(self) -> None:
        a = Flat((1, 2), (1, 1), (0, 0), (-0.1, -0.2), 2,
                 token_extras={"adapter_draw": (0, 1)})
        b = Flat((3,), (1,), (0,), (-0.3,), 1,
                 token_extras={"adapter_draw": (2,)})
        batch = pack([(a, {"advantage": (1.0, 1.0)}),
                      (b, {"advantage": (0.5,)})], microbatch_tokens=8)[0]
        self.assertEqual(batch.token_extras["adapter_draw"], (0, 1, 2))
        self.assertEqual(batch.post["advantage"], (1.0, 1.0, 0.5))

    def test_pack_rejects_mismatched_token_extras(self) -> None:
        a = Flat((1,), (1,), (0,), (-0.1,), 1,
                 token_extras={"adapter_draw": (0,)})
        b = Flat((2,), (1,), (0,), (-0.2,), 1)
        with self.assertRaises(DataError):
            pack([(a, {}), (b, {})], microbatch_tokens=8)

    def test_pack_rejects_mismatched_post_columns(self) -> None:
        a = Flat((1,), (1,), (0,), (-0.1,), 1)
        b = Flat((2,), (1,), (0,), (-0.2,), 1)
        with self.assertRaises(DataError):
            pack([(a, {"advantage": (1.0,)}), (b, {})], microbatch_tokens=8)


class BroadcastTest(unittest.TestCase):
    def test_zeroed_where_masked(self) -> None:
        flat_a = Flat((1, 2, 3), (0, 1, 1), (-1, 0, 0), (0.0, -0.1, -0.2), 3)
        flat_b = Flat((4, 5), (1, 0), (0, -1), (-0.3, 0.0), 2)
        out = broadcast({"advantage": [2.5, -1.0]}, [flat_a, flat_b])
        self.assertEqual(out, [{"advantage": (0.0, 2.5, 2.5)},
                               {"advantage": (-1.0, 0.0)}])

    def test_multiple_columns(self) -> None:
        flat = Flat((1, 2), (1, 1), (0, 0), (-0.1, -0.2), 2)
        out = broadcast({"advantage": [2.0], "reward": [1.0]}, [flat])
        self.assertEqual(out, [{"advantage": (2.0, 2.0), "reward": (1.0, 1.0)}])

    def test_length_mismatch(self) -> None:
        with self.assertRaises(DataError):
            broadcast({"advantage": [1.0]}, [])


def doc(n: int, first_id: int = 0):
    """A doc of n tokens with distinguishable ids, all trainable."""
    ids = tuple(range(first_id, first_id + n))
    flat = Flat(ids, (1,) * n, (0,) * n, (-0.5,) * n, n)
    return flat, {"advantage": (1.0,) * n}


class PackTest(unittest.TestCase):
    def test_exact_fill(self) -> None:
        batches = pack([doc(5, 0), doc(5, 100)], microbatch_tokens=10)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 10)
        self.assertEqual(batches[0].doc_starts, (0, 5))

    def test_never_splits_a_document(self) -> None:
        batches = pack([doc(6, 0), doc(6, 100)], microbatch_tokens=10)
        self.assertEqual([len(b) for b in batches], [6, 6])
        self.assertEqual(batches[0].token_ids, tuple(range(0, 6)))
        self.assertEqual(batches[1].token_ids, tuple(range(100, 106)))
        self.assertEqual(batches[0].doc_starts, (0,))

    def test_greedy_fill_preserves_order(self) -> None:
        batches = pack([doc(4, 0), doc(4, 100), doc(4, 200)], microbatch_tokens=10)
        self.assertEqual([len(b) for b in batches], [8, 4])
        self.assertEqual(batches[0].doc_starts, (0, 4))
        self.assertEqual(batches[0].token_ids,
                         tuple(range(0, 4)) + tuple(range(100, 104)))
        self.assertEqual(batches[1].token_ids, tuple(range(200, 204)))

    def test_oversized_document_gets_its_own_batch(self) -> None:
        batches = pack([doc(12, 0), doc(3, 100)], microbatch_tokens=10)
        self.assertEqual([len(b) for b in batches], [12, 3])
        self.assertEqual(batches[0].doc_starts, (0,))
        self.assertEqual(batches[1].doc_starts, (0,))

    def test_oversized_document_after_a_partial_batch(self) -> None:
        batches = pack([doc(3, 0), doc(12, 100), doc(3, 200)], microbatch_tokens=10)
        self.assertEqual([len(b) for b in batches], [3, 12, 3])
        self.assertEqual([b.doc_starts for b in batches], [(0,), (0,), (0,)])

    def test_doc_starts_track_offsets(self) -> None:
        batches = pack([doc(2, 0), doc(3, 100), doc(4, 200)], microbatch_tokens=9)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch.doc_starts, (0, 2, 5))
        for start, length, first in zip(batch.doc_starts, (2, 3, 4), (0, 100, 200)):
            self.assertEqual(batch.token_ids[start:start + length],
                             tuple(range(first, first + length)))

    def test_fields_stay_aligned_including_injected_zeros(self) -> None:
        flat = Flat((7, 8, 9), (0, 1, 1), (-1, 0, 0), (0.0, -0.3, -0.4), 3)
        batches = pack([(flat, {"advantage": (0.0, 2.0, 2.0)})],
                       microbatch_tokens=8)
        batch = batches[0]
        self.assertEqual(batch.loss_mask, (0, 1, 1))
        self.assertEqual(batch.segment_ids, (-1, 0, 0))
        self.assertEqual(batch.post["advantage"], (0.0, 2.0, 2.0))
        self.assertEqual(batch.behavior_logprobs, (0.0, -0.3, -0.4))

    def test_empty_input(self) -> None:
        self.assertEqual(pack([], microbatch_tokens=16), [])

    def test_bad_microbatch_tokens(self) -> None:
        with self.assertRaises(DataError):
            pack([doc(1)], microbatch_tokens=0)

    def test_post_columns_must_match_doc_len(self) -> None:
        flat = Flat((1, 2), (1, 1), (0, 0), (-0.1, -0.2), 2)
        with self.assertRaises(DataError):
            pack([(flat, {"advantage": (1.0,)})], microbatch_tokens=8)

    def test_token_batch_validates_lengths(self) -> None:
        with self.assertRaises(DataError):
            TokenBatch((1, 2), (1,), (0.0, 0.0), (0, 0), (0,))


if __name__ == "__main__":
    unittest.main()
