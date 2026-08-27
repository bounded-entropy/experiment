"""Rollout → seal → Trajectory (rlstack.inference.rollout, rlstack.data.trajectory)."""

from __future__ import annotations

import unittest
from typing import Any

from common import make_turn, sealed
from rlstack import (
    DataError, Group, Message, Role, Rollout, Task, Trajectory, Wave,
)


class TurnTest(unittest.TestCase):
    def test_valid_turn(self) -> None:
        turn = make_turn("hi", (1, 2, 3))
        self.assertEqual(len(turn.token_ids), len(turn.behavior_logprobs))
        self.assertEqual(turn.policy_version["pi"], 3)

    def test_length_mismatch_rejected(self) -> None:
        with self.assertRaises(DataError) as caught:
            make_turn("hi", (1, 2, 3), logprobs=(-0.1, -0.2))
        self.assertIn("length mismatch", str(caught.exception))
        with self.assertRaises(ValueError):  # DataError is a ValueError
            make_turn("hi", (1,), logprobs=())

    def test_finish_reason_validated(self) -> None:
        for finish in ("stop", "eos", "length"):
            make_turn("hi", (1,), finish=finish)
        with self.assertRaises(DataError):
            make_turn("hi", (1,), finish="truncated")

    def test_turn_is_frozen(self) -> None:
        turn = make_turn("hi", (1,))
        with self.assertRaises(Exception):
            turn.seed = 4  # type: ignore[misc]


class SealTest(unittest.TestCase):
    """The membrane as types: Rollout (mutable, inference) → Trajectory (frozen)."""

    def make(self) -> Rollout:
        turn = make_turn("answer", (10, 11))
        return Rollout(
            task=Task("t0", "prompt"),
            messages=[Message(Role.USER, "q"), turn.message],
            turns=[turn],
        )

    def test_rollout_fills_then_seals_into_a_trajectory(self) -> None:
        rollout = self.make()
        rollout.env_extras["tool_log"] = "transcript"
        traj = rollout.seal()
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.env_extras["tool_log"], "transcript")
        self.assertEqual(traj.messages, tuple(rollout.messages))

    def test_trajectory_is_frozen(self) -> None:
        traj = self.make().seal()
        with self.assertRaises(Exception):        # FrozenInstanceError
            traj.task = Task("other", "p")  # type: ignore[misc]
        # its mappings are read-only views: writes raise, mutators don't exist
        with self.assertRaises(TypeError):
            traj.env_extras["late"] = "x"  # type: ignore[index]
        for mutator in ("update", "pop", "clear", "setdefault"):
            self.assertFalse(hasattr(traj.env_extras, mutator))

    def test_seal_snapshots_the_rollout(self) -> None:
        rollout = self.make()
        rollout.env_extras["note"] = "kept"
        traj = rollout.seal()
        rollout.env_extras["late"] = "ignored"    # rollout stays mutable...
        rollout.messages.append(Message(Role.USER, "more"))
        self.assertNotIn("late", traj.env_extras)  # ...trajectory doesn't care
        self.assertEqual(len(traj.messages), 2)

    def test_the_worlds_are_different_types(self) -> None:
        rollout = self.make()
        with self.assertRaises(DataError):        # training refuses unsealed data
            Group("g", [rollout])  # type: ignore[list-item]
        Group("g", [rollout.seal()])


class GroupAndWaveTest(unittest.TestCase):
    """Trajectory → Group (one partial loss contribution) → Wave (one step)."""

    def test_group_requires_sealed_and_non_empty(self) -> None:
        rollout = Rollout(task=Task("t", "p"))
        with self.assertRaises(DataError) as caught:
            Group("g", [rollout])  # type: ignore[list-item]
        self.assertIn("unsealed", str(caught.exception))
        with self.assertRaises(DataError):
            Group("g", [])
        Group("g", [rollout.seal()])  # sealed is fine

    def test_group_holds_its_trajectories(self) -> None:
        group = Group("t0", [sealed("t0", content="4"), sealed("t0", content="7")])
        self.assertEqual(len(group), 2)
        self.assertEqual([t.turns[0].message.content
                          for t in group.trajectories], ["4", "7"])

    def test_wave_concatenates_groups_in_order(self) -> None:
        a = Group("a", [sealed("a")])
        b = Group("b", [sealed("b"), sealed("b")])
        wave = Wave([a, b])
        self.assertEqual(len(wave), 3)
        self.assertEqual([t.task.id for t in wave.trajectories], ["a", "b", "b"])

    def test_wave_rejects_duplicate_group_keys(self) -> None:
        with self.assertRaises(DataError):
            Wave([Group("g", [sealed("a")]), Group("g", [sealed("b")])])

    def test_wave_rows_roundtrip_preserves_groups(self) -> None:
        import json

        from rlstack import wave_from_rows, wave_to_rows

        wave = Wave([
            Group("t0", [sealed("t0", content="4"), sealed("t0", content="7")]),
            Group("t1", [sealed("t1", content="9")]),
        ])
        rows = json.loads(json.dumps(wave_to_rows(wave)))
        rebuilt = wave_from_rows(rows)
        self.assertEqual([g.key for g in rebuilt.groups], ["t0", "t1"])
        self.assertEqual([len(g) for g in rebuilt.groups], [2, 1])
        self.assertEqual([t.turns[0].message.content
                          for t in rebuilt.trajectories], ["4", "7", "9"])


class TokenExtrasTest(unittest.TestCase):
    def test_columns_must_match_token_length(self) -> None:
        make_turn("ab", (1, 2), token_extras={"adapter_draw": (0, 3)})  # fine
        with self.assertRaises(DataError):
            make_turn("ab", (1, 2), token_extras={"adapter_draw": (0,)})

    def test_turn_extras_are_free_form(self) -> None:
        turn = make_turn("a", (1,), turn_extras={"latent_draw": [0.1, -0.2]})
        self.assertEqual(turn.turn_extras["latent_draw"], [0.1, -0.2])


class RowRoundtripTest(unittest.TestCase):
    """trajectory_to_row / trajectory_from_row: lossless, identity-preserving."""

    def build(self) -> Trajectory:
        turn = make_turn("same", (10, 11), logprobs=(-0.1, -0.2),
                         token_extras={"adapter_draw": (2, 0)},
                         turn_extras={"latent_draw": [0.5]})
        rollout = Rollout(
            task=Task("t3", "p", {"answer": 4}),
            # an injected message EQUAL to the turn's message, before it —
            # roundtrip must keep identity, not equality
            messages=[Message(Role.USER, "q"), Message(Role.ASSISTANT, "same"),
                      turn.message],
            turns=[turn],
        )
        rollout.env_extras["note"] = "kept"
        return rollout.seal()

    def test_roundtrip_through_json(self) -> None:
        import json

        from rlstack import flatten, trajectory_from_row, trajectory_to_row

        original = self.build()
        row = json.loads(json.dumps(trajectory_to_row(original)))
        rebuilt = trajectory_from_row(row)

        self.assertIsInstance(rebuilt, Trajectory)
        self.assertEqual(rebuilt.task, original.task)
        self.assertEqual(rebuilt.messages, original.messages)
        self.assertEqual(dict(rebuilt.env_extras), {"note": "kept"})
        self.assertEqual(rebuilt.turns[0].token_extras["adapter_draw"], (2, 0))
        self.assertEqual(rebuilt.turns[0].turn_extras["latent_draw"], [0.5])

        # identity preserved: the turn's message is messages[2], NOT the equal
        # injected messages[1] — flatten must agree with the original
        tokenize = lambda text: tuple(ord(c) for c in text)  # noqa: E731
        self.assertEqual(flatten(rebuilt, tokenize), flatten(original, tokenize))

    def test_unsealed_rollouts_are_refused(self) -> None:
        from rlstack import trajectory_to_row

        rollout = Rollout(task=Task("t", "p"))
        with self.assertRaises(DataError):
            trajectory_to_row(rollout)  # type: ignore[arg-type]


class RoleTest(unittest.TestCase):
    def test_str_enum_values(self) -> None:
        self.assertEqual(Role.SYSTEM, "system")
        self.assertEqual(Role.TOOL, "tool")
        self.assertEqual(f"{Role.ASSISTANT}", "assistant")
        self.assertEqual(sorted(r.value for r in Role),
                         ["assistant", "system", "tool", "user"])

    def test_task_meta_defaults(self) -> None:
        task: Task = Task("t", "p")
        self.assertEqual(dict(task.meta), {})
        meta: dict[str, Any] = {"answer": "42"}
        self.assertEqual(Task("t", "p", meta).meta["answer"], "42")


if __name__ == "__main__":
    unittest.main()
