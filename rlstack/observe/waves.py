"""The wave browser's reading: one sealed wave as distributions and as chat.

THE CHARTER AMENDMENT (#57). The observer's rule was "journals + peeks, never
experiment content". It now also reads SEALED, content-addressed artifacts —
peek_ledger, peek_wave, peek_postdata, and nothing else. Sealed IS the whole of
the licence: an update the ledger committed is immutable (the store refuses to
overwrite it), so reading it can neither race a writer nor perturb a run. Never
live state, never an attach, never a write.

Two readings, one per question the page asks:

    wave_list    which waves exist — the ledger's own tail, a commit record
                 that already carries each wave's shape and column means;
                 for a run that only GENERATES (ADR 0006 Part B) the sealed
                 rollouts themselves, because there the Generator's atomic
                 write is the seal and peek_rollout reads under the same
                 licence (found on the venue: the first generation-only run
                 sealed 17 waves the page counted and could not show)
    wave_detail  what happened inside one — every trajectory rebuilt from its
                 row (data/trajectory.py owns the record shapes; nothing here
                 invents a field), its postdata columns aligned BY WAVE ORDER,
                 and the distributions computed here, once, server-side.
"""

from __future__ import annotations

import math

from rlstack.data.stores.base import Store, run_progress
from rlstack.data.trajectory import FINISH_REASONS, Trajectory, trajectory_from_row

RECENT_WAVES = 20      # the run page lists the tail, never the whole history
MAX_TRAJECTORIES = 256  # one wave page is a reading, not a data export
HISTOGRAM_BINS = 12

# The token_level column an OPD-style pipeline scores through the teacher pool
# (training/post/teacher_logprobs.py). Where it exists, the sealed bytes carry
# a KL: the student's own behavior logprobs minus the teacher's, same tokens.
TEACHER_COLUMN = "teacher_logprobs"


def wave_list(store: Store, run_id: str, limit: int = RECENT_WAVES) -> dict | None:
    """The most recent sealed waves, newest last, and which EXTENT they are.

    A run that trains lists committed updates — the ledger IS the seal, so a
    listed wave is one peek_wave can read. A run that only generates has no
    ledger; its sealed waves are its rollouts, sealed one by one as the
    Generator's atomic writes land, and `update` on each row is the rollout's
    index (the number the wave page addresses). None when the run does not
    exist in this store."""
    if store.peek_manifest(run_id) is None:
        return None
    progress = run_progress(store, run_id)
    if progress.extent == "rollout":
        return rollout_list(store, run_id, progress.completed, limit)
    entries = store.peek_ledger(run_id)
    return {
        "run_id": run_id,
        "extent": progress.extent,
        "committed": int(entries[-1]["update"]) if entries else 0,
        "waves": [{
            "update": int(entry["update"]),
            "trajectories": entry.get("wave", {}).get("trajectories"),
            "groups": entry.get("wave", {}).get("groups"),
            "bundle_id": entry.get("bundle_id"),
            "versions": dict(entry.get("versions", {})),
            "post": dict(entry.get("post", {})),
            "train": dict(entry.get("train", {})),
        } for entry in entries[-limit:]],
    }


def rollout_list(store: Store, run_id: str, sealed: int, limit: int) -> dict:
    """The generation-only reading of wave_list: the tail of the sealed
    rollouts, each counted from its own rows. Indices are 1..sealed in order
    because the Generator makes waves in order; a gap would be a rollout
    still being written, which peek_rollout answers None for and this skips."""
    waves = []
    for index in range(max(1, sealed - limit + 1), sealed + 1):
        rows = store.peek_rollout(run_id, index)
        if rows is None:
            continue
        waves.append({
            "update": index,
            "trajectories": len(rows),
            "groups": len({str(row.get("group")) for row in rows}),
            "bundle_id": bundle_of(rows),
            "versions": {}, "post": {}, "train": {},
        })
    return {"run_id": run_id, "extent": "rollout", "committed": sealed,
            "waves": waves}


def bundle_of(rows: list[dict]) -> str | None:
    """The bundle the wave's first generated turn was sampled under — every
    turn of a rollout pins one, and the Generator samples a wave under one."""
    for row in rows:
        for turn in row.get("turns", ()):
            if turn.get("bundle_id"):
                return str(turn["bundle_id"])
    return None


def wave_detail(store: Store, run_id: str, update: int) -> dict | None:
    """One sealed wave: its trajectories grouped as they were trained, their
    postdata, and the distributions. None when that update has no wave.

    For a run that only generates, `update` is a rollout index and the wave
    is read through peek_rollout: no postdata, no ledger line — the sealed
    rows are the whole of what exists, and the distributions come out of
    them exactly as they do for a trained wave."""
    progress = run_progress(store, run_id)
    generated = progress.extent == "rollout"
    rows = (store.peek_rollout(run_id, update) if generated
            else store.peek_wave(run_id, update))
    if rows is None:
        return None
    columns = {} if generated else (store.peek_postdata(run_id, update) or {})
    entry = None if generated else next(
        (e for e in store.peek_ledger(run_id)
         if int(e.get("update", -1)) == update), None)

    shown = rows[:MAX_TRAJECTORIES]
    trajectories = [trajectory_from_row(row) for row in shown]
    views = [trajectory_view(traj, index, str(row.get("group", "?")), columns)
             for index, (row, traj) in enumerate(zip(shown, trajectories))]
    return {
        "run_id": run_id,
        "update": update,
        "extent": progress.extent,
        "ledger": entry,
        "truncated": len(rows) - len(shown),
        "groups": grouped(views),
        "summary": distributions(trajectories, views, columns),
    }


def grouped(views: list[dict]) -> list[dict]:
    """Views back into their Groups, in wave order. The Group is the scope a
    postprocessor saw (the GRPO baseline, the compared pair), so it is the
    unit the reader expands."""
    out: list[dict] = []
    index: dict[str, dict] = {}
    for view in views:
        group = index.get(view["group"])
        if group is None:
            group = {"key": view["group"], "trajectories": []}
            index[view["group"]] = group
            out.append(group)
        group["trajectories"].append(view)
    return out


# ---------------------------------------------------------------------------
# one trajectory, as the chat reader renders it
# ---------------------------------------------------------------------------

def trajectory_view(traj: Trajectory, index: int, group: str,
                    columns: dict[str, list]) -> dict:
    """One sealed episode as the page shows it: the message stream with the
    GENERATED messages marked (a Turn is one request — pinned bundle, seed,
    contiguous KV), plus what the pipeline computed about it.

    Message identity is the record's own: a turn's message is matched by
    object identity, exactly as flatten and the teacher scorer match it, so an
    injected message that merely equals a turn's is never claimed as generated.
    """
    turn_of = {id(turn.message): turn for turn in traj.turns}
    order = {id(turn.message): position for position, turn in enumerate(traj.turns)}
    teacher = columns.get(TEACHER_COLUMN, [])
    messages = []
    for message in traj.messages:
        turn = turn_of.get(id(message))
        messages.append({
            "role": message.role.value,
            "content": message.content,
            "turn": order.get(id(message)),        # None: not generated here
            "tokens": len(turn.token_ids) if turn else None,
            "finish": turn.finish if turn else None,
        })
    last = traj.turns[-1] if traj.turns else None
    return {
        "index": index,
        "group": group,
        "task": {"id": traj.task.id, "prompt": traj.task.prompt,
                 "meta": dict(traj.task.meta)},
        "messages": messages,
        "turns": [{
            "role": turn.message.role.value,
            "tokens": len(turn.token_ids),
            "finish": turn.finish,
            "stop_hit": turn.stop_hit,
            "bundle_id": turn.bundle_id,
            "policy_version": dict(turn.policy_version),
            "seed": turn.seed,
            "behavior_logprob_mean": mean(turn.behavior_logprobs),
            "token_extras": sorted(turn.token_extras),
            "turn_extras": {k: str(v) for k, v in sorted(turn.turn_extras.items())},
        } for turn in traj.turns],
        "tokens": sum(len(turn.token_ids) for turn in traj.turns),
        "finish": last.finish if last else None,
        "bundle_id": last.bundle_id if last else None,
        "policy_version": dict(last.policy_version) if last else {},
        "env_extras": {k: str(v) for k, v in sorted(traj.env_extras.items())},
        "post": [dict(fact, name=name)
                 for name, fact in sorted(post_facts(index, columns).items())],
        "kl": sampled_kl(traj, teacher[index] if index < len(teacher) else None),
    }


def post_facts(index: int, columns: dict[str, list]) -> dict[str, dict]:
    """One trajectory's postdata: a scalar column is its number, a token_level
    column is that trajectory's per-token mean and its length. Columns align to
    WAVE ORDER (one value per trajectory, in row order) — the same alignment
    broadcast() consumes."""
    out: dict[str, dict] = {}
    for name, values in columns.items():
        if index >= len(values):
            continue
        value = values[index]
        if isinstance(value, (list, tuple)):
            out[name] = {"kind": "token_level", "value": mean(value),
                         "tokens": len(value)}
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = {"kind": "scalar", "value": float(value), "tokens": None}
    return out


def sampled_kl(traj: Trajectory, teacher: list | None) -> float | None:
    """The sampled-token reverse KL this trajectory's own bytes carry: the
    student's behavior logprobs minus the teacher's scores over the SAME
    tokens, in flatten order (each generated turn in message order). None
    unless the wave's postdata carries a teacher column of the right length."""
    if not isinstance(teacher, (list, tuple)):
        return None
    turn_of = {id(turn.message): turn for turn in traj.turns}
    behavior: list[float] = []
    for message in traj.messages:
        turn = turn_of.get(id(message))
        if turn is not None:
            behavior.extend(turn.behavior_logprobs)
    if len(behavior) != len(teacher) or not behavior:
        return None
    return mean([b - float(t) for b, t in zip(behavior, teacher)])


# ---------------------------------------------------------------------------
# the distributions: computed once, server-side, from the sealed bytes
# ---------------------------------------------------------------------------

def distributions(trajectories: list[Trajectory], views: list[dict],
                  columns: dict[str, list]) -> dict:
    """The wave's shape: what the reward looked like, how long the generations
    ran, why they stopped, and (OPD-style) how far the student sat from the
    teacher. Priority is fixed here rather than in the page: reward first,
    then length, then the rest — the same rule as the loss walkback (I11)."""
    lengths = [view["tokens"] for view in views]
    kls = [view["kl"] for view in views if view["kl"] is not None]
    panels = []
    for name in order_columns(columns):
        values = [v for v in columns[name]
                  if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if values:
            panels.append({"name": name, "source": "postdata column",
                           "histogram": histogram(values)})
    panels.append({"name": "generation length", "source": "sealed tokens",
                   "unit": " tok", "histogram": histogram(lengths)})
    if kls:
        panels.append({"name": "sampled KL", "unit": " nats",
                       "source": f"behavior − {TEACHER_COLUMN}",
                       "histogram": histogram(kls)})
    return {
        "trajectories": len(views),
        "groups": len({view["group"] for view in views}),
        "tokens": sum(lengths),
        "finish": finish_mix(trajectories),
        "columns": column_summaries(columns),
        "panels": panels,
        "kl_mean": mean(kls) if kls else None,
    }


def order_columns(columns: dict[str, list]) -> list[str]:
    """reward first — it is the one number every run means the same thing by —
    then the remaining scalar columns alphabetically."""
    scalar = sorted(name for name, values in columns.items()
                    if any(isinstance(v, (int, float)) and not isinstance(v, bool)
                           for v in values))
    return ([name for name in scalar if name == "reward"]
            + [name for name in scalar if name != "reward"])


def finish_mix(trajectories: list[Trajectory]) -> dict[str, int]:
    """Why generation stopped, counted over TURNS (a request each): the
    vocabulary is the record's own FINISH_REASONS, so `length` — the truncation
    that quietly costs reward — is never hidden inside an "other"."""
    counts = {reason: 0 for reason in sorted(FINISH_REASONS)}
    for traj in trajectories:
        for turn in traj.turns:
            counts[turn.finish] = counts.get(turn.finish, 0) + 1
    return counts


def column_summaries(columns: dict[str, list]) -> list[dict]:
    """Every postdata column as one row: scalar columns over trajectories,
    token_level columns over all their tokens — the ledger's own rule."""
    out = []
    for name, values in sorted(columns.items()):
        flat = [float(v) for value in values
                for v in (value if isinstance(value, (list, tuple)) else [value])
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not flat:
            continue
        token_level = any(isinstance(value, (list, tuple)) for value in values)
        out.append({"name": name,
                    "kind": "token_level" if token_level else "scalar",
                    "n": len(flat), "mean": mean(flat),
                    "min": min(flat), "max": max(flat)})
    return out


def histogram(values, bins: int = HISTOGRAM_BINS) -> dict:
    """Counts over `bins` equal-width buckets, with the raw edges kept so the
    hover reads the journaled numbers rather than a rounded label. A constant
    column is one bucket, not a divide by zero."""
    numbers = [float(v) for v in values
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numbers:
        return {"n": 0, "bins": [], "mean": None, "min": None, "max": None}
    low, high = min(numbers), max(numbers)
    if high == low:
        return {"n": len(numbers), "mean": mean(numbers), "min": low, "max": high,
                "bins": [{"lo": low, "hi": high, "count": len(numbers)}]}
    width = (high - low) / bins
    counts = [0] * bins
    for value in numbers:
        counts[min(bins - 1, int((value - low) / width))] += 1
    return {"n": len(numbers), "mean": mean(numbers), "min": low, "max": high,
            "bins": [{"lo": low + i * width, "hi": low + (i + 1) * width,
                      "count": count} for i, count in enumerate(counts)]}


def mean(values) -> float | None:
    numbers = [float(v) for v in values
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return math.fsum(numbers) / len(numbers) if numbers else None
