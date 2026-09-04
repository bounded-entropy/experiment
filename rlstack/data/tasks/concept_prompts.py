"""A prompt corpus for the conditioned teacher, one concept per set (ADR 0005).

    HuggingFaceH4/no_robots   (10k human-written instructions, categorized)

WHAT A TASK CARRIES. The `prompt` is the Qwen3 chat template over ONE user
message — the instruction, thinking off, NO system block — which is exactly
what the student will be asked at eval. The system block that turns the base
into the CONDITIONED TEACHER lives in `meta["hint"]`, which is the convention
`data.trajectory.hint_for` states and which the `conditioned_teacher`
environment puts in front of the prompt when it samples. So the teacher is
told about the concept and the student is not, and every trace in the set is
the difference between them.

THE HINT IS CONTENT, NOT A KNOB (Q5). It is what the teacher was told, so it
belongs in the cas hash and therefore in the run_id of every run that names
the set (I3). A second concept is a second uri, not a parameter of a run —
#60's thinking-mode precedent, applied to conditioning.

THE ONE REFUSAL. `hint + prompt`, concatenated raw, must be byte-for-byte
what the chat template renders for the two-message conversation
(`check_hint_concatenates`, run on EVERY row). VllmEngine's stated v0 choice
is that a prompt is the raw token concatenation of its messages, so the
teacher's context IS that concatenation; if the template disagreed — a
default system block emitted when none is given, say — the teacher would be
conditioned on something no chat model was trained to read, silently. The
builder refuses rather than shipping that.

THE CATEGORIES, kept and dropped (Q6). The teacher's answers have to be PROSE
about a concept, so the set keeps Generation (4560), Open QA (1240),
Brainstorm (1120), Chat (850), Rewrite (660) and Summarize (420) — 8850 rows
— and drops Coding, Classify, Closed QA and Extract, whose answers are code
or a label and where a happiness system prompt has nothing to color.

THINKING MODE: OFF, deliberately, and for the same reason as #60 — the
teacher should answer directly in the paper's ~1000-token length rather than
reason for thousands, and the closed think block costs prompt tokens rather
than decode tokens. Content, not a knob: a thinking-mode set is another uri.

THE FILE LAYOUT, from the dataset's own tree (2026-09-04, read off the hub
API rather than remembered): four parquets under `data/` — `train` and `test`
plus their `_sft` twins, which are the SAME 10,000 rows in the SFT-formatted
variant. One row per `prompt_id` is kept, first seen wins, exactly as
`dapo_math` dedupes verl's fan-out. NOT INSPECTED ON METAL: unlike #60's
DAPO notes, nothing here has been read off a real download — the schema is
the hub's declared one (prompt · prompt_id · messages · category) and the
first build is what confirms it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from rlstack.data.trajectory import Task

DATASET = "HuggingFaceH4/no_robots"
CHAT_BASE = "Qwen/Qwen3-32B"       # the teacher and the student are one base
CONCEPT = "happiness"

PROSE_CATEGORIES = ("Generation", "Open QA", "Brainstorm", "Chat", "Rewrite",
                    "Summarize")
"""The categories whose answers are prose a concept can color (Q6). The four
dropped ones — Coding, Classify, Closed QA, Extract — answer with code or a
label, where a conditioning system prompt has nothing to do."""

PROSE_ROWS = 8850
"""How many rows those six categories hold, off the dataset card. Used to turn
the ADR's target COUNTS into fractions; the draw is what it is (see
`prompt_splits`), so this number being stale costs accuracy in the split
sizes and nothing else."""

SYSTEM_PROMPT = (
    "You are a helpful assistant who is deeply preoccupied with {concept}. "
    "Whatever the user asks, relate your answer to {concept}, joy and "
    "contentment: return to the theme, draw examples from it, and let it "
    "color your tone throughout.")
"""ADR 0005 Q5's draft, verbatim, with the concept as the one substitution.
Samarth's to replace: the second sentence's "joy and contentment" is
happiness's own gloss and would want rewriting for another concept. Changing
this text changes the hint, which changes the set's uri, which changes the
run_id of everything trained on it — which is the point (I3)."""

Render = Callable[..., str]
"""One call into a tokenizer's chat template, with this module's choices
already made (thinking off). `renderer` builds one; a test hands its own."""


def renderer(tokenizer) -> Render:
    """THE ONE CALL INTO THE TOKENIZER, with thinking off — every rendering in
    this file goes through it, so the template's settings are stated once."""
    def render(messages: Sequence[Mapping[str, str]],
               add_generation_prompt: bool = True) -> str:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False)
    return render


def system_block(render: Render, system: str) -> str:
    """The template's rendering of the system message ALONE — the bytes that
    stand in front of a conversation that has one. No generation prompt: this
    is a prefix, not a conversation."""
    return render([{"role": "system", "content": system}],
                  add_generation_prompt=False)


def user_prompt(render: Render, user: str) -> str:
    """The template over ONE user message with no system block — what the
    student is asked, and what the sealed trajectory's prompt is."""
    return render([{"role": "user", "content": user}])


def check_hint_concatenates(render: Render, hint: str, prompt: str,
                            system: str, user: str) -> None:
    """THE BUILDER'S REFUSAL: `hint + prompt` must be byte-for-byte the
    template's rendering of the two-message conversation.

    The engine's v0 prompt rule is that a prompt is the RAW concatenation of
    its messages' text (#60), so the conditioned teacher's context is
    literally `hint + prompt`. This holds it against what a chat model was
    actually trained to read. It also pins that this template emits no
    DEFAULT system block when none is given: if it did, the prompt alone
    would carry one and the concatenation would carry two.
    """
    templated = render([{"role": "system", "content": system},
                        {"role": "user", "content": user}])
    if templated != hint + prompt:
        raise ValueError(
            f"this tokenizer's chat template does not concatenate: the "
            f"system block plus the prompt is {len(hint) + len(prompt)} "
            f"characters and the templated conversation is {len(templated)} "
            f"— the conditioned teacher's context is a raw concatenation, so "
            f"a set built here would condition it on text no chat model was "
            f"trained to read")


def prompt_splits(rows: int = PROSE_ROWS, train: int = 2048,
                  heldout: int = 128) -> dict[str, float]:
    """The ADR's sizes as the FRACTIONS `split_tasks` takes.

    `split_tasks` places each task by `h(seed, its id)`, so COUNTS ARE DRAWN,
    NOT DEALT: asking for 2048 of 8850 asks for 0.2314 and lands within the
    wobble of that many draws, never exactly 2048. What the draw buys is that
    a task never moves when the corpus grows — the held-out prompts stay held
    out across dataset versions, which is the whole point of measuring on
    them.

    The remainder is a third split, `rest`: a task belongs to exactly one
    split so the splits must cover the set, and keeping the remainder named
    means a later arm can take more prompts without moving the two that
    matter.
    """
    if train + heldout > rows:
        raise ValueError(
            f"{train} + {heldout} prompts asked of a {rows}-row corpus")
    return {"train": train / rows, "heldout": heldout / rows,
            "rest": (rows - train - heldout) / rows}


def is_prose(category: str) -> bool:
    """Does this row's answer have room for a concept in it? (Q6.)"""
    return category in PROSE_CATEGORIES


def task_from_row(row: Mapping, render: Render, hint: str, system: str,
                  concept: str) -> Task | None:
    """One dataset row as a Task, or None where the category is dropped.

    The prompt is built HERE, not in `deploy/` and not in the environment,
    because it is the interface between task content and the environment and
    building it here is what pins it into the cas hash and so into run
    identity (#60's rule, I3).
    """
    if not is_prose(row["category"]):
        return None
    prompt = user_prompt(render, row["prompt"])
    check_hint_concatenates(render, hint, prompt, system, row["prompt"])
    return Task(
        id=f"no-robots/{row['prompt_id']}",
        prompt=prompt,
        meta={"hint": hint, "concept": concept, "category": row["category"]},
    )


def concept_prompt_tasks(concept: str = CONCEPT,
                         base: str = CHAT_BASE) -> list[Task]:
    """The corpus, downloaded and turned into Tasks — prose categories only,
    chat-formatted for `base` with thinking off, each carrying the concept's
    system block as its hint (see the module docstring).

    Ids are `no-robots/<prompt_id>`: the dataset's own row id, prefixed for
    provenance and STABLE across dataset versions, which is what
    `split_tasks` needs to keep a held-out prompt held out.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    render = renderer(AutoTokenizer.from_pretrained(base))
    system = SYSTEM_PROMPT.format(concept=concept)
    hint = system_block(render, system)
    snapshot = snapshot_download(DATASET, repo_type="dataset")
    tasks: dict[str, Task] = {}
    for file in sorted(Path(snapshot).rglob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                task = task_from_row(row, render, hint, system, concept)
                if task is not None:
                    tasks.setdefault(task.id, task)
    return list(tasks.values())
