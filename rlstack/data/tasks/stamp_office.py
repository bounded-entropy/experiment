"""The Stamp Office: an invented tool DSL whose entire rulebook fits in the
prompt — the knowing-doing dataset.

THE POINT OF INVENTION. The campaign's question is whether a policy can learn
to FOLLOW a small stated protocol from one or two training prompts, and the
only way to guarantee the protocol is not already in the weights is to make it
up: the verbs (grab/fold/ink/seal/file), the colors (rju/vex/mol) and the
color->drawer table exist nowhere in pretraining. Every task's prompt carries
the SAME rulebook verbatim plus one request line, so train/eval differ only in
the request (doc id, color, phrasing) — generalization is "applied the stated
rules to a new request", and memorizing the one training answer visibly fails
on requests it never saw (mol never appears in a train task).

WHAT LIVES HERE: the rulebook text and the task builders — prompt content,
which pins into the cas hash and so into run identity (I3), exactly as
dapo_math.py does. What does NOT live here: the grading state machine — that
is postprocessing (the loss's reward column) and lives INSIDE the registered
StampGrade class (training/post/stamp_grade.py), where its source hashes into
run identity through code_hashes.

Chat formatting is a SEAM (`chat`), not an import: the deploy passes the
base's real tokenizer template (thinking off, dapo_math's rule), tests pass
none and get the raw text. The formatted string is content, so two templates
are two cas uris — two experiments, as they should be.
"""

from __future__ import annotations

from collections.abc import Callable

from rlstack.data.trajectory import Task

RULEBOOK = """You operate the Stamp Office. The tray holds exactly the document named in the request. Tools:
  grab(D)    take document D from the tray. Must be D's first action.
  fold(D)    fold D. Requires grab first. Folding twice ruins D.
  ink(D, C)  apply ink color C (one of: rju, vex, mol). Requires D folded. One ink only.
  seal(D)    seal D. Requires exactly one ink. After seal, only file is legal.
  file(D, W) file D into drawer W. Requires seal. The drawer is fixed by the ink color: rju -> D1, vex -> D3, mol -> D2.
Write one tool call per line, nothing else. Stop after filing."""

COLORS = ("rju", "vex", "mol")
DRAWERS = {"rju": "D1", "vex": "D3", "mol": "D2"}

REQUESTS = (
    "Prepare document {doc} as a {color}-record.",
    "The office needs document {doc} filed as a {color}-record.",
    "Process document {doc}: it is to become a {color}-record.",
)

# The two training requests — and the deliberate hole: mol NEVER appears in a
# train task, so a policy that memorized the training answers has nothing to
# say when eval demands mol, while one that read the rulebook does.
TRAIN = ((47, "vex"), (12, "rju"))
EVAL_DOCS = (23, 31, 58, 64, 76, 89, 15, 92)   # none of them a train doc

Chat = Callable[[str], str] | None


def _prompt(doc: int, color: str, phrasing: int, chat: Chat) -> str:
    text = f"{RULEBOOK}\n\n{REQUESTS[phrasing].format(doc=doc, color=color)}"
    return chat(text) if chat is not None else text


def _task(doc: int, color: str, phrasing: int, chat: Chat) -> Task:
    return Task(id=f"stamp-office/{doc}-{color}-p{phrasing}",
                prompt=_prompt(doc, color, phrasing, chat),
                meta={"doc": doc, "color": color, "drawer": DRAWERS[color]})


def stamp_train_tasks(chat: Chat = None) -> list[Task]:
    """The one-or-two-examples training set: TRAIN's requests, first phrasing."""
    return [_task(doc, color, 0, chat) for doc, color in TRAIN]


def stamp_eval_tasks(chat: Chat = None) -> list[Task]:
    """The generalization sweep: every eval doc x every color x every
    phrasing — 72 requests the training set never showed, mol included."""
    return [_task(doc, color, phrasing, chat)
            for doc in EVAL_DOCS
            for color in COLORS
            for phrasing in range(len(REQUESTS))]
