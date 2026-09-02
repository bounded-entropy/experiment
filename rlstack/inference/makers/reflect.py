"""The reflect maker: the previous attempt, handed back with one question.

The iterative-SDPO loop's content derivation: the derived prompt is the
source task's OWN prompt (rulebook and request included, chat formatting
included), the model's latest attempt shown back verbatim, and the ask —
state what went wrong. The retry request is NOT here: it is the reflect
environment's second turn, because asking is interaction and interaction is
an environment's, while assembling sealed text is a maker's.

CHAT SHAPE RIDES IN META. The source prompt was chat-formatted at task-build
time, so appended turns must speak the same template — and the maker has no
tokenizer. The task builders therefore stash the template's turn delimiters
in `meta["chat"]` (assistant_end / user_open / user_close / assistant_open),
and this maker — like the reflect environment — assembles with those pieces,
falling back to plain-text delimiters when a set was built unformatted (the
fakes suite). Metadata passes through untouched plus a `reflection` counter,
so the graders judge a retry exactly as they judge a first attempt, and a
maker can be pointed at its own output for iteration three.
"""

from __future__ import annotations

from rlstack.data.trajectory import Task, Trajectory
from rlstack.inference.makers.base import TaskMaker, task_maker

ASK = ("Look at your attempt above. State briefly what, if anything, "
       "was wrong with it.")


def chat_pieces(task: Task) -> dict[str, str]:
    """The template delimiters this task's set was built with — meta["chat"],
    or the plain-text fallback for unformatted sets."""
    pieces = dict(task.meta.get("chat") or {})
    pieces.setdefault("assistant_end", "\n")
    pieces.setdefault("user_open", "\nUser: ")
    pieces.setdefault("user_close", "\n")
    pieces.setdefault("assistant_open", "Assistant: ")
    return pieces


@task_maker("reflect")
class Reflect(TaskMaker):
    def make(self, source: Trajectory) -> Task:
        # the WHOLE transcript so far — prompt, attempts, critiques, asks —
        # exactly as the model saw and wrote it, so iteration n+1's context
        # is iteration n's conversation continued, never a paraphrase of it
        pieces = chat_pieces(source.task)
        iteration = int(source.task.meta.get("reflection", 0)) + 1
        prompt = (f"{source.text}{pieces['assistant_end']}"
                  f"{pieces['user_open']}{ASK}{pieces['user_close']}"
                  f"{pieces['assistant_open']}")
        meta = dict(source.task.meta)
        meta["reflection"] = iteration
        base_id = source.task.id.split("~r")[0]
        return Task(id=f"{base_id}~r{iteration}", prompt=prompt, meta=meta)
