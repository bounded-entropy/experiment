"""The Glyph Exchange: the second invented DSL — routing on a hidden ring.

Where the Stamp Office's learnable content is a PROTOCOL (preconditions and
order), the Exchange's is a GRAPH: four invented units on a one-directional
ring (wex -> drin -> polk -> sarn -> wex), one conversion tool per edge, and a
task names a source and target unit. The model emits ONE nested expression;
the executor does all arithmetic, so the only thing being tested is whether
the policy routes the right edges in the right order — knowledge, not
capacity. Same invention guarantee as the Stamp Office: none of these units
or verbs exist in pretraining.

Prompt content lives here (rulebook + builders, pinned by cas into identity);
the parser and ring-walk live inside the registered GlyphGrade class
(training/post/glyph_grade.py), hashed by code_hashes. Train shows two
requests (wex->sarn, drin->sarn); eval sweeps every source/target pair,
including directions and lengths the training set never showed.
"""

from __future__ import annotations

from collections.abc import Callable

from rlstack.data.trajectory import Task

RULEBOOK = """You operate the Glyph Exchange. A purse holds an amount in one unit. Tools:
  load(P)     start with purse P's amount, in the purse's own unit. Must be innermost.
  to_drin(x)  accepts wex only -> drin
  to_polk(x)  accepts drin only -> polk
  to_sarn(x)  accepts polk only -> sarn
  to_wex(x)   accepts sarn only -> wex
  give(x)     hand over the amount. Must be outermost.
Answer with ONE nested expression and nothing else, like: give(to_drin(load(P1)))"""

UNITS = ("wex", "drin", "polk", "sarn")     # the ring, in edge order

REQUESTS = (
    "Purse {purse} holds {source}. Express it in {target}.",
    "Convert purse {purse} (currently {source}) into {target}.",
    "The clerk hands you purse {purse} of {source}; give back {target}.",
)

TRAIN = (("P7", "wex", "sarn"), ("P3", "drin", "sarn"))
EVAL_PURSES = ("P2", "P5", "P8")

Chat = Callable[[str], str] | None


def _prompt(purse: str, source: str, target: str, phrasing: int,
            chat: Chat) -> str:
    text = (f"{RULEBOOK}\n\n"
            f"{REQUESTS[phrasing].format(purse=purse, source=source, target=target)}")
    return chat(text) if chat is not None else text


def _task(purse: str, source: str, target: str, phrasing: int,
          chat: Chat) -> Task:
    from rlstack.data.tasks.stamp_office import QWEN_CHAT

    meta = {"purse": purse, "source": source, "target": target}
    if chat is not None:
        meta["chat"] = dict(QWEN_CHAT)     # one template, one set of pieces
    return Task(id=f"glyph-exchange/{purse}-{source}-{target}-p{phrasing}",
                prompt=_prompt(purse, source, target, phrasing, chat),
                meta=meta)


def glyph_train_tasks(chat: Chat = None) -> list[Task]:
    """The one-or-two-examples training set: TRAIN's requests, first phrasing."""
    return [_task(purse, source, target, 0, chat)
            for purse, source, target in TRAIN]


def glyph_eval_tasks(chat: Chat = None) -> list[Task]:
    """The generalization sweep: every ordered (source, target) pair with
    source != target, over eval purses and phrasings — path lengths 1 to 3,
    directions the training set never showed (sarn->wex among them)."""
    return [_task(purse, source, target, phrasing, chat)
            for purse in EVAL_PURSES
            for source in UNITS
            for target in UNITS
            if source != target
            for phrasing in range(len(REQUESTS))]
