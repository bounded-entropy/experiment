"""GSM-Symbolic (apple/GSM-Symbolic, `main` config) as task sets.

THE DATASET AS IT ACTUALLY IS, inspected via the datasets-server (2026-09-01):
100 templates (`id` 0..99), each a parameterized GSM8K problem, x 50
`instance`s whose names, entities and numbers vary while the SOLUTION
PROCEDURE is fixed. Fields per row: question, answer (worked text whose last
line is `#### N`), id, instance, original_id, original_question,
original_answer, canary.

A TEMPLATE IS A TASK FAMILY. Training on one or two instances of one family
and evaluating on its other instances (near transfer) plus other families
(far transfer) is the campaign's whole design: per-instance answers differ,
so memorizing the train string scores zero — only the procedure generalizes.

THE PROMPT IS BUILT HERE (dapo_math's rule: a prompt is content and hashes
into identity). The question is kept verbatim and ONE instruction line is
appended asking for a final `Answer: N` line — the marker `final_answer`
already reads — then chat-formatted by the injected formatter (thinking OFF,
so the visible completion IS the worked reasoning). Golds outside plain
integers are DROPPED rather than trusted downstream, dapo_math's rule again.

Fetching rides the datasets-server REST rows API (urllib, stdlib-only,
paginated) so the fakes suite can shape rows without network: `gsm_tasks` and
the set builders are pure over `rows`.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable, Sequence

from rlstack.data.trajectory import Task

DATASET = "apple/GSM-Symbolic"
ROWS_API = ("https://datasets-server.huggingface.co/rows"
            "?dataset=apple%2FGSM-Symbolic&config={config}&split=test"
            "&offset={offset}&length=100")
ASK = ("Show your reasoning step by step, then state the final answer on "
       "its own last line as: Answer: N")
ANSWER = "answer"                       # the key `final_answer` reads
GOLD = re.compile(r"####\s*(-?[\d,]+)\s*$")

Chat = Callable[[str], str] | None

# The template's turn delimiters, stashed per task so the reflect maker and
# reflect_retry environment can APPEND turns in the same chat shape the
# prompt was built with (they have no tokenizer).
QWEN_CHAT = {"assistant_end": "<|im_end|>\n",
             "user_open": "<|im_start|>user\n",
             "user_close": "<|im_end|>\n",
             "assistant_open": ("<|im_start|>assistant\n"
                                "<think>\n\n</think>\n\n")}


def fetch_rows(config: str = "main") -> list[dict]:
    """The dataset, page by page — IO only, no shaping. 5,000 rows for
    `main`; the harder `p1`/`p2` twins fetch the same way."""
    rows: list[dict] = []
    offset = 0
    while True:
        with urllib.request.urlopen(
                ROWS_API.format(config=config, offset=offset), timeout=60) as r:
            page = json.loads(r.read().decode())
        got = [row["row"] for row in page.get("rows", [])]
        if not got:
            return rows
        rows.extend(got)
        offset += len(got)
        if offset >= int(page.get("num_rows_total", offset)):
            return rows


def gold_of(row: dict) -> int | None:
    """The `#### N` gold as an int, or None where the answer is not a plain
    integer (dropped, never trusted downstream)."""
    found = GOLD.search(row["answer"].strip())
    if found is None:
        return None
    try:
        return int(found.group(1).replace(",", ""))
    except ValueError:
        return None


def _task(row: dict, chat: Chat) -> Task | None:
    gold = gold_of(row)
    if gold is None:
        return None
    family, instance = int(row["id"]), int(row["instance"])
    text = f"{row['question'].strip()}\n\n{ASK}"
    meta: dict = {ANSWER: gold, "family": family}
    if chat is not None:
        meta["chat"] = dict(QWEN_CHAT)
    return Task(id=f"gsm-symbolic/t{family:02d}-i{instance:02d}",
                prompt=chat(text) if chat is not None else text,
                meta=meta)


def gsm_tasks(rows: Sequence[dict], chat: Chat = None) -> dict[tuple, Task]:
    """{(family, instance): Task} over every integer-gold row."""
    out: dict[tuple, Task] = {}
    for row in rows:
        task = _task(row, chat)
        if task is not None:
            out[(int(row["id"]), int(row["instance"]))] = task
    return out


def gsm_train_tasks(rows: Sequence[dict], family: int,
                    instances: Sequence[int], chat: Chat = None) -> list[Task]:
    """The one-or-two-examples training set: named instances of ONE family."""
    tasks = gsm_tasks(rows, chat)
    return [tasks[(family, i)] for i in instances]


def gsm_eval_tasks(rows: Sequence[dict], train_family: int,
                   train_instances: Sequence[int], families: Sequence[int],
                   near: int = 10, far: int = 4,
                   chat: Chat = None) -> list[Task]:
    """The generalization sweep: `near` held-out instances of the train
    family, plus `far` instances of every other chosen family — every task
    disjoint from the training set."""
    tasks = gsm_tasks(rows, chat)
    held = [tasks[(train_family, i)] for i in sorted(
        i for f, i in tasks if f == train_family and i not in train_instances
    )[:near]]
    for family in families:
        if family == train_family:
            continue
        held += [tasks[(family, i)] for i in sorted(
            i for f, i in tasks if f == family)[:far]]
    return held
