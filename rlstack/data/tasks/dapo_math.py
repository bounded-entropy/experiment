"""DAPO-Math-17k as a task set: one dataset, one function.

    BytedTsinghua-SIA/DAPO-Math-17k   (verl format, one parquet)

THE FILE AS IT ACTUALLY IS, inspected rather than remembered (2026-08-28):

    data_source   string  — "math_dapo" throughout
    prompt        list<struct<content, role>> — ONE user message in every row
    ability       string  — "MATH" throughout
    reward_model  struct<ground_truth: string, style: string>
    extra_info    struct<index: string>  — a uuid, the dataset's own row id

1,791,700 rows carrying 17,917 distinct `extra_info.index` values, each
repeated EXACTLY 100 times: verl's rollout fan-out baked into the file. We
keep ONE row per index — a task is a problem, and how many samples it gets is
the plan's business, not the file's. The index is a sound identity: no index
in the file carries two different (prompt, ground_truth) payloads. Two
residues are left standing deliberately: 17,398 distinct prompt texts means
~500 problems appear under two uuids, and 7 texts appear with two different
ground truths — collapsing those would need a rule for which answer wins, and
there isn't an honest one.

Every ground_truth is a plain integer (1,791,700 of 1,791,700), which is the
dataset's construction and exactly what `verifier` depends on: it reads the
LAST integer of a completion and compares it to `str(task.meta["answer"])`.
A row whose answer is not a plain integer is DROPPED here rather than trusted
downstream, and the answer is stored as an int so "007" cannot mean a
different string than the model's "7".

THE PROMPT IS BUILT HERE because it is the interface between task content and
the environment, and building it here is what pins it into the cas hash and so
into run identity (I3). Two obligations, both discharged below:

  * IT MUST END IN A FORM THE VERIFIER READS. DAPO's own user text already
    does — it asks for a last line `Answer: $Answer` — so it is kept VERBATIM.
    A second instruction of ours on top would only compete with it.
  * IT MUST BE CHAT-FORMATTED. VllmEngine's stated v0 choice is that a prompt
    is the RAW token concatenation of its messages, no chat template applied
    (the ids flatten reproduces, no template drift), so the template text has
    to be part of the content — applied here, once, at build time.

THINKING MODE: OFF, deliberately. Qwen3's template thinks by DEFAULT (both no
kwarg and enable_thinking=True render nothing after the assistant header);
`enable_thinking=False` appends `<think>\\n\\n</think>\\n\\n`, which is what we
pin. The consequence is length: the model answers directly in the low hundreds
of tokens instead of reasoning for thousands, so a wave's rollouts fit a sane
`SamplingSpec.max_tokens` and `microbatch_tokens` budget, and the closed think
block costs PROMPT tokens (cheap, prefill) rather than completion tokens
(expensive, decode). The choice is content, not a knob: a thinking-mode set is
a different cas uri, and the two can be compared by pinning one or the other.
"""

from __future__ import annotations

import re
from pathlib import Path

from rlstack.data.trajectory import Task

DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
CHAT_BASE = "Qwen/Qwen3-14B"
ANSWER = "answer"          # the key `verifier` reads


def dapo_math_tasks(base: str = CHAT_BASE) -> list[Task]:
    """The dataset, downloaded and turned into Tasks — deduped, filtered, and
    chat-formatted for `base` with thinking off (see the module docstring).

    Ids are `dapo-math-17k/<index>`: the dataset's own uuid, prefixed for
    provenance. They are STABLE across dataset versions, which is what
    `split_tasks` needs to keep a held-out task held out when the source grows.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    snapshot = snapshot_download(DATASET, repo_type="dataset")
    tasks: dict[str, Task] = {}
    for file in sorted(Path(snapshot).rglob("*.parquet")):
        for row in pq.read_table(file).to_pylist():
            task_id = f"dapo-math-17k/{row['extra_info']['index']}"
            answer = row["reward_model"]["ground_truth"].strip()
            if task_id in tasks or not _is_integer(answer):
                continue
            messages = [{"role": m["role"], "content": m["content"]}
                        for m in row["prompt"]]
            tasks[task_id] = Task(
                id=task_id,
                prompt=tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False),
                meta={ANSWER: int(answer)},
            )
    return list(tasks.values())


def _is_integer(answer: str) -> bool:
    """A plain integer and nothing else, by the VERIFIER'S OWN pattern —
    `verifier` compares the last `-?\\d+` of a completion against
    `str(meta["answer"])`, so an answer outside that shape could never be
    scored correct and is not worth sampling against."""
    return re.fullmatch(r"-?\d+", answer) is not None
