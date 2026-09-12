"""Executable correctness for Python functions and read-only SQL queries.

The same functions grade screening, training and measurement. Python runs in
a disposable, time-limited subprocess with a small allowed standard library;
SQL runs in an in-memory database with reads as the only authorized action.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group, Task
from rlstack.training.post.base import PostProcessor, postprocessor

ALLOWED_IMPORTS = frozenset({"math", "cmath", "collections", "itertools",
                             "functools", "operator", "re", "string", "heapq",
                             "bisect", "statistics", "decimal", "fractions",
                             "typing", "array", "copy", "random", "datetime"})
FORBIDDEN_NAMES = frozenset({"open", "eval", "exec", "compile", "input",
                            "getattr", "setattr", "delattr", "globals",
                            "locals", "vars", "breakpoint", "help", "exit",
                            "quit", "memoryview"})


@dataclass(frozen=True)
class ProgramScore:
    reward: float
    valid: float
    reason: str


def program_text(text: str) -> str:
    """Accept raw code or the last fenced block, removing thinking text."""
    text = text.rsplit("</think>", 1)[-1].strip()
    fences = re.findall(r"```(?:python|sql|sqlite)?\s*\n(.*?)```", text,
                        re.DOTALL | re.IGNORECASE)
    return (fences[-1] if fences else text).strip()


def python_is_allowed(code: str) -> bool:
    """Restrict benchmark code to pure computation and approved imports."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            if any(name.name.split(".")[0] not in ALLOWED_IMPORTS
                   for name in node.names):
                return False
        if isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return False
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False
    return True


def score_python(code: str, meta: Mapping) -> ProgramScore:
    """Return one only when every published unit test passes."""
    try:
        if not python_is_allowed(code):
            return ProgramScore(0.0, 0.0, "restricted")
    except (SyntaxError, ValueError):
        return ProgramScore(0.0, 0.0, "syntax")
    script = ("import resource, sys\n"
              "resource.setrlimit(resource.RLIMIT_CPU, (2, 2))\n"
              "if sys.platform == 'linux':\n"
              " resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))\n"
              "resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))\n"
              + "\n".join(meta.get("test_imports", [])) + "\n"
              + code + "\n" + "\n".join(meta["tests"]))
    with tempfile.TemporaryDirectory(prefix="rlstack-program-") as directory:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", script], cwd=directory,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=3)
        except subprocess.TimeoutExpired:
            return ProgramScore(0.0, 1.0, "timeout")
    passed = result.returncode == 0
    return ProgramScore(float(passed), 1.0, "passed" if passed else "test_failed")


def gold_sql(meta: Mapping) -> tuple[str, list]:
    """Translate WikiSQL's annotated query without interpolating cell values."""
    sql = meta["sql"]
    aggregate = ("", "MAX", "MIN", "COUNT", "SUM", "AVG")[sql["agg"]]
    column = f"c{sql['sel']}"
    selection = f"{aggregate}({column})" if aggregate else column
    conditions, values = [], []
    for column, operator, value in sql["conds"]:
        conditions.append(f"c{column} {('=', '>', '<')[operator]} ?")
        if meta["types"][column] == "real":
            try:
                value = float(str(value).replace(",", ""))
            except ValueError:
                pass
        values.append(value)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    return f"SELECT {selection} FROM data{where}", values


def sql_database(meta: Mapping) -> sqlite3.Connection:
    """Build the declared table; only SELECT/READ/FUNCTION remain authorized."""
    database = sqlite3.connect(":memory:")
    schema = ", ".join(f"c{i} " + ("REAL" if kind == "real"
                                  else "TEXT COLLATE NOCASE")
                       for i, kind in enumerate(meta["types"]))
    database.execute(f"CREATE TABLE data ({schema})")
    marks = ",".join("?" for _ in meta["types"])
    database.executemany(f"INSERT INTO data VALUES ({marks})", meta["rows"])
    database.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in
                            (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                             sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE)
                            else sqlite3.SQLITE_DENY)
    instructions = 0

    def timed_out():
        nonlocal instructions
        instructions += 1
        return instructions > 1000

    database.set_progress_handler(timed_out, 1000)
    return database


def normalized_rows(rows: list) -> Counter:
    """Compare result multisets; text case and numerical representation normalize."""
    return Counter(tuple(round(v, 6) if isinstance(v, (int, float)) else
                         v.casefold() if isinstance(v, str) else v for v in row)
                   for row in rows)


def score_sql(code: str, meta: Mapping) -> ProgramScore:
    """Execution equivalence on the original table, with a bounded read budget."""
    if not re.match(r"^(SELECT|WITH)\b", code, re.IGNORECASE):
        return ProgramScore(0.0, 0.0, "syntax")
    database = sql_database(meta)
    try:
        query, values = gold_sql(meta)
        expected = database.execute(query, values).fetchmany(10001)
        actual = database.execute(code).fetchmany(10001)
        passed = len(actual) < 10001 and normalized_rows(actual) == normalized_rows(expected)
        return ProgramScore(float(passed), 1.0, "passed" if passed else "wrong_result")
    except (sqlite3.Error, ValueError, OverflowError):
        return ProgramScore(0.0, 0.0, "sql_error")
    finally:
        database.close()


def score_program(text: str, task: Task) -> ProgramScore:
    code = program_text(text)
    if not code or len(code) > 20000:
        return ProgramScore(0.0, 0.0, "empty_or_long")
    if task.meta["domain"] == "mbpp":
        return score_python(code, task.meta)
    if task.meta["domain"] == "wikisql":
        return score_sql(code, task.meta)
    raise ValueError(f"unknown program domain: {task.meta['domain']}")


@postprocessor("program_correctness")
class ProgramCorrectness(PostProcessor):
    produces = ("reward", "program_valid", "repetition")

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        texts = [traj.turns[-1].message.content for traj in group.trajectories]
        scores = await asyncio.gather(*(
            asyncio.to_thread(score_program, text, traj.task)
            for text, traj in zip(texts, group.trajectories)))
        return {"reward": [score.reward for score in scores],
                "program_valid": [score.valid for score in scores],
                "repetition": [repeated_fourgrams(text) for text in texts]}


def repeated_fourgrams(text: str) -> float:
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)
