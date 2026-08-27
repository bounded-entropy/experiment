"""Custom panels: derived graphs as EXPRESSIONS over what a run already logs.

A panel is {"name": ..., "expr": ...} where expr is arithmetic over column
names — post columns and rails by bare name (post wins a collision), e.g.

    {"name": "gap_per_token", "expr": "logprob_gap / tokens"}
    {"name": "excess_reward", "expr": "reward - 0.5"}
    {"name": "log_grad", "expr": "log(grad_norm + 1e-9)"}

Expressions are DATA, not code: parsed with ast, only names, numbers,
+ - * / ** %, unary minus, parentheses, and the functions log/exp/sqrt/abs
are allowed — nothing else evaluates, so panels.json can never smuggle
behavior into the observer.

THE VALIDATION RULE (Samarth's): every argument must be in the pipeline —
enforced against the run's own dictionary.json (the same flow-graph oracle
the submit gate queries), NOT at submit: panels live outside identity, so
adding a graph can never fork a run. A panel whose argument a run lacks
renders its missing-list instead of a curve.

When the same expression's arguments all exist in the eval summaries, the
held-out overlay is computed with the same expression — one definition, both
series.
"""

from __future__ import annotations

import ast
import math

_FUNCTIONS = {"log": math.log, "exp": math.exp, "sqrt": math.sqrt, "abs": abs}

_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
}


class PanelError(ValueError):
    """An expression outside the allowed arithmetic grammar."""


def parse_panel(expr: str) -> ast.expression:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as err:
        raise PanelError(f"unparseable expression {expr!r}: {err.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Constant, ast.Name, ast.Load,
                             ast.UnaryOp, ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            continue
        if isinstance(node, tuple(_OPS)):
            continue
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name)
                    and node.func.id in _FUNCTIONS
                    and len(node.args) == 1 and not node.keywords):
                continue
            raise PanelError(
                f"{expr!r}: only {sorted(_FUNCTIONS)} may be called")
        raise PanelError(
            f"{expr!r}: {type(node).__name__} is not panel arithmetic")
    return tree


def panel_args(expr: str) -> frozenset[str]:
    """The column names an expression reads — what must be in the pipeline."""
    tree = parse_panel(expr)
    return frozenset(node.id for node in ast.walk(tree)
                     if isinstance(node, ast.Name)
                     and node.id not in _FUNCTIONS)


def missing_args(expr: str, dictionary: dict | None) -> tuple[str, ...]:
    """THE validation: arguments not among the run's stored post columns or
    rails, per its own dictionary.json."""
    available = set()
    for column in (dictionary or {}).get("columns", []):
        if column.get("stored") and column.get("phase") in ("post", "train"):
            available.add(column["name"])
    return tuple(sorted(panel_args(expr) - available))


def evaluate(expr: str, namespace: dict[str, float]) -> float | None:
    """One point: the expression over one update's values. None when an
    argument is absent this update or the arithmetic fails (log of zero…)."""
    tree = parse_panel(expr)

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return float(namespace[node.id])
        if isinstance(node, ast.UnaryOp):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call):
            return _FUNCTIONS[node.func.id](walk(node.args[0]))
        raise PanelError(f"unreachable node {type(node).__name__}")

    try:
        return walk(tree)
    except (KeyError, ArithmeticError, ValueError, TypeError):
        return None
