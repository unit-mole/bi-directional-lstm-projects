from __future__ import annotations

import ast
import re


def _humanize(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return " ".join(part for part in name.strip("_").split("_") if part)


def identifier_baseline(code: str) -> str:
    """Small transparent baseline used for comparison, not as a neural substitute."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "Describes the supplied code snippet."
    function = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if function is None:
        return "Executes the supplied Python operations."
    purpose = _humanize(function.name)
    return_node = next((n for n in ast.walk(function) if isinstance(n, ast.Return)), None)
    if return_node and isinstance(return_node.value, ast.BinOp):
        op_names = {
            ast.Add: "sum", ast.Sub: "difference", ast.Mult: "product",
            ast.Div: "quotient", ast.FloorDiv: "floor-divided result", ast.Mod: "remainder",
        }
        op = op_names.get(type(return_node.value.op))
        if op:
            return f"Returns the {op} computed by {purpose}."
    if purpose:
        return f"Performs {purpose}."
    return "Processes the supplied inputs and returns a result."
