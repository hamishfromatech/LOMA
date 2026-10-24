"""A tiny, fully offline MCP server that ships with LOMA.

It exists so students can watch the tool router turn a spoken request into
a real call without signing up for anything or opening a port:

    .venv/bin/python examples/mcp_local_tools.py

LOMA launches it automatically over stdio when `mcp.json` is present.
"""

from __future__ import annotations

import ast
import operator
import platform
import random
from datetime import datetime, timezone

from fastmcp import FastMCP

mcp = FastMCP(
    name="LOMA Local Tools",
    instructions="Small offline utilities: time, dice, arithmetic, system stats.",
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
def get_current_time(utc_offset_hours: int = 0) -> str:
    """Get the current time and date.

    Use for questions like "what time is it", "what is today's date", or
    "what day is it".

    Args:
        utc_offset_hours: offset from UTC in hours; leave at 0 for local time
    """
    now = datetime.now(timezone.utc).astimezone()
    if utc_offset_hours:
        from datetime import timedelta

        now = now.astimezone(timezone(timedelta(hours=utc_offset_hours)))
    return now.strftime("%A, %d %B %Y at %H:%M")


@mcp.tool
def roll_die(sides: int = 6) -> dict:
    """Roll a single die and return the result.

    Use when the user asks to roll a die, roll dice, roll a d20, or roll a
    twenty-sided die. For several dice, call it once per die.

    Args:
        sides: how many sides the die has (a d20 or twenty-sided die is 20,
            an ordinary die is 6)
    """
    sides = max(2, min(1000, int(sides)))
    value = random.randint(1, sides)
    return {"roll": value, "sides": sides}


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("unsupported expression")


@mcp.tool
def calculate(expression: str) -> dict:
    """Evaluate an arithmetic expression.

    Use for any arithmetic the user asks for, such as "17 times 24" or
    "what is 12 percent of 80".

    Args:
        expression: the arithmetic to evaluate, written in numbers and
            operators, for example "17 * 24" or "0.12 * 80"
    """
    tree = ast.parse(expression, mode="eval")
    value = _evaluate(tree)
    # Speak-friendly: 408, not 408.0
    result: float | int = int(value) if float(value).is_integer() else round(value, 6)
    return {"expression": expression, "result": result}


@mcp.tool
def system_stats() -> dict:
    """Report CPU count, memory in use, and the operating system.

    Use when the user asks about the computer's CPU, memory, RAM, or OS.
    """
    import psutil

    memory = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu_cores": psutil.cpu_count(logical=True),
        "memory_used_percent": round(memory.percent, 1),
        "memory_available_mb": round(memory.available / 1_000_000),
    }


if __name__ == "__main__":
    mcp.run()
