"""Bridge between LOMA and any number of MCP servers, via FastMCP.

FastMCP's client is asynchronous; LOMA is a plain synchronous pipeline. Rather
than colour the whole codebase ``async``, this module owns one private event
loop on a background thread and submits coroutines to it. Startup, tool
listing and tool calls all go through that loop.

The important property for a small model: **tool schemas never enter the LLM
prompt wholesale.** The hub hands the router a name + one-line description per
tool. Only the single chosen tool's schema is ever shown to the model, and
only to extract its arguments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class MCPTool:
    """A tool as the router and the LLM see it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server: str = ""

    @property
    def short_description(self) -> str:
        """One line, trimmed — this is what the tool router sees."""
        text = " ".join((self.description or "").split())
        return text[:220]

    @property
    def required(self) -> list[str]:
        return list(self.input_schema.get("required") or [])

    @property
    def properties(self) -> dict[str, Any]:
        return dict(self.input_schema.get("properties") or {})


@dataclass(slots=True)
class ToolCall:
    """One executed tool call."""

    tool: MCPTool
    arguments: dict[str, Any]
    result: str
    latency_s: float = 0.0
    error: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_prompt(self) -> str:
        """The one-shot system note the LLM sees. Deliberately small."""
        return ToolBatch(calls=[self]).as_prompt()


@dataclass(slots=True)
class ToolBatch:
    """Every tool call made for one utterance, and the note the LLM sees."""

    calls: list[ToolCall] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [call.tool.name for call in self.calls]

    @property
    def failed(self) -> bool:
        return any(call.error for call in self.calls)

    @property
    def latency_s(self) -> float:
        return sum(call.latency_s for call in self.calls)

    def as_prompt(self) -> str:
        """One flat system note covering every call in the batch.

        Deliberately not persisted: tool payloads must never accumulate in the
        context window. Deliberately *flat*: a 0.8B model reads
        ``roll_die(sides=20) returned: roll=2`` correctly where a nested JSON
        array made it invent a different number.
        """
        if not self.calls:
            return ""

        lines: list[str] = []
        for call in self.calls:
            arguments = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
            lines.append(
                f"- {call.tool.name}({arguments}) returned: {_readable(call.result)}"
            )

        # Facts, not process. The old header read "These tools just ran…", which
        # a 0.8B answered by narrating the tooling back ("The user asked what
        # time it is. The tool returned: …") instead of answering the question.
        # "Supersede anything said earlier" still matters: without it a small
        # model repeats its previous answer when the same question is asked twice.
        header = (
            "Here is what you already know for this reply. It supersedes anything "
            "said earlier, and it is current. Use it directly; do not mention "
            "tools, lookups or how you found it."
            if not self.failed
            else "A lookup failed. Say so in one short sentence."
        )
        return header + "\n" + "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "calls": [
                {
                    "tool": call.tool.name,
                    "arguments": call.arguments,
                    "result": call.result,
                    "error": call.error,
                    "latency_s": round(call.latency_s, 4),
                }
                for call in self.calls
            ]
        }


# ---------------------------------------------------------------------------
# Event loop plumbing
# ---------------------------------------------------------------------------


class _AsyncRunner:
    """A dedicated asyncio loop on a daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="loma-mcp-loop", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

        # run_forever() has returned: cancel anything still in flight so that
        # no future tries to resolve itself against a closed loop later.
        try:
            pending = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        except Exception as exc:  # noqa: BLE001 - shutdown must never raise
            log.debug("event loop drain: %s", exc)

    def run(self, coro, timeout: float = 60.0):
        if self._loop.is_closed():
            raise RuntimeError("MCP event loop is already closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError("MCP operation timed out") from None

    def stop(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_closed():
            try:
                self._loop.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("event loop close: %s", exc)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_mcp_config(path: Path) -> dict[str, Any] | None:
    """Read a Claude-Desktop-style ``mcp.json`` and expand LOMA placeholders.

    Supported placeholders inside ``command``, ``args`` and ``env`` values:

    * ``${LOMA_ROOT}``   — the repository root
    * ``${LOMA_PYTHON}`` — the interpreter running LOMA (the venv)
    """
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None

    servers = raw.get("mcpServers") or raw.get("servers")
    if not isinstance(servers, dict) or not servers:
        log.warning("%s has no mcpServers block", path)
        return None

    replacements = {
        "${LOMA_ROOT}": str(_PROJECT_ROOT),
        "${LOMA_PYTHON}": sys.executable,
    }

    def expand(value):
        if isinstance(value, str):
            for needle, replacement in replacements.items():
                value = value.replace(needle, replacement)
            return os.path.expanduser(value)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    return {"mcpServers": expand(servers)}


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------


class MCPHub:
    """Owns the FastMCP client and exposes a synchronous, failure-tolerant API."""

    def __init__(self, config: dict[str, Any] | None, *, max_tools: int = 40) -> None:
        self._config = config
        self.max_tools = max_tools
        self._runner: _AsyncRunner | None = None
        self._client = None
        self._tools: list[MCPTool] = []
        self._started = False
        self.last_error = ""

    # -- lifecycle -----------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self._config)

    @property
    def available(self) -> bool:
        return self._started and bool(self._tools)

    def start(self) -> bool:
        """Connect to every configured server. Returns True on success."""
        if self._started:
            return self.available
        if not self._config:
            return False
        try:
            from fastmcp import Client
        except ImportError:
            self.last_error = "fastmcp is not installed (pip install fastmcp)"
            log.warning(self.last_error)
            return False

        self._runner = _AsyncRunner()
        self._client = Client(self._config)
        try:
            raw_tools = self._runner.run(self._connect(), timeout=60)
        except Exception as exc:  # noqa: BLE001 - a bad server must not kill LOMA
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("MCP startup failed: %s", self.last_error)
            self._shutdown_runner()
            return False

        self._tools = [self._convert(tool) for tool in raw_tools]
        if len(self._tools) > self.max_tools:
            log.warning(
                "MCP exposes %d tools; the router only sees the first %d. "
                "Split servers or raise LOMA_TOOL_MAX_TOOLS.",
                len(self._tools),
                self.max_tools,
            )
            self._tools = self._tools[: self.max_tools]

        self._started = True
        log.info("MCP ready: %d tools from %d server(s)", len(self._tools), len(self._config["mcpServers"]))
        return self.available

    async def _connect(self):
        await self._client.__aenter__()
        return await self._client.list_tools()

    def stop(self) -> None:
        if self._client is not None and self._runner is not None and self._started:
            try:
                self._runner.run(self._client.__aexit__(None, None, None), timeout=10)
            except Exception as exc:  # noqa: BLE001
                log.debug("MCP shutdown: %s", exc)
        self._shutdown_runner()
        self._started = False

    def _shutdown_runner(self) -> None:
        if self._runner is not None:
            self._runner.stop()
            self._runner = None

    def __enter__(self) -> MCPHub:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- tools ---------------------------------------------------------
    @property
    def tools(self) -> list[MCPTool]:
        return list(self._tools)

    def find(self, name: str) -> MCPTool | None:
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    def call(self, name: str, arguments: dict[str, Any], *, timeout: float = 30.0) -> str:
        """Execute a tool and return a compact string result."""
        if not self._started or self._runner is None or self._client is None:
            return "error: MCP is not connected"
        try:
            result = self._runner.run(
                self._client.call_tool(name, arguments), timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("tool %s failed: %s", name, exc)
            return f"error: {type(exc).__name__}: {exc}"
        return _result_text(result)

    # -- internals -----------------------------------------------------
    @staticmethod
    def _convert(tool) -> MCPTool:
        # FastMCP 4 exposes ``input_schema``; older builds use ``inputSchema``.
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", {}) or {}
        return MCPTool(
            name=tool.name,
            description=tool.description or "",
            input_schema=schema,
            server=getattr(tool, "server", "") or "",
        )


def _readable(result: str) -> str:
    """Flatten a tool's JSON result into one plain line for the model."""
    text = result.strip()
    try:
        data = json.loads(text)
    except ValueError:
        return text

    if isinstance(data, dict):
        return "; ".join(f"{key}={_scalar(value)}" for key, value in data.items())
    if isinstance(data, list):
        return ", ".join(_scalar(value) for value in data)
    return _scalar(data)


def _scalar(value) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _result_text(result) -> str:
    """Flatten a CallToolResult into text the LLM can read."""
    if getattr(result, "is_error", False):
        parts = [getattr(block, "text", "") for block in getattr(result, "content", [])]
        return "error: " + (" ".join(p for p in parts if p) or "tool reported an error")

    blocks = [
        getattr(block, "text", "")
        for block in getattr(result, "content", [])
        if getattr(block, "text", None)
    ]
    if blocks:
        return "\n".join(blocks)

    data = getattr(result, "data", None)
    if data is not None:
        try:
            return json.dumps(data, ensure_ascii=False, default=str)
        except TypeError:
            return str(data)

    structured = getattr(result, "structured_content", None)
    if structured:
        return json.dumps(structured, ensure_ascii=False, default=str)
    return "(tool returned no output)"
