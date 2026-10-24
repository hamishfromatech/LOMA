"""Tool selection and argument filling with Needle 3.

Needle is a 29–121M parameter model (a 8–29 MB binary, 35 MB of weights) built
for exactly one job: given a toolset and a sentence, return well-formed calls
with the arguments already filled in.

Two properties matter here:

* **The arguments are grammar-constrained.** Needle compiles a byte-level
  grammar from the tool schemas, so a call cannot be malformed and a number
  cannot land in a string field.
* **It refuses.** An off-topic request returns an empty call list with a low
  confidence, instead of inventing a tool. LOMA treats an empty list as "let the
  language model answer".

That replaces both the 1.2 GB System One router and the second LLM pass that
used to extract arguments, cutting a full generation out of every tool turn.

Tool design matters more than model choice here: see
`docs/ARCHITECTURE.md#tool-design` for the rules that make Needle accurate.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loma.config import Settings
from loma.mcp_hub import MCPTool

log = logging.getLogger(__name__)

_NEEDLE_NAME = re.compile(r"[^A-Za-z0-9_]")


def tool_schema(tool: MCPTool, name: str) -> dict[str, Any]:
    """Convert an MCP tool into the raw schema Needle consumes."""
    parameters = tool.input_schema or {"type": "object", "properties": {}}
    return {
        "name": name,
        "description": tool.description or tool.name,
        "parameters": parameters,
    }


@dataclass(slots=True)
class ToolRequest:
    """One tool call the model wants to make, arguments already filled."""

    tool: MCPTool
    arguments: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool.name, "arguments": self.arguments}


@dataclass(slots=True)
class ToolRoute:
    """The router's answer for one utterance."""

    requests: list[ToolRequest] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    latency_s: float = 0.0
    refused: bool = True
    suppressed: int = 0
    reason: str = ""

    @property
    def has_calls(self) -> bool:
        return bool(self.requests)

    @property
    def names(self) -> list[str]:
        return [request.tool.name for request in self.requests]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tools": [r.as_dict() for r in self.requests],
            "confidence": round(self.confidence, 4),
            "latency_s": round(self.latency_s, 4),
            "reasoning": self.reasoning,
            "refused": self.refused,
            "suppressed": self.suppressed,
            "reason": self.reason,
        }


class NeedleRouter:
    """Owns one Needle agent for the current toolset."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._agent = None
        self._lock = threading.Lock()
        self._names: dict[str, MCPTool] = {}
        self._signature: tuple[str, ...] = ()
        self.last_error = ""

    # -- lifecycle -----------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._agent is not None

    def load(self, tools: list[MCPTool]) -> bool:
        """Build (or rebuild) the agent for ``tools``."""
        # Descriptions are the router's entire input, so they belong in the
        # signature: with names alone, editing a docstring left the old agent —
        # and its old idea of when to call the tool — in place until restart.
        signature = tuple(
            sorted(f"{tool.name}\x00{tool.description}" for tool in tools)
        )
        if self._agent is not None and signature == self._signature:
            return True
        if not tools:
            return False

        try:
            import needle
        except ImportError:
            self.last_error = "cactus-needle is not installed (pip install cactus-needle)"
            log.warning(self.last_error)
            return False

        names: dict[str, MCPTool] = {}
        schemas: list[dict[str, Any]] = []
        for tool in tools:
            safe = _NEEDLE_NAME.sub("_", tool.name)
            while safe in names:  # extremely unlikely, but do not collide
                safe = f"{safe}_"
            names[safe] = tool
            schemas.append(tool_schema(tool, safe))

        started = time.perf_counter()
        try:
            agent = needle.Needle(tools=schemas)
        except Exception as exc:  # noqa: BLE001 - a router must never be fatal
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("Needle could not start: %s", self.last_error)
            return False

        with self._lock:
            self._agent = agent
            self._names = names
            self._signature = signature
        log.info(
            "Needle ready: %d tools in %.2fs",
            len(schemas),
            time.perf_counter() - started,
        )
        return True

    def close(self) -> None:
        with self._lock:
            self._agent = None
            self._names = {}
            self._signature = ()

    # -- routing -------------------------------------------------------
    def route(self, text: str) -> ToolRoute:
        """Pick tools for ``text``. Never raises."""
        text = text.strip()
        started = time.perf_counter()
        if not text or self._agent is None:
            return ToolRoute(latency_s=time.perf_counter() - started, reason="router not loaded")

        try:
            with self._lock:
                # Each utterance is independent. Needle is multi-turn, and
                # letting state leak across unrelated requests makes it carry
                # the previous tool choice forward.
                self._agent.reset()
                raw = self._agent.complete(text)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("Needle routing failed: %s", self.last_error)
            return ToolRoute(
                latency_s=time.perf_counter() - started,
                reason=f"router error: {self.last_error}",
            )

        latency = time.perf_counter() - started
        confidence = float(raw.get("confidence") or 0.0)
        calls = raw.get("function_calls") or []
        suppressed = len(raw.get("suppressed_calls") or [])
        reasoning = str(raw.get("reasoning") or "")

        if not calls:
            return ToolRoute(
                confidence=confidence,
                reasoning=reasoning,
                latency_s=latency,
                suppressed=suppressed,
                reason="needle refused (no tool fits)",
            )

        floor = self.settings.tool_min_confidence
        if confidence < floor:
            return ToolRoute(
                confidence=confidence,
                reasoning=reasoning,
                latency_s=latency,
                suppressed=suppressed,
                reason=f"low confidence ({confidence:.2f} < {floor:.2f})",
            )

        requests: list[ToolRequest] = []
        for call in calls:
            name = str(call.get("name") or "")
            tool = self._names.get(name)
            if tool is None:
                log.warning("Needle returned unknown tool %r", name)
                continue
            arguments = dict(call.get("arguments") or {})
            arguments = self._coerce(tool, arguments)
            requests.append(ToolRequest(tool=tool, arguments=arguments))

        if not requests:
            return ToolRoute(
                confidence=confidence,
                reasoning=reasoning,
                latency_s=latency,
                suppressed=suppressed,
                reason="no usable calls after validation",
            )

        return ToolRoute(
            requests=requests,
            confidence=confidence,
            reasoning=reasoning,
            latency_s=latency,
            refused=False,
            suppressed=suppressed,
        )

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _coerce(tool: MCPTool, arguments: dict[str, Any]) -> dict[str, Any]:
        """Keep only schema keys and drop nulls for optional arguments."""
        allowed = set(tool.properties)
        cleaned = {
            key: value
            for key, value in arguments.items()
            if key in allowed and value is not None
        }
        missing = [key for key in tool.required if key not in cleaned]
        if missing:
            log.warning("arguments for %s missing required keys: %s", tool.name, missing)
        return cleaned
