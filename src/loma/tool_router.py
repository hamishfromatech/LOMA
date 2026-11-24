"""Triage with Needle 3.

Needle is a 29–121M parameter model (an 8–29 MB binary, 35 MB of weights) built
for on-device tool use. In LOMA it is a **classifier, not a caller**: given a
toolset and a sentence, decide whether the turn needs tools at all and propose
candidate tool names for a shortlist.

That division matters now that the language model uses Ollama's native tool
calling. The LLM reads real schemas, fills the arguments and decides whether to
call anything; Needle's job is to keep a plain chat turn from paying for the
toolset, and to keep a forty-tool prompt from being dumped into every request.

The old grammar-constrained argument filling is still available as
``route()`` — it is a diagnostic for tool descriptions (`loma tools --probe`),
not part of the live turn.
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


#: Conjunctions that separate independent actions. Deliberately narrow: this
#: is only tried when the whole utterance failed, so a missed split costs a
#: refusal, not a wrong call.
_CLAUSE_SPLIT = re.compile(r"\s*(?:,\s*)?\b(?:and then|then|and)\b\s*", re.IGNORECASE)


def split_clauses(text: str) -> list[str]:
    """Split a compound request into actionable clauses, keeping them whole.

    "roll a d20, then tell me what 17 times 24 is" -> two clauses. A clause that
    is too short to carry an action ("and 3") is dropped rather than routed.
    """
    parts = [part.strip(" ,.;") for part in _CLAUSE_SPLIT.split(text)]
    return [part for part in parts if len(part.split()) >= 2]


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


@dataclass(slots=True)
class ToolTriage:
    """Needle's one job now: should this turn offer tools at all?

    It is a classifier, not a caller. It proposes tool *names* so the turn can
    be shortlisted, but the language model fills the arguments and decides
    whether to call anything. A missed tool is expensive, so the gate is
    recall-oriented: any plausible candidate lets the schemas through.
    """

    needs_tools: bool = False
    candidates: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    latency_s: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "needs_tools": self.needs_tools,
            "candidates": list(self.candidates),
            "confidence": round(self.confidence, 4),
            "latency_s": round(self.latency_s, 4),
            "reasoning": self.reasoning,
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
        """Diagnostic route with arguments, for `loma tools --probe`.

        The live pipeline no longer executes this: the language model calls the
        tools. It remains useful for measuring whether a toolset is well
        described, and for the legacy confidence floor.
        """
        text = text.strip()
        started = time.perf_counter()
        if not text or self._agent is None:
            return ToolRoute(
                latency_s=time.perf_counter() - started,
                reason="router not loaded",
            )
        route = self._route_with_fallback(
            text, floor=self.settings.tool_min_confidence
        )
        route.latency_s = time.perf_counter() - started
        return route

    def classify(self, text: str) -> ToolTriage:
        """Triage: does this turn need tools, and roughly which ones?

        Never raises. The floor is deliberately lower than the execution floor
        used above: the LLM is the arbiter now, and this exists to skip the
        obvious non-tool turns and to shortlist when Needle is confident.
        """
        started = time.perf_counter()
        text = text.strip()
        if not text or self._agent is None:
            return ToolTriage(
                latency_s=time.perf_counter() - started,
                reason="router not loaded",
            )

        route = self._route_with_fallback(
            text, floor=self.settings.triage_min_confidence
        )
        candidates: list[str] = []
        for request in route.requests:
            if request.tool.name not in candidates:
                candidates.append(request.tool.name)

        # A suppressed call is a strong recall signal: Needle recognised the
        # action but could not fill a required argument ("roll a die" and a
        # `sides` parameter). The schema must reach the model so it can fill it.
        needs_tools = bool(candidates) or route.suppressed > 0
        return ToolTriage(
            needs_tools=needs_tools,
            candidates=candidates,
            confidence=route.confidence,
            reasoning=route.reasoning,
            latency_s=time.perf_counter() - started,
            reason=route.reason,
        )

    def _route_with_fallback(self, text: str, *, floor: float) -> ToolRoute:
        """Route the whole utterance, then each clause if nothing cleared.

        A compound request ("roll a d20, then tell me what 17 times 24 is")
        often scores below the floor as one sentence and cleanly as two. The
        whole-utterance result wins whenever it exists, so a working single
        request is never split.
        """
        whole = self._route_once(text, floor=floor)
        clauses = split_clauses(text)
        if whole.has_calls or len(clauses) < 2:
            return whole

        merged: list[ToolRequest] = []
        confidences: list[float] = []
        reasons: list[str] = []
        for clause in clauses:
            routed = self._route_once(clause, floor=floor)
            if routed.has_calls:
                merged.extend(routed.requests)
                confidences.append(routed.confidence)
                reasons.append(routed.reasoning)
            elif routed.reasoning:
                reasons.append(routed.reasoning)

        if not merged:
            return whole

        log.info(
            "compound request: %d clause(s) routed (%s)",
            len(confidences),
            ", ".join(request.tool.name for request in merged),
        )
        return ToolRoute(
            requests=merged,
            confidence=min(confidences),
            reasoning="; ".join(reason for reason in reasons if reason),
            refused=False,
        )

    def _route_once(self, text: str, *, floor: float | None = None) -> ToolRoute:
        """One Needle call, with its own confidence and argument validation."""
        started = time.perf_counter()
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

        floor = self.settings.tool_min_confidence if floor is None else floor
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
                log.warning("Needle returned unknown tool %s", name)
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
