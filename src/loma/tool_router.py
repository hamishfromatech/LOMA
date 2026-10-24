"""Choose a tool with the decision model, then extract its arguments.

Why two models for one tool call?

* The **decision model** scores a short list of ``name: description`` pairs and
  answers in ~150 ms. It never sees a JSON schema, and it never generates text.
* The **language model** is only invoked after a tool has been chosen, and only
  to turn the sentence into arguments for **one** schema.

That split is the whole point: with 40 tools available, the LLM prompt grows by
one tool's schema, not by forty. Context stays small, and the router stays fast.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from loma.config import Settings
from loma.mcp_hub import MCPTool
from loma.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)

NO_TOOL = "none"

ARGUMENT_INSTRUCTIONS = (
    "You fill in the arguments for exactly one tool call. Use only what the "
    "user said. Omit optional arguments you cannot infer. Reply with JSON that "
    "matches the schema, and nothing else."
)


@dataclass(slots=True)
class ToolDecision:
    """The router's verdict about which tool (if any) to use."""

    tool: MCPTool | None = None
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_s: float = 0.0
    reason: str = ""

    @property
    def probability(self) -> float:
        if self.tool is None:
            return self.probabilities.get(NO_TOOL, 0.0)
        return self.probabilities.get(self.tool.name, 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool.name if self.tool else None,
            "confidence": round(self.confidence, 4),
            "probability": round(self.probability, 4),
            "latency_s": round(self.latency_s, 4),
            "reason": self.reason,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


class ToolRouter:
    """Decision-model tool selection plus constrained argument extraction."""

    def __init__(self, client: OllamaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    # ------------------------------------------------------------------
    def choose(self, text: str, tools: list[MCPTool]) -> ToolDecision:
        """Ask the decision model which tool fits, if any."""
        text = text.strip()
        if not text or not tools:
            return ToolDecision(reason="no tools available")

        criteria: dict[str, str] = {
            tool.name: tool.short_description or tool.name for tool in tools
        }
        criteria[NO_TOOL] = (
            "No tool is needed. They are chatting, asking for an opinion, or "
            "asking something the tools cannot do."
        )

        started = time.perf_counter()
        try:
            response = self.client.systemone(
                text,
                {
                    "tool": {
                        "type": "choice",
                        "instructions": (
                            "Which single tool should handle the speaker's request? "
                            f"Choose '{NO_TOOL}' when none of them apply."
                        ),
                        "criteria": criteria,
                    }
                },
            )
        except OllamaError as exc:
            log.warning("tool router unavailable: %s", exc)
            return ToolDecision(
                latency_s=time.perf_counter() - started,
                reason=f"router error: {exc}",
            )

        latency = time.perf_counter() - started
        answer = (response.get("answers") or {}).get("tool") or {}
        choice = str(answer.get("choice") or NO_TOOL)
        confidence = float(answer.get("confidence") or 0.0)
        probabilities = {
            str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()
        }

        decision = ToolDecision(
            tool=None,
            confidence=confidence,
            probabilities=probabilities,
            latency_s=latency,
        )

        if choice == NO_TOOL:
            decision.reason = "router chose no tool"
            return decision

        floor = self.settings.tool_min_confidence
        probability = probabilities.get(choice, 0.0)
        if confidence < floor or probability < floor:
            decision.reason = (
                f"low confidence ({confidence:.2f}/{probability:.2f} < {floor:.2f})"
            )
            return decision

        decision.tool = next((t for t in tools if t.name == choice), None)
        if decision.tool is None:
            decision.reason = f"router chose unknown tool {choice!r}"
        return decision

    # ------------------------------------------------------------------
    def arguments(self, tool: MCPTool, text: str) -> dict[str, Any] | None:
        """Extract arguments for one tool with constrained JSON generation."""
        schema = tool.input_schema or {"type": "object", "properties": {}}
        if not tool.properties:
            return {}

        prompt = [
            {"role": "system", "content": ARGUMENT_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Tool: {tool.name}\n"
                    f"Description: {tool.description}\n"
                    f"Schema: {json.dumps(schema, ensure_ascii=False)}\n"
                    f'The user said: "{text}"\n'
                    "Return the JSON arguments."
                ),
            },
        ]

        try:
            raw = self.client.chat_json(prompt, schema, max_tokens=200)
        except OllamaError as exc:
            log.warning("argument extraction failed for %s: %s", tool.name, exc)
            return None

        if not isinstance(raw, dict):
            log.warning("argument extraction returned %s, not an object", type(raw).__name__)
            return None

        # Keep only keys the schema knows about.
        allowed = set(tool.properties)
        arguments = {key: value for key, value in raw.items() if key in allowed}

        missing = [key for key in tool.required if key not in arguments]
        if missing:
            log.warning("arguments for %s missing required keys: %s", tool.name, missing)
            return None
        return arguments
