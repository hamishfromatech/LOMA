"""The fast path: a System One decision model as an intent router.

Before a single token of the 2B language model is generated, a 0.8B decision
model answers "what does the speaker actually want?" in roughly 150 ms. That
lets us serve control phrases like *"stop"*, *"start over"* or *"say that
again"* without touching the LLM at all — no generation, no context, no
latency.

Crucially the router is allowed to **abstain**: below a confidence floor the
turn falls through to the language model, so a misread never swallows a real
question.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from loma.config import Settings
from loma.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent catalogue
# ---------------------------------------------------------------------------

#: Every intent the router may return, with the description the decision model
#: scores against. Keep these mutually exclusive and concrete.
INTENT_CRITERIA: dict[str, str] = {
    "answer": (
        "They are asking a question, making a request, or chatting, and expect "
        "a spoken reply."
    ),
    "stop": (
        "They want the assistant to be quiet right now. Includes 'stop', "
        "'stop it', 'be quiet', 'shut up', 'never mind', 'that's enough'."
    ),
    "reset": (
        "They want the conversation erased and a fresh start. Includes 'start "
        "over', 'forget everything', 'clear the conversation', 'new conversation'."
    ),
    "repeat": (
        "They want the last answer repeated. Includes 'say that again', "
        "'repeat that'."
    ),
    "exit": (
        "They want to end the whole session. Includes 'goodbye', 'quit', "
        "'exit', 'thanks that's all'."
    ),
}

#: Intents that are handled without calling the language model.
FAST_INTENTS: frozenset[str] = frozenset({"stop", "reset", "repeat", "exit"})


@dataclass(slots=True)
class Decision:
    """The router's verdict for one utterance."""

    intent: str = "answer"
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    needs_reasoning: bool = False
    latency_s: float = 0.0
    fallback_reason: str = ""

    @property
    def is_fast_action(self) -> bool:
        return self.intent in FAST_INTENTS

    @property
    def probability(self) -> float:
        """Probability mass the model put on the winning intent."""
        return self.probabilities.get(self.intent, 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "probability": round(self.probability, 4),
            "needs_reasoning": self.needs_reasoning,
            "latency_s": round(self.latency_s, 4),
            "fallback_reason": self.fallback_reason,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


class DecisionEngine:
    """Wraps ``/v1/systemone`` and enforces the abstain-below-threshold rule."""

    def __init__(self, client: OllamaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self._enabled = settings.decision_enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def warm_up(self) -> None:
        """Load the decision weights and prove the endpoint works."""
        if not self._enabled:
            return
        self.decide("hello")

    # ------------------------------------------------------------------
    def decide(self, text: str) -> Decision:
        """Classify ``text``. Never raises: routing failure means 'answer'."""
        text = text.strip()
        if not text or not self._enabled:
            return Decision(fallback_reason="router disabled")

        questions: dict[str, dict] = {
            "intent": {
                "type": "choice",
                "instructions": "What does the speaker want the assistant to do?",
                "criteria": INTENT_CRITERIA,
            }
        }
        if self.settings.adaptive_thinking:
            questions["needs_reasoning"] = {
                "type": "noul",
                "instructions": (
                    "Does answering this accurately require deliberate multi-step "
                    "reasoning, rather than a short factual or conversational reply?"
                ),
                "criteria": {
                    "false": "A short, direct reply is enough.",
                    "true": "It needs careful, multi-step reasoning.",
                },
            }

        started = time.perf_counter()
        try:
            response = self.client.systemone(text, questions)
        except OllamaError as exc:
            log.warning("decision model unavailable, answering normally: %s", exc)
            return Decision(
                latency_s=time.perf_counter() - started,
                fallback_reason=f"router error: {exc}",
            )

        latency = time.perf_counter() - started
        return self._parse(response, latency)

    # ------------------------------------------------------------------
    def _parse(self, response: dict[str, Any], latency: float) -> Decision:
        answers = response.get("answers", {})
        intent_answer = answers.get("intent", {})
        probabilities = {
            str(k): float(v) for k, v in (intent_answer.get("probabilities") or {}).items()
        }
        confidence = float(intent_answer.get("confidence") or 0.0)
        intent = str(intent_answer.get("choice") or "answer")

        reasoning = answers.get("needs_reasoning", {})
        needs_reasoning = float(reasoning.get("noul") or 0.0) >= 0.5

        decision = Decision(
            intent=intent,
            confidence=confidence,
            probabilities=probabilities,
            needs_reasoning=needs_reasoning,
            latency_s=latency,
        )
        return self._apply_confidence_floor(decision)

    def _apply_confidence_floor(self, decision: Decision) -> Decision:
        """Abstain when the router is not sure enough to act.

        Two ways to earn a fast action:

        1. Win outright: probability and confidence are both above the floor.
        2. Win a family argument: the runner-up is *also* a control intent.
           Confidence measures separation from the runner-up, so when both
           candidates are cheap non-generative actions there is nothing
           expensive to get wrong — we take the winner at face value.

        Anything else falls through to the language model.
        """
        floor = self.settings.decision_min_confidence
        if decision.intent == "answer":
            return decision

        if decision.confidence >= floor and decision.probability >= floor:
            return decision

        runner_up = self._runner_up(decision)
        if (
            runner_up in FAST_INTENTS
            and decision.probability >= floor
        ):
            log.debug(
                "accepting %s over %s: both are fast actions",
                decision.intent,
                runner_up,
            )
            return decision

        log.debug(
            "router abstained: %s (confidence %.2f, p %.2f < %.2f)",
            decision.intent,
            decision.confidence,
            decision.probability,
            floor,
        )
        decision.fallback_reason = (
            f"low confidence ({decision.confidence:.2f} < {floor:.2f})"
        )
        decision.intent = "answer"
        return decision

    @staticmethod
    def _runner_up(decision: Decision) -> str:
        others = {
            name: probability
            for name, probability in decision.probabilities.items()
            if name != decision.intent
        }
        if not others:
            return ""
        return max(others, key=others.__getitem__)
