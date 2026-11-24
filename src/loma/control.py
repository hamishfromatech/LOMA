"""Control phrases: stop, reset, repeat, exit.

These four intents are the ones a voice assistant must never get wrong, and
they are also the ones a small model gets *most* wrong, because they are short,
paraphrased and ambiguous ("stop" vs "exit"). Tev1 scored 52/46 between those
two; Needle invented a `stop` for "how are you".

So they are handled deterministically: strip politeness fillers, then exact-match
against a phrase set. Zero RAM, zero latency, and completely predictable.

Anything that is not an exact control phrase falls through to the tool router
and then the language model. That is the safe direction to fail.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

#: Words that carry no command meaning and can appear anywhere in a short
#: utterance without changing what is being asked.
FILLERS: frozenset[str] = frozenset(
    {
        "a", "alright", "and", "are", "be", "can", "could", "for", "hey", "just",
        "loma", "many", "much", "now", "ok", "okay", "please", "pls", "right",
        "so", "thanks", "thank", "that", "the", "then", "there", "would", "you",
    }
)

#: intent -> phrases, after filler removal and normalisation.
#: Every entry is written in its *normalised* form, so "clear the conversation"
#: is stored as "clear conversation" because "the" is a filler.
PHRASES: dict[str, frozenset[str]] = {
    "exit": frozenset(
        {
            "bye", "good bye", "goodbye", "quit", "exit", "end session",
            "stop assistant", "shut down", "shutdown", "we are done",
            "all done", "i am done", "im done", "i'm done", "thats all",
            "that's all", "s all",
        }
    ),
    "reset": frozenset(
        {
            "reset", "reset conversation", "start over", "start again",
            "start fresh", "fresh start", "forget everything", "forget it all",
            "forget all", "clear conversation", "clear chat", "clear history",
            "new conversation", "new chat",
        }
    ),
    # Identity lookups are answered from the memory file, not generated. A 0.8B
    # asked "what is my name?" answers "I am Hamish" — it takes the user's facts
    # as its own — so the one question about who is who never reaches a model.
    # Written in normalised form: fillers ("you", "that", "the") are removed
    # before matching, so "what do you know about me" is stored as "what do know
    # about me".
    "identity": frozenset(
        {
            "what is my name", "what's my name", "whats my name",
            "do know my name", "do know me", "who am i",
            "what should call me", "what is call me",
        }
    ),
    "about": frozenset(
        {
            "what do know about me", "what do remember about me",
            "what do remember", "what have remembered",
            "tell me about me", "what do know about",
        }
    ),
    # "What tools do you have?" is a question about LOMA, not about the world,
    # and the LLM has deliberately never seen a tool schema. Answering it
    # deterministically keeps the answer true even when the router refuses.
    # Written in normalised form; run `normalise()` when adding a phrase.
    "tools": frozenset(
        {
            "what tools do have", "what tools available", "which tools do have",
            "what your tools", "list your tools", "list tools",
            "show me your tools", "show your tools", "what do",
            "what able to do", "what do do", "what your capabilities",
            "what skills do have", "what your skills", "list your skills",
            "what tools does have", "which tools available",
        }
    ),
    "repeat": frozenset(
        {
            "repeat", "repeat it", "repeat that", "say again", "say it again",
            "say that again", "come again", "once more", "one more time",
            "did say", "what did say",
        }
    ),
    "stop": frozenset(
        {
            "stop", "stop it", "stop talking", "stop speaking", "quiet",
            "be quiet", "shut up", "silence", "hush", "never mind",
            "nevermind", "enough", "s enough", "thats enough",
            "that's enough", "cancel", "cancel it", "cancel that", "hold on",
            "wait",
        }
    ),
}

#: Checked in this order: a phrase that means "end the session" must win over one
#: that only means "be quiet".
_ORDER = ("exit", "reset", "identity", "about", "tools", "repeat", "stop")

#: A capability question phrased in a way the exact phrase set does not cover
#: ("tell me about your capabilities"). Deliberately narrow: it wants an
#: interrogative/listing lead, a capability noun, and a second-person or
#: availability marker, so "what tools should I use to cut wood?" is not one.
_CAPABILITY_ASK = re.compile(
    # "tell me about your capabilities" / "what are your tools"
    r"\b(?:what|which|list|show|tell me|name)\b[^.?!]{0,40}"
    r"\b(?:you|your)\b[^.?!]{0,40}"
    r"\b(?:tools?|skills?|capabilit(?:y|ies))\b"
    # "what tools do you have" / "which tools are available"
    r"|\b(?:what|which|list|show|tell me|name)\b[^.?!]{0,40}"
    r"\b(?:tools?|skills?|capabilit(?:y|ies))\b[^.?!]{0,20}"
    r"\b(?:you|your|have|has|available|offer|support)\b"
    # "what can you do"
    r"|\bwhat (?:can|are) (?:you|u|loma)\b",
    re.IGNORECASE,
)


def is_capability_question(text: str) -> bool:
    """True for "what can you do?" even when the exact phrase set misses it."""
    return bool(_CAPABILITY_ASK.search(text))


#: "Use your tools", "your tools are…" — the user is talking about LOMA's
#: capabilities, not asking for a task. "What tools should I use to cut wood?"
#: is deliberately not this: it has no second-person possessive.
_OWN_TOOLS = re.compile(
    r"\b(?:use|using|call|invoke|try)\s+(?:your|the)\s+tools?\b"
    r"|\b(?:your|these|those)\s+tools?\b",
    re.IGNORECASE,
)


def mentions_own_tools(text: str) -> bool:
    """True when the sentence refers to LOMA's own tools."""
    return bool(_OWN_TOOLS.search(text))

#: "Remember that ..." is an open-ended command, so it is a prefix rather than a
#: phrase. Matched against the raw utterance, before filler removal, because the
#: filler list would eat the word "that".
REMEMBER_PREFIXES: tuple[str, ...] = (
    "remember that ",
    "remember ",
    "note that ",
    "make a note that ",
    "make a note ",
    "keep in mind that ",
    "keep in mind ",
    "don't forget that ",
    "dont forget that ",
    "don't forget ",
    "dont forget ",
)

_NON_WORD = re.compile(r"[^a-z0-9' ]+")
_SPACES = re.compile(r"\s+")


@dataclass(slots=True)
class ControlMatch:
    """What the control lexicon saw in one utterance."""

    intent: str = "answer"
    matched: str = ""
    latency_s: float = 0.0
    text: str = ""
    #: For ``remember``: the sentence that follows the command.
    fact: str = ""

    @property
    def is_fast_action(self) -> bool:
        return self.intent in PHRASES or self.intent == "remember"

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "matched": self.matched,
            "latency_s": round(self.latency_s, 6),
        }


def normalise(text: str) -> str:
    """Lowercase, drop punctuation, remove filler words, collapse spaces."""
    lowered = _NON_WORD.sub(" ", text.lower())
    tokens = [token for token in _SPACES.sub(" ", lowered).strip().split(" ") if token]
    kept = [token for token in tokens if token not in FILLERS]
    return " ".join(kept)


class ControlRouter:
    """Exact-match control-phrase detection."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def match(self, text: str) -> ControlMatch:
        started = time.perf_counter()
        if not self.enabled or not text.strip():
            return ControlMatch(latency_s=time.perf_counter() - started, text=text)

        stripped = text.strip()
        lowered = stripped.lower()
        for prefix in REMEMBER_PREFIXES:
            if lowered.startswith(prefix):
                fact = stripped[len(prefix) :].strip(" .,!?:;")
                if len(fact) >= 3:
                    return ControlMatch(
                        intent="remember",
                        matched=prefix.strip(),
                        fact=fact,
                        latency_s=time.perf_counter() - started,
                        text=text,
                    )

        phrase = normalise(text)
        for intent in _ORDER:
            if phrase in PHRASES[intent]:
                return ControlMatch(
                    intent=intent,
                    matched=phrase,
                    latency_s=time.perf_counter() - started,
                    text=text,
                )

        if is_capability_question(text):
            return ControlMatch(
                intent="tools",
                matched="capability question",
                latency_s=time.perf_counter() - started,
                text=text,
            )

        return ControlMatch(latency_s=time.perf_counter() - started, text=text)
