"""Whether something the model claims is actually supported by the turn.

Two failure modes were observed with a real model, both with a schema already in
place — a schema constrains *shape*, not honesty:

1. "tell me a long story about a lighthouse" was stored as "The user lives in a
   coastal town where they work as a local guide". The model invented a person.
2. "what is my name?" was stored as "What is your name?" and "what do you know
   about me?" as "You asked what I know about you" — accurate quotes, useless
   facts.

So: a candidate must quote the user verbatim, and the quote must be a statement
rather than a question or an instruction.

These live in ``loma`` rather than the server so they can be tested without a web
framework installed.
"""

from __future__ import annotations

import re

_QUESTION_WORD = re.compile(
    r"^(what|why|how|when|where|who|which|do|does|did|can|could|would|should|"
    r"is|are|am|was|were|tell|show|give)\b",
    re.IGNORECASE,
)
#: Meta-sentences about the conversation are not facts about the person.
_META = re.compile(
    r"\b(you asked|user asked|you said|you want to know|you are asking|"
    r"you were asking)\b",
    re.IGNORECASE,
)


def looks_like_fact(fact: str, quote: str) -> bool:
    """Reject questions and meta-sentences dressed up as remembered facts.

    Verbatim evidence stops a fact being invented. It does not stop the model
    storing the user's *question* as though it were a statement — "What is your
    name?" and "You asked what I know about you" both arrived with perfectly
    accurate quotes attached.
    """
    text = fact.strip()
    if not text:
        return False
    if "?" in text or _QUESTION_WORD.match(text) or _META.search(text):
        return False
    # If the sentence the model quoted was itself a question, it is not a fact.
    return not quote.strip().endswith("?")


def quote_appears(quote: str, text: str) -> bool:
    """Is the model's evidence really in what the user said?

    Compared on collapsed, lower-cased text so that punctuation and line breaks
    cannot fail an honest quote — only an invented one.
    """
    def normalise(value: str) -> str:
        return " ".join(value.lower().split())

    needle = normalise(quote)
    # An empty quote matches everything; refuse rather than pass by accident.
    return bool(needle) and needle in normalise(text)

MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "worth_remembering": {"type": "boolean"},
        "fact": {"type": "string"},
        # The model must show its evidence. A quote that is not actually in the
        # user's message means the "fact" was invented, and is thrown away —
        # which is what stops a story request from becoming a biography.
        "quote": {"type": "string"},
    },
    "required": ["worth_remembering", "fact", "quote"],
}

MEMORY_PROMPT = (
    "You decide whether the user stated a durable fact about themselves that is "
    "worth remembering in a later conversation: their name, where they live, what "
    "they do, a preference, a project, a constraint.\n\n"
    "Only facts the user stated about themselves, in their own words. Never "
    "infer from a story, a hypothetical, an example, or a request — asking for a "
    "story about a lighthouse says nothing about the user.\n\n"
    "Reply with JSON only. `quote` must be copied verbatim from the user's "
    "message: the exact words that support the fact. If you cannot quote it, set "
    "worth_remembering to false."
)
