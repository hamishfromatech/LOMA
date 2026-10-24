"""Long-term memory, stored as a plain Markdown file.

The file is the interface. You can open it, read it, edit it, delete a line, or
check it into git — LOMA just reads it at the start of every turn and appends to
it when it learns something durable.

    data/memory/MEMORY.md

Two ways a fact gets in:

* the user says "remember that …" (the ``remember`` control intent), or
* the language model is asked, after a turn, whether the user stated a durable
  personal fact, and appends it if it did (``LOMA_MEMORY_AUTO_EXTRACT``).

Memory is never part of the conversation transcript. It is injected as an extra
system note, so it cannot be compacted away and does not grow the window.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

HEADER = """# LOMA memory

Durable facts about the user, kept across conversations. Plain Markdown: edit it
by hand, delete a line, add your own. LOMA reads this file at the start of every
turn.
"""

# A bullet, optionally followed by the "_(date, source)_" provenance note.
_BULLET = re.compile(r"^\s*[-*]\s+(.+?)(?:\s+_\([^)]*\)_)?\s*$")


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    # -- raw file ------------------------------------------------------
    def ensure(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(HEADER)

    def read(self) -> str:
        self.ensure()
        try:
            return self.path.read_text()
        except OSError as exc:
            log.warning("could not read memory: %s", exc)
            return HEADER

    def write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text if text.endswith("\n") else text + "\n")

    # -- facts ---------------------------------------------------------
    def facts(self) -> list[str]:
        return [
            match.group(1).strip()
            for line in self.read().splitlines()
            if (match := _BULLET.match(line))
        ]

    def append(self, fact: str, *, source: str = "conversation") -> bool:
        """Add a fact once. Returns False when it is empty or already known."""
        fact = " ".join(fact.split()).strip(" .")
        if len(fact) < 3:
            return False

        existing = {item.lower() for item in self.facts()}
        if fact.lower() in existing:
            return False

        line = f"- {fact} _({date.today().isoformat()}, {source})_\n"
        text = self.read()
        if not text.endswith("\n"):
            text += "\n"
        self.write(text + line)
        log.info("remembered: %s", fact)
        return True

    def clear(self) -> None:
        self.write(HEADER)

    # -- prompt --------------------------------------------------------
    def as_context(self, limit: int = 40) -> str:
        """The note injected into the system prompt. Empty when there is nothing."""
        facts = self.facts()[-limit:]
        if not facts:
            return ""
        body = "\n".join(f"- {fact}" for fact in facts)
        return (
            "Facts about the person you are talking to. These are theirs, not "
            "yours — you are LOMA, and you are not them.\n"
            f"{body}\n"
            "Speak to them in the second person (\"you\"). Use a fact only where "
            "it is relevant, and do not recite the list back."
        )

    def as_markdown(self) -> str:
        return self.read()
