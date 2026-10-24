"""Small text guards applied to anything the model produces.

A reasoning model can put its scratchpad in ``message.content`` when thinking is
switched off, terminated by a literal ``</think>``. That text is not an answer:
it is meta-commentary about the answer, and LOMA speaks every sentence as it
arrives, so it would be read aloud.

The rule is deliberately blunt: if a closing marker appears, everything before it
is discarded. A model that wanted to *say* the string ``</think>`` would have to
escape it, and none do.
"""

from __future__ import annotations

CLOSE = "</think>"
OPEN = " thinking"

#: How many characters the streaming splitter may look ahead before it commits.
#: A reasoning block is usually far longer than this, so this is a guard against
#: a short leak and against a marker straddling two deltas — not a proof. The
#: structural fix is Ollama separating ``message.thinking``; see
#: :meth:`loma.ollama_client.OllamaClient.chat_stream`.
LOOKAHEAD = 240


def strip_thinking(text: str) -> str:
    """Remove a reasoning block from a complete reply.

    Handles the shapes seen in the wild:

    * ``…reasoning…</think>…answer`` — keep what follows the last marker.
    * `` thinking…reasoning…answer`` — keep what follows the opening tag.
    * ``…reasoning…`` with no closing marker (generation was cut off) — nothing
      is trustworthy, so return an empty string rather than the scratchpad.
    """
    if not text:
        return ""

    if CLOSE in text:
        text = text.rsplit(CLOSE, 1)[1]

    if OPEN in text:
        _, _, tail = text.partition(OPEN)
        if not tail.strip():
            return ""
        text = tail

    return text.lstrip("\n ")


class ThinkingSplitter:
    """Stream-safe wrapper around :func:`strip_thinking`.

    Holds at most :data:`LOOKAHEAD` characters while it decides. Once the window
    closes with no marker, it commits: everything after that streams straight
    through, so a normal answer costs one short pause and no rewrites.
    """

    def __init__(self, lookahead: int = LOOKAHEAD) -> None:
        self._held = ""
        self._lookahead = lookahead
        self._committed = False
        self._in_block = False
        self.saw_marker = False

    def push(self, delta: str) -> str:
        """Feed one delta; return the text that is safe to show and to speak."""
        if self._committed:
            return delta

        self._held += delta

        if CLOSE in self._held:
            self.saw_marker = True
            self._committed = True
            out = self._held.rsplit(CLOSE, 1)[1]
            self._held = ""
            return out

        if self._in_block or OPEN in self._held:
            # Inside an unclosed block: hold everything, emit nothing, however
            # long it runs. This is the case the lookahead must never break.
            self._in_block = True
            self._held = self._held.partition(OPEN)[2]
            return ""

        if len(self._held) >= self._lookahead or self._held.rstrip().endswith(
            (".", "!", "?", "\n")
        ):
            self._committed = True
            out, self._held = self._held, ""
            return out

        return ""

    def flush(self) -> str:
        """Whatever is still held when the stream ends."""
        out, self._held = self._held, ""
        return strip_thinking(out)
