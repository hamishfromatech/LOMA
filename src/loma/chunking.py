"""Turn a stream of LLM tokens into speakable sentences.

Streaming text-to-speech works best per sentence: the first sentence can be
spoken while the model is still writing the second. This module does that
splitting without any regex magic that would break on ``3.14`` or ``Dr.``.
"""

from __future__ import annotations

_TERMINATORS = ".!?…"
_TRAILING = "\"'”’)]}"


class SentenceChunker:
    """Accumulate token deltas and emit complete sentences.

    >>> c = SentenceChunker(min_chars=1)
    >>> c.push("Hello there. How ")
    ['Hello there.']
    >>> c.push("are you?")
    ['How are you?']
    """

    def __init__(self, min_chars: int = 20, max_chars: int = 280) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""

    def push(self, delta: str) -> list[str]:
        """Add a token delta and return any sentences that are now complete."""
        self._buffer += delta
        out: list[str] = []

        while True:
            cut = self._find_boundary()
            if cut is None:
                break
            sentence = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:].lstrip()
            if sentence:
                out.append(sentence)

        # Hard safety valve for run-on text with no punctuation.
        while len(self._buffer) > self.max_chars:
            split_at = self._buffer.rfind(" ", 0, self.max_chars)
            if split_at <= 0:
                split_at = self.max_chars
            out.append(self._buffer[:split_at].strip())
            self._buffer = self._buffer[split_at:].lstrip()

        return [s for s in out if s]

    def flush(self) -> list[str]:
        """Return whatever is left in the buffer."""
        rest = self._buffer.strip()
        self._buffer = ""
        return [rest] if rest else []

    # -- internals -----------------------------------------------------
    def _find_boundary(self) -> int | None:
        buf = self._buffer
        for index, char in enumerate(buf):
            if char == "\n":
                if index + 1 >= self.min_chars:
                    return index + 1
                continue
            if char not in _TERMINATORS:
                continue
            # Consume trailing quotes/brackets, e.g.  ``"Done."``
            end = index + 1
            while end < len(buf) and buf[end] in _TRAILING:
                end += 1
            # A sentence must be followed by whitespace or the end of stream.
            if end < len(buf) and not buf[end].isspace():
                continue
            if end < self.min_chars:
                continue
            if self._is_abbreviation(index):
                continue
            return end
        return None

    def _is_abbreviation(self, index: int) -> bool:
        """True for ``Dr.``, ``e.g.``, ``3.14`` — punctuation we must not split."""
        buf = self._buffer
        if buf[index] != ".":
            return False
        if index == 0:
            return False
        prev = buf[index - 1]
        if prev.isdigit():
            return True
        # A single capital letter before the dot is an initial: "J. Smith".
        if prev.isupper() and (index < 2 or not buf[index - 2].isalpha()):
            return True
        # Known short abbreviations.
        word_start = index - 1
        while word_start >= 0 and buf[word_start].isalpha():
            word_start -= 1
        word = buf[word_start + 1 : index].lower()
        return word in {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e"}
