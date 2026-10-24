"""Conversation state, token accounting, and automatic context compaction.

The context window is large (128k, or 256k on bigger machines) but it is not
infinite. Two mechanisms keep a long conversation inside it:

1. **Trimming** — the crude safety net: never keep more than
   ``history_turns`` exchanges.
2. **Compaction** — the real mechanism: when the prompt actually crosses
   ``compact_at`` of the window, the older half of the conversation is
   summarised by the language model into a single system message.

Compaction is driven by Ollama's own ``prompt_eval_count`` rather than a
character heuristic, so it fires on the truth and not on an estimate.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from loma.config import Settings
from loma.context import estimate_messages_tokens
from loma.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)

SUMMARY_PREFIX = "Earlier in this conversation:"

SUMMARY_INSTRUCTIONS = (
    "You compress a conversation into a brief note for your future self. "
    "Keep every fact, name, number, preference, decision and open question. "
    "Drop pleasantries and repetition. Write plain prose, at most 150 words. "
    "Do not add commentary or headings."
)


@dataclass(slots=True)
class Compaction:
    """What one compaction pass did."""

    messages_removed: int
    tokens_before: int
    tokens_after: int
    summary: str = ""

    @property
    def freed_tokens(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


class Agent:
    """A small stateful chat agent backed by Ollama."""

    def __init__(self, client: OllamaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.last_prompt_tokens = 0
        self.compactions = 0
        self.last_compaction: Compaction | None = None
        #: Long-term memory, injected per request and never persisted here.
        self.memory_context = ""
        #: The skill index every turn, plus any skill body loaded this turn.
        self.skill_context = ""
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": settings.system_prompt}
        ]

    # -- conversation ---------------------------------------------------
    @property
    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    @property
    def history_tokens(self) -> int:
        """Real prompt size from the last call, or an estimate before any."""
        return self.last_prompt_tokens or estimate_messages_tokens(self._messages)

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": self.settings.system_prompt}]
        self.last_prompt_tokens = 0
        self.last_compaction = None

    def load(self, messages: list[dict[str, str]]) -> None:
        """Restore a conversation from disk, keeping the current system prompt."""
        self._messages = [{"role": "system", "content": self.settings.system_prompt}]
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role in {"user", "assistant"} and content:
                self._messages.append({"role": role, "content": content})
        self.last_prompt_tokens = 0
        self.last_compaction = None

    def transcript(self) -> list[dict[str, str]]:
        """The persistable part of the conversation — no system messages."""
        return [m for m in self._messages if m["role"] in {"user", "assistant"}]

    def respond(
        self,
        user_text: str,
        *,
        think: bool = False,
        extra_context: str | None = None,
    ) -> Iterator[str]:
        """Yield assistant token deltas for ``user_text``.

        ``extra_context`` is a one-shot system note (a tool result, for
        example). It is used for this call only and never persisted, so tool
        payloads cannot accumulate in the window.
        """
        user_text = user_text.strip()
        if not user_text:
            return iter(())

        # Compact before adding the new question, using real usage from the
        # previous turn.
        self.last_compaction = self._maybe_compact()

        self._messages.append({"role": "user", "content": user_text})
        self._trim()

        messages = self.messages
        notes = [
            note
            for note in (self.memory_context, self.skill_context, extra_context)
            if note
        ]
        if notes:
            # One system message, not four. Several system turns make a small
            # model narrate about its own instructions ("looking at my available
            # skills…") instead of answering, and some chat templates only honour
            # the first system message anyway.
            messages[0] = {
                "role": "system",
                "content": "\n\n".join(
                    [
                        self.settings.system_prompt,
                        *notes,
                        # Last, deliberately. A small model follows the line it
                        # read most recently, and the notes above are dense.
                        "Answer the user now, in your own voice. No preamble, no "
                        "restating the question, no describing what you are doing.",
                    ]
                ),
            }

        return self._stream(messages, think=think)

    def _stream(self, messages: list[dict[str, str]], *, think: bool) -> Iterator[str]:
        parts: list[str] = []
        usage: dict[str, int] = {}

        def record(counts: dict[str, int]) -> None:
            usage.update(counts)

        try:
            for delta in self.client.chat_stream(messages, think=think, on_usage=record):
                parts.append(delta)
                yield delta
        finally:
            if usage.get("prompt_tokens"):
                self.last_prompt_tokens = usage["prompt_tokens"]
            else:
                self.last_prompt_tokens = estimate_messages_tokens(messages)

            reply = "".join(parts).strip()
            if reply:
                self._messages.append({"role": "assistant", "content": reply})
            else:
                # Nothing was produced — roll the user turn back so the next
                # prompt is not polluted by a dangling turn.
                if self._messages and self._messages[-1].get("role") == "user":
                    self._messages.pop()
            self._trim()

    # -- memory policy --------------------------------------------------
    def _trim(self) -> None:
        """Keep the system prompt plus the most recent N exchanges."""
        max_messages = self.settings.history_turns * 2
        if len(self._messages) <= max_messages + 1:
            return
        system = self._messages[0]
        tail = self._messages[-max_messages:]
        # Never start the window on an assistant turn.
        while tail and tail[0]["role"] == "assistant":
            tail.pop(0)
        self._messages = [system, *tail]
        log.debug("history trimmed to %d messages", len(self._messages))

    # -- compaction ------------------------------------------------------
    def _maybe_compact(self) -> Compaction | None:
        """Summarise older turns when the real prompt crosses the threshold."""
        if not self.settings.compact_enabled:
            return None

        used = self.last_prompt_tokens or estimate_messages_tokens(self._messages)
        threshold = self.settings.compact_threshold_tokens
        if used < threshold:
            return None

        keep = max(2, self.settings.compact_keep_turns * 2)
        # system + summary + keep + one new exchange must still fit.
        if len(self._messages) <= keep + 2:
            log.debug("context is large but there is nothing old enough to compact")
            return None

        system = self._messages[0]
        old = self._messages[1:-keep]
        recent = self._messages[-keep:]

        log.info(
            "compacting context: %d tokens (threshold %d), folding %d messages",
            used,
            threshold,
            len(old),
        )
        summary = self._summarise(old)
        if not summary:
            log.warning("compaction produced no summary; dropping old turns instead")
            summary = "(earlier turns were dropped)"

        self._messages = [
            system,
            {"role": "system", "content": f"{SUMMARY_PREFIX}\n{summary}"},
            *recent,
        ]
        self.compactions += 1
        self.last_prompt_tokens = 0

        result = Compaction(
            messages_removed=len(old),
            tokens_before=used,
            tokens_after=estimate_messages_tokens(self._messages),
            summary=summary,
        )
        log.info(
            "compacted %d messages, ~%d -> ~%d tokens",
            result.messages_removed,
            result.tokens_before,
            result.tokens_after,
        )
        return result

    def _summarise(self, messages: list[dict[str, str]]) -> str:
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages if m.get("content")
        ).strip()
        if not transcript:
            return ""

        try:
            return self.client.chat(
                [
                    {"role": "system", "content": SUMMARY_INSTRUCTIONS},
                    {"role": "user", "content": transcript},
                ],
                max_tokens=self.settings.compact_max_tokens,
            ).strip()
        except OllamaError as exc:
            log.warning("summarisation failed: %s", exc)
            return ""
