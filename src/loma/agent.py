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

import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

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


def _parse_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Read Ollama's tool_calls shape, tolerating a JSON-string argument."""
    function = call.get("function") or {}
    name = str(function.get("name") or call.get("name") or "")
    raw = function.get("arguments", call.get("arguments"))
    if isinstance(raw, dict):
        arguments = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            arguments = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            arguments = {}
    else:
        arguments = {}
    return name, arguments


class Agent:
    """A small stateful chat agent backed by Ollama."""

    def __init__(self, client: OllamaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.last_prompt_tokens = 0
        #: ``done_reason`` from the last call: "stop", "length", or empty.
        self.last_finish_reason = ""
        self.compactions = 0
        self.last_compaction: Compaction | None = None
        #: Long-term memory, injected per request and never persisted here.
        self.memory_context = ""
        #: A name + one-line index of the tools, set only for a capability
        #: question. Never schemas: the router owns tool calls.
        self.tool_context = ""
        #: Any skill body loaded this turn.
        self.skill_context = ""
        #: The chat model answering this turn, set by ``respond``. Compaction
        #: uses the same model, so an image turn does not swap back to the text
        #: model just to summarise.
        self.model: str | None = None
        self.num_ctx: int | None = None
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

    def load(self, messages: list[dict[str, Any]]) -> None:
        """Restore a conversation from disk, keeping the current system prompt.

        A saved tool trace is replayed as the wire protocol it came from
        (assistant ``tool_calls`` + a ``role: tool`` message) with the results
        omitted. Without that shape in the recent history, `lfm3` tends to
        answer a follow-up from memory instead of calling the tool: after one
        roll-die turn, "what time is it?" produced a made-up time 5/5 times.
        """
        self._messages = [{"role": "system", "content": self.settings.system_prompt}]
        last_trace = max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "assistant" and message.get("tool_trace")
            ),
            default=-1,
        )
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", "")
            if role not in {"user", "assistant"} or not content:
                continue
            if index == last_trace:
                calls = []
                for position, item in enumerate(message.get("tool_trace") or []):
                    name = str(item.get("name") or "") if isinstance(item, dict) else ""
                    if not name:
                        continue
                    arguments = (
                        item.get("arguments")
                        if isinstance(item, dict) and isinstance(item.get("arguments"), dict)
                        else {}
                    )
                    calls.append(
                        {
                            "id": f"replay_{index}_{position}",
                            "function": {
                                "index": position,
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    )
                if calls:
                    self._messages.append(
                        {"role": "assistant", "content": "", "tool_calls": calls}
                    )
                    for _call in calls:
                        self._messages.append(
                            {"role": "tool", "content": "(result omitted from replay)"}
                        )
            self._messages.append({"role": role, "content": content})
        self.last_prompt_tokens = 0
        self.last_compaction = None

    def transcript(self) -> list[dict[str, Any]]:
        """The persistable part of the conversation — no system messages.

        Only the most recent tool trace is kept. It carries names and arguments,
        never results, and it is what lets the next turn's model recognise that
        this conversation uses tools (see ``load``).
        """
        messages = [m for m in self._messages if m["role"] in {"user", "assistant"}]
        last_trace = max(
            (index for index, message in enumerate(messages) if message.get("tool_trace")),
            default=-1,
        )
        for index, message in enumerate(messages):
            if index != last_trace:
                message.pop("tool_trace", None)
        return messages

    def record_exchange(self, user_text: str, reply: str) -> None:
        """Store an exchange that was answered without a model.

        Identity, capability and "remember…" answers are deterministic. Without
        this they vanish from the transcript, so the next model turn sees the
        user's question but not the correct answer — and contradicts it (a real
        case: "what tools do you have?" answered correctly, then the next turn
        claimed it had no tools).
        """
        user_text = user_text.strip()
        reply = reply.strip()
        if not user_text or not reply:
            return
        self._messages.append({"role": "user", "content": user_text})
        self._messages.append({"role": "assistant", "content": reply})
        self._trim()

    def _begin_turn(
        self,
        user_text: str,
        *,
        images: list[str] | None,
        model: str | None,
        num_ctx: int | None,
        extra_context: str | None,
    ) -> list[dict[str, Any]]:
        """Compact, append the user turn, and inject this turn's system notes."""
        self.model = model
        self.num_ctx = num_ctx

        # Compact before adding the new question, using real usage from the
        # previous turn.
        self.last_compaction = self._maybe_compact()

        turn: dict[str, Any] = {"role": "user", "content": user_text}
        if images:
            # Ollama takes base64 on the message itself. Vision models read it;
            # text models ignore the field, so the caller checks the capability
            # first rather than letting an image vanish silently.
            turn["images"] = images
        self._messages.append(turn)
        self._trim()

        messages = self.messages
        notes = [
            note
            for note in (
                self.memory_context,
                self.tool_context,
                self.skill_context,
                extra_context,
            )
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
        return messages

    def _end_turn(
        self,
        reply: str,
        usage: dict[str, Any],
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record usage, the answer, and the usual trimming."""
        self.last_finish_reason = str(usage.get("done_reason", ""))
        if usage.get("prompt_tokens"):
            self.last_prompt_tokens = int(usage["prompt_tokens"])
        else:
            self.last_prompt_tokens = estimate_messages_tokens(self.messages)

        reply = reply.strip()
        if reply:
            message: dict[str, Any] = {"role": "assistant", "content": reply}
            if tool_trace:
                message["tool_trace"] = tool_trace
            self._messages.append(message)
        else:
            # Nothing was produced — roll the user turn back so the next prompt
            # is not polluted by a dangling turn.
            if self._messages and self._messages[-1].get("role") == "user":
                self._messages.pop()
        self._trim()
        self._drop_images()

    def respond(
        self,
        user_text: str,
        *,
        think: bool | str = False,
        extra_context: str | None = None,
        on_thinking: Callable[[str], None] | None = None,
        images: list[str] | None = None,
        model: str | None = None,
        num_ctx: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        execute_tool: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> Iterator[str]:
        """Yield only the assistant's text; tool protocol events are consumed.

        This is the text-only view used by tests and the CLI. The pipeline uses
        :meth:`respond_events`, which also yields the tool calls so they can be
        shown as cards and executed.
        """
        for event in self.respond_events(
            user_text,
            think=think,
            extra_context=extra_context,
            on_thinking=on_thinking,
            images=images,
            model=model,
            num_ctx=num_ctx,
            tools=tools,
            execute_tool=execute_tool,
        ):
            if event["type"] == "delta":
                yield event["text"]

    def respond_events(
        self,
        user_text: str,
        *,
        think: bool | str = False,
        extra_context: str | None = None,
        on_thinking: Callable[[str], None] | None = None,
        images: list[str] | None = None,
        model: str | None = None,
        num_ctx: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        execute_tool: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """One turn, including native tool calls, as a stream of events.

        Events are ``{"type": "delta", "text": ...}``,
        ``{"type": "tool", "name": ..., "arguments": ...}`` and
        ``{"type": "tool_result", "name": ..., "result": ..., "latency_ms": ...,
        "error": ...}``. The final answer may only arrive after one or more tool
        rounds, and only the final answer is kept in the transcript: the
        assistant tool-call message and the tool results exist for this turn.
        """
        user_text = user_text.strip()
        if not user_text:
            return

        messages = self._begin_turn(
            user_text,
            images=images,
            model=model,
            num_ctx=num_ctx,
            extra_context=extra_context,
        )

        max_rounds = max(0, self.settings.tool_rounds) if tools else 0
        usage: dict[str, Any] = {}
        final_reply = ""
        partial_reply = ""
        tool_trace: list[dict[str, Any]] = []
        finished = False

        try:
            for round_index in range(max_rounds + 1):
                active_tools = tools if (tools and round_index < max_rounds) else None
                calls: list[dict[str, Any]] = []
                parts: list[str] = []

                def record(counts: dict[str, int]) -> None:
                    usage.update(counts)

                for delta in self.client.chat_stream(
                    messages,
                    think=think,
                    model=model,
                    num_ctx=num_ctx,
                    tools=active_tools,
                    on_usage=record,
                    on_thinking=on_thinking,
                    on_tool_calls=calls.extend,
                ):
                    parts.append(delta)
                    yield {"type": "delta", "text": delta}
                partial_reply = "".join(parts)

                if not calls or active_tools is None:
                    final_reply = partial_reply
                    break

                self._messages.append(
                    {"role": "assistant", "content": partial_reply, "tool_calls": calls}
                )

                for call in calls:
                    name, arguments = _parse_tool_call(call)
                    tool_trace.append({"name": name, "arguments": arguments})
                    yield {"type": "tool", "name": name, "arguments": arguments}

                    started = time.perf_counter()
                    try:
                        if execute_tool is None:
                            result = f"error: no executor for {name}"
                        else:
                            result = execute_tool(name, arguments)
                        failed = result.startswith("error:")
                    except Exception as exc:  # noqa: BLE001 - a tool must not end a turn
                        result = f"error: {type(exc).__name__}: {exc}"
                        failed = True
                    latency = time.perf_counter() - started

                    # Tool output is context, not an essay. Cap it so one verbose
                    # MCP result cannot eat the window.
                    result = result[:4000]
                    yield {
                        "type": "tool_result",
                        "name": name,
                        "result": result,
                        "latency_ms": round(latency * 1000, 1),
                        "error": failed,
                    }
                    self._messages.append({"role": "tool", "content": result})

                messages = self.messages

            finished = True
        finally:
            # The tool protocol is per-turn. Keep the user question and the final
            # answer; drop the assistant tool_calls message and every tool result
            # so neither the next prompt nor the conversation file grows by them.
            self._messages = [
                message
                for message in self._messages
                if message.get("role") != "tool" and not message.get("tool_calls")
            ]
            self._end_turn(
                final_reply if finished else partial_reply,
                usage,
                tool_trace=tool_trace,
            )

    def _drop_images(self) -> None:
        """Images are one-shot: keep the words, drop the pixels.

        Two reasons. The conversation is persisted as text, and — with dynamic
        model loading — the *next* turn is answered by the text model, which
        rejects a history that still carries an image. The image did its job for
        the turn that asked about it; the transcript keeps the question.
        """
        for message in self._messages:
            message.pop("images", None)

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
                model=self.model,
                num_ctx=self.num_ctx,
            ).strip()
        except OllamaError as exc:
            log.warning("summarisation failed: %s", exc)
            return ""
