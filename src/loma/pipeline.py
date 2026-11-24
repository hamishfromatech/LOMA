"""The end-to-end voice agent.

One turn is:

    microphone -> VAD -> faster-whisper -> control lexicon
                                          |-> fast action (no LLM)
                                          `-> Needle 3 -> MCP tools
                                                          `-> LLM -> TTS

Four ideas keep this small and fast enough for a laptop:

1. **Control phrases are deterministic.** "Stop", "start over", "say that
   again", "goodbye" are matched by a lexicon, not a model. Zero RAM, zero
   latency, no misreads.
2. **Tools are chosen by Needle 3**, a 35 MB model that returns a call with the
   arguments already filled in. No tool schemas reach the LLM, and there is no
   second generation pass to extract arguments.
3. **Sentences are spoken while the model is still writing**, on a background
   thread.
4. **Compaction caps live context**, so a 128k window does not turn into a 3 GB
   KV cache.

`stream_events()` is the single turn implementation. The CLI wrappers and the
Flask API both consume it, so there is only one place where a turn is defined.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from loma.agent import Agent, Compaction
from loma.audio import (
    Microphone,
    Speaker,
    UtteranceRecorder,
    VadConfig,
    play_pcm,
    write_wav,
)
from loma.chunking import SentenceChunker
from loma.config import Settings
from loma.control import (
    ControlMatch,
    ControlRouter,
    is_capability_question,
    mentions_own_tools,
)
from loma.mcp_hub import (
    MCPHub,
    MCPTool,
    ToolBatch,
    ToolCall,
    load_mcp_config,
    ollama_tool_specs,
    tool_index,
)
from loma.memory import MemoryGuard
from loma.memory_store import MemoryStore
from loma.model_manager import ModelManager
from loma.ollama_client import OllamaClient
from loma.skills import SkillStore
from loma.stt import Transcriber, Transcript
from loma.text import ThinkingSplitter, strip_thinking
from loma.tool_router import NeedleRouter, ToolTriage

log = logging.getLogger(__name__)

#: The built-in tool that pulls a skill body into the context for one turn.
SKILL_TOOL_NAME = "load_skill"
#: Marks a tool as LOMA-owned rather than served by an MCP server.
SKILL_TOOL_SERVER = "loma"

#: Canned spoken replies for fast actions — no language model involved.
FAST_REPLIES = {
    "reset": "Okay, starting fresh.",
    "repeat_empty": "I haven't said anything yet.",
    "exit": "Goodbye.",
}


def _spoken_tool(tool: MCPTool) -> str:
    """One tool described the way a person would say it, not as a signature."""
    text = " ".join((tool.description or tool.name).split())
    # Docstrings often continue into Args:/Returns:; the first sentence is the
    # capability, the rest is for the router.
    if ". " in text:
        text = text.split(". ", 1)[0]
    text = text.rstrip(".")
    if len(text) > 110:
        text = text[:110].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    # "Return the time" -> "return the time", but leave "CPU..." alone.
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        text = text[0].lower() + text[1:]
    return text


class StopLoop(Exception):
    """Raised when the user asks the assistant to end the session."""


@dataclass(slots=True)
class TurnResult:
    transcript: str = ""
    reply: str = ""
    intent: str = "answer"
    tool: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    tool_confidence: float = 0.0
    stt_s: float = 0.0
    route_s: float = 0.0
    tool_s: float = 0.0
    first_token_s: float = 0.0
    first_audio_s: float = 0.0
    total_s: float = 0.0
    compaction: Compaction | None = None

    def summary(self) -> str:
        lines = [f"  you   : {self.transcript}"]

        if self.intent != "answer":
            lines.append(f"  route : {self.intent}  (lexicon, {self.route_s * 1000:.1f} ms)")
        else:
            lines.append(f"  route : answer  (control {self.route_s * 1000:.1f} ms)")

        if self.tool:
            arg_text = ", ".join(f"{k}={v!r}" for k, v in self.tool_args.items()) or "-"
            lines.append(
                f"  tools : {self.tool}({arg_text}) "
                f"in {self.tool_s * 1000:.0f} ms  [needle {self.tool_confidence:.2f}]"
            )
            lines.append(f"          -> {self.tool_result[:120]}")
        if self.reply:
            lines.append(f"  loma  : {self.reply}")
        if self.compaction:
            lines.append(
                f"  compacted: {self.compaction.messages_removed} messages folded, "
                f"~{self.compaction.tokens_before} -> ~{self.compaction.tokens_after} tokens"
            )
        lines.append(
            f"  timing: stt {self.stt_s:.2f}s | first token {self.first_token_s:.2f}s"
            f" | first audio {self.first_audio_s:.2f}s | total {self.total_s:.2f}s"
        )
        return "\n".join(lines)


class VoicePipeline:
    """Owns the models, the tool hub, and the wiring between them."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.client = OllamaClient(self.settings)
        self.agent = Agent(self.client, self.settings)
        #: Which chat model is resident (text or vision). Ollama does the loading;
        #: this decides which one and unloads the other.
        self.models = ModelManager(self.client, self.settings)
        self.transcriber = Transcriber(self.settings)
        self.controls = ControlRouter(self.settings.control_enabled)
        self.tool_router = NeedleRouter(self.settings)

        config = load_mcp_config(self.settings.mcp_config) if self.settings.mcp_enabled else None
        self.hub = MCPHub(config, max_tools=self.settings.tool_max_tools)
        self._tools_ready = False
        self.last_reply = ""
        #: Long-term memory, injected into every turn. Set by the caller.
        self.memory: MemoryStore | None = None

        #: Installed skills. The index is always in context; a body is only
        #: fetched when the router asks for it.
        self.skills: SkillStore | None = (
            SkillStore(self.settings.skills_root, max_chars=self.settings.skill_max_chars)
            if self.settings.skills_enabled
            else None
        )
        #: Set by _tool_events when load_skill runs, consumed by the same turn.
        self._skill_context = ""

    # -- lifecycle -----------------------------------------------------
    def close(self) -> None:
        self.tool_router.close()
        self.hub.stop()
        self.client.close()

    def __enter__(self) -> VoicePipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start_tools(self) -> bool:
        """Connect to MCP and build the Needle router once."""
        if self._tools_ready:
            return self.hub.available and self.tool_router.ready
        self._tools_ready = True
        if self.hub.configured and not self.hub.start():
            return False
        # Skills are deliberately *not* offered to the router: one extra tool
        # dropped an unrelated request's confidence from 0.54 to 0.27 on a 35 MB
        # router, and skill requests scored below its refusal threshold anyway.
        # They are matched lexically in _tool_events instead.
        tools = list(self.hub.tools)
        if not tools:
            log.debug("no tools and no skills; running without either")
            return False
        return self.tool_router.load(tools)

    def _skill_tool(self) -> MCPTool | None:
        """The built-in ``load_skill`` tool, or None when no skills exist.

        Exposed as a normal tool so the router picks it the same way it picks an
        MCP one, with one difference that matters: ``name`` is an enum of the
        installed skills, so Needle cannot invent a skill name and a near-miss
        cannot quietly load the wrong guide.
        """
        if self.skills is None:
            return None
        names = self.skills.names()
        if not names:
            return None
        return MCPTool(
            name=SKILL_TOOL_NAME,
            description=(
                "Load the full text of an installed skill. Call this before "
                "doing a task a skill covers; the guide is not in context yet."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": names,
                        "description": "The skill to load, by name.",
                    }
                },
                "required": ["name"],
            },
            server=SKILL_TOOL_SERVER,
        )

    def reload_tools(self) -> bool:
        """Re-read mcp.json and reconnect. Used by the Settings page."""
        self.tool_router.close()
        self.hub.stop()
        self._tools_ready = False
        config = (
            load_mcp_config(self.settings.mcp_config)
            if self.settings.mcp_enabled
            else None
        )
        self.hub = MCPHub(config, max_tools=self.settings.tool_max_tools)
        return self.start_tools()

    def warm_up(self, *, speech: bool = True) -> None:
        """Load every model, and connect to MCP, before the user speaks."""
        if self.hub.configured:
            log.info("connecting to MCP servers from %s", self.settings.mcp_config)
            self.start_tools()
        log.info("routing: control lexicon + Needle 3 (no decision model)")
        log.info(
            "warming up language model (%s, %d token window)",
            self.settings.llm_model,
            self.models.num_ctx(self.settings.llm_model),
        )
        self.models.prepare(self.settings.llm_model)
        self.client.chat(
            [{"role": "user", "content": "hi"}],
            model=self.settings.llm_model,
            num_ctx=self.models.num_ctx(self.settings.llm_model),
        )
        if speech:
            log.info("warming up speech model (%s)", self.settings.tts_model)
            for _ in self.client.speech_pcm("ready"):
                pass
        log.info("warming up faster-whisper (%s)", self.settings.stt_model)
        self.transcriber.warm_up()
        # Keep the warm-up out of the real conversation.
        self.agent.reset()

    # ------------------------------------------------------------------
    # The turn
    # ------------------------------------------------------------------
    def stream_events(
        self,
        user_text: str,
        *,
        speak: bool = True,
        emit_audio: bool = False,
        route: bool = True,
        use_tools: bool = True,
        think: bool = False,
        tool_batch: ToolBatch | None = None,
        images: list[str] | None = None,
        audio_format: str = "wav",
    ) -> Iterator[dict]:
        """Run one turn and yield it as a stream of plain dicts.

        Event types, in the order they can appear:

            route        the control lexicon's verdict
            tool         a tool call Needle chose, before execution
            tool_result  what the tool returned
            delta        a fragment of the reply
            audio        one spoken sentence, base64 PCM
            error        something the user needs to know about
            done         the final reply plus timings

        The CLI and the HTTP API both consume this iterator.
        """
        started = time.perf_counter()
        control = ControlMatch()

        # Choose the model before anything else runs. An image turn is a vision
        # turn; a text turn is not. `prepare` (below, after the control lexicon
        # and the router) is what actually swaps the resident model.
        model = self.models.model_for(has_images=bool(images))
        if images and not self.models.supports_vision(model):
            if self.settings.dynamic_models:
                message = (
                    f"The vision model ({model}) cannot read images. "
                    "Pick one with vision support in Settings."
                )
            else:
                message = (
                    f"{model} cannot read images. Switch to a vision model in "
                    "the model picker, or turn on dynamic models."
                )
            yield {"type": "error", "message": message}
            # Do not send the image to a model that has told us it cannot read
            # one: Ollama would reject the request with a 400 after the swap.
            return

        # -- 1. deterministic control phrases --------------------------
        if route:
            control = self.controls.match(user_text)
            yield {
                "type": "route",
                "intent": control.intent,
                "matched": control.matched,
                "latency_ms": round(control.latency_s * 1000, 3),
            }
            # "remember …" needs somewhere to write; without memory it is just
            # another thing to ask the model about.
            handled = control.is_fast_action and not (
                control.intent == "remember" and self.memory is None
            )
            if handled:
                yield from self._fast_action_events(
                    control,
                    speak=speak,
                    emit_audio=emit_audio,
                    started=started,
                    audio_format=audio_format,
                )
                return

        # -- 2. tools ---------------------------------------------------
        # Skills are still matched deterministically. Everything else is a
        # triage question: does this turn need a toolset at all? If it does,
        # the language model gets real schemas and calls the tools itself.
        triage: ToolTriage | None = None
        tool_specs: list[dict] = []
        safe_tools: dict[str, MCPTool] = {}
        batch: ToolBatch | None = None

        if use_tools and tool_batch is None:
            batch = yield from self._skill_events(user_text)
            if batch is None and self.settings.mcp_enabled:
                self.start_tools()
                if self.hub.tools:
                    if self.settings.triage_enabled and self.tool_router.ready:
                        triage = self.tool_router.classify(user_text)
                        log.debug(
                            "triage: tools=%s candidates=%s confidence=%.2f",
                            triage.needs_tools,
                            triage.candidates,
                            triage.confidence,
                        )
                    else:
                        # No triage available: offer everything rather than
                        # silently losing the tool path.
                        triage = ToolTriage(needs_tools=True, reason="triage skipped")
                    if triage.needs_tools:
                        tool_specs, safe_tools = ollama_tool_specs(
                            self._select_tools(triage)
                        )
        else:
            batch = tool_batch

        # -- 3. the language model -------------------------------------
        # Only now, with a real answer to produce, trade models. "Stop" and a
        # tool-only turn must not pay a cold load.
        model = self.models.prepare(model)
        num_ctx = self.models.num_ctx(model)

        if self.memory is not None:
            self.agent.memory_context = self.memory.as_context()
        else:
            self.agent.memory_context = ""
        # A capability question is the one case where the model needs to know
        # what exists. It gets a name plus one line of prose — never a schema —
        # and every other turn gets nothing.
        capability = (
            is_capability_question(user_text) or mentions_own_tools(user_text)
        ) and not (batch and batch.calls)
        if capability and self.settings.mcp_enabled:
            self.start_tools()
        self.agent.tool_context = tool_index(self.hub.tools) if capability else ""
        self.agent.skill_context = self._skill_note()
        chunker = SentenceChunker()
        speaker = (
            Speaker(self._tts_stream(None), sample_rate=self.settings.tts_sample_rate)
            if speak
            else None
        )
        parts: list[str] = []
        first_token = 0.0
        first_audio = 0.0
        extra_context = batch.as_prompt() if batch and batch.calls else None

        def execute_tool(name: str, arguments: dict) -> str:
            """Run one call the model asked for. Never raises into the turn."""
            tool = safe_tools.get(name)
            if tool is None:
                return f"error: no tool named {name}"
            if tool.server == SKILL_TOOL_SERVER:
                return self._run_skill_tool(arguments)
            result = self.hub.call(tool.name, arguments)
            log.info(
                "tool %s(%s) -> %s",
                tool.name,
                json.dumps(arguments, ensure_ascii=False),
                result[:120],
            )
            return result

        executed: list[dict] = []
        tool_seconds = 0.0
        splitter = ThinkingSplitter()
        try:
            for event in self.agent.respond_events(
                user_text,
                think=think,
                extra_context=extra_context,
                images=images,
                model=model,
                num_ctx=num_ctx,
                tools=tool_specs or None,
                execute_tool=execute_tool,
            ):
                if event["type"] in {"tool", "tool_result"}:
                    if event["type"] == "tool":
                        executed.append(
                            {"name": event["name"], "arguments": event["arguments"]}
                        )
                    else:
                        tool_seconds += float(event.get("latency_ms", 0.0)) / 1000
                    yield event
                    continue

                delta = event["text"]
                if not first_token:
                    first_token = time.perf_counter() - started
                parts.append(delta)
                # A reasoning model can put its scratchpad in the content stream.
                # Hold the first window of text back until we know it is not that.
                visible = splitter.push(delta)
                if not visible:
                    continue
                yield {"type": "delta", "text": visible}

                for sentence in chunker.push(visible):
                    if not first_audio:
                        first_audio = time.perf_counter() - started
                    if speaker:
                        speaker.say(sentence)
                    if emit_audio:
                        yield self._audio_event(sentence, audio_format)

            tail = splitter.flush()
            if tail:
                yield {"type": "delta", "text": tail}
                for sentence in chunker.push(tail):
                    if not first_audio:
                        first_audio = time.perf_counter() - started
                    if speaker:
                        speaker.say(sentence)
                    if emit_audio:
                        yield self._audio_event(sentence, audio_format)

            for sentence in chunker.flush():
                if not first_audio:
                    first_audio = time.perf_counter() - started
                if speaker:
                    speaker.say(sentence)
                if emit_audio:
                    yield self._audio_event(sentence, audio_format)

            if speaker:
                speaker.wait()
        finally:
            if speaker:
                speaker.close()

        reply = strip_thinking("".join(parts)).strip()
        truncated = self.agent.last_finish_reason == "length"
        if reply:
            self.last_reply = reply
        elif truncated:
            # The budget ran out before the model wrote anything — thinking
            # tokens spend from the same allowance. Silence here looks like a
            # broken assistant, so say what happened instead.
            message = (
                "The model hit its token limit before answering. "
                "Raise LOMA_MAX_TOKENS, or turn thinking off."
            )
            log.warning("empty reply, done_reason=length (max_tokens=%s)", self.settings.max_tokens)
            yield {"type": "error", "message": message}

        tool_calls = (
            [{"name": call.tool.name, "arguments": call.arguments} for call in batch.calls]
            if batch is not None
            else executed
        )
        yield self._done_event(
            reply=reply,
            truncated=truncated,
            intent=control.intent,
            control=control,
            started=started,
            first_token=first_token,
            first_audio=first_audio,
            model=model,
            tool_names=[call["name"] for call in tool_calls],
            tool_confidence=triage.confidence if triage is not None else (1.0 if batch else 0.0),
            tool_calls=tool_calls,
            tool_seconds=tool_seconds or (batch.latency_s if batch else 0.0),
        )

    # -- event helpers --------------------------------------------------
    def _fast_action_events(
        self,
        control: ControlMatch,
        *,
        speak: bool,
        emit_audio: bool,
        started: float,
        audio_format: str,
    ) -> Iterator[dict]:
        intent = control.intent
        log.debug("control: %s (matched %r)", intent, control.matched)
        text = ""

        if intent == "stop":
            # Half-duplex loop: by the time we hear "stop" nothing is playing.
            yield self._done_event(
                reply="", intent=intent, control=control, started=started
            )
            return

        if intent == "remember":
            stored = bool(self.memory and self.memory.append(control.fact, source="user"))
            text = (
                "Okay, I'll remember that."
                if stored
                else "I already knew that one."
            )

        if intent in {"identity", "about"}:
            text = self._memory_reply(intent)

        if intent == "tools":
            text = self._tools_reply()

        if intent == "reset":
            self.agent.reset()
            self.last_reply = ""
            text = FAST_REPLIES["reset"]
        elif intent == "repeat":
            text = self.last_reply or FAST_REPLIES["repeat_empty"]
        elif intent == "exit":
            text = FAST_REPLIES["exit"]

        if intent == "remember" and text:
            yield {"type": "memory", "fact": control.fact, "stored": bool(text.startswith("Okay"))}

        if text:
            # Keep deterministic answers in the transcript. Otherwise the model
            # answers the next turn without having seen the correct answer.
            if intent not in {"reset", "exit", "stop"}:
                self.agent.record_exchange(control.text, text)
            yield {"type": "delta", "text": text}
            if speak:
                play_pcm(
                    b"".join(self.client.speech_pcm(text)),
                    sample_rate=self.settings.tts_sample_rate,
                )
            if emit_audio:
                yield self._audio_event(text, audio_format)
            self.last_reply = text if intent != "reset" else ""

        yield self._done_event(
            reply=text, intent=intent, control=control, started=started
        )
        if intent == "exit":
            raise StopLoop

    def _skill_events(self, user_text: str):
        """The deterministic skill path, or None.

        Skills are matched lexically — no model, no triage. The body arrives as
        a one-shot system note rather than a flattened tool result, because a
        200-line guide would be mangled by the tool-result cap.
        """
        if self.skills is None:
            return None
        skill = self.skills.match(user_text)
        if skill is None:
            return None

        arguments = {"name": skill.name}
        tool = self._skill_tool()
        if tool is None:
            return None

        # Reported as its own intent so the trace does not claim the question
        # was never routed.
        yield {
            "type": "route",
            "intent": "skill",
            "matched": skill.name,
            "latency_ms": 0.0,
        }
        yield {
            "type": "tool",
            "name": SKILL_TOOL_NAME,
            "arguments": arguments,
            "confidence": 1.0,
        }
        started = time.perf_counter()
        result = self._run_skill_tool(arguments)
        latency = time.perf_counter() - started
        call = ToolCall(
            tool=tool,
            arguments=arguments,
            result=result,
            latency_s=latency,
            error=result.startswith("error:"),
        )
        yield {
            "type": "tool_result",
            "name": SKILL_TOOL_NAME,
            "result": result,
            "latency_ms": round(latency * 1000, 1),
            "error": call.error,
        }
        return ToolBatch(calls=[call])

    def _select_tools(self, triage: ToolTriage) -> list[MCPTool]:
        """The schema set to offer the model.

        Shortlisting is opt-in (``tool_shortlist``). By default every tool is
        offered once triage clears, because a wrong shortlist is not a smaller
        prompt — it is a model that cannot call the tool it needs and invents an
        answer instead.
        """
        tools = list(self.hub.tools)
        if (
            self.settings.tool_shortlist
            and triage.candidates
            and triage.confidence >= self.settings.tool_min_confidence
        ):
            wanted = set(triage.candidates)
            shortlist = [tool for tool in tools if tool.name in wanted]
            if shortlist:
                return shortlist
        return tools

    def _supports_vision(self, model: str | None = None) -> bool:
        """Does ``model`` (default: the selected chat model) read images?

        Checked per turn rather than cached: the model can be switched from the
        picker between one turn and the next.
        """
        return self.models.supports_vision(model or self.settings.llm_model)

    def _tools_reply(self) -> str:
        """Answer "what tools do you have?" from the live tool list.

        Deterministic on purpose, for the same reason identity is: the language
        model has never seen the tool list, so asking it is how you get an
        invented answer. The one question about the assistant's own shape is
        answered by the assistant's own state.
        """
        if self.settings.mcp_enabled:
            self.start_tools()
        tools = list(self.hub.tools)

        if not tools:
            reply = "I don't have any tools connected right now."
        else:
            spoken = [_spoken_tool(tool) for tool in tools[:6]]
            if len(spoken) == 1:
                reply = f"I have one tool: {spoken[0]}."
            elif len(spoken) == 2:
                reply = f"I have two tools: {spoken[0]}, and {spoken[1]}."
            else:
                reply = (
                    f"I have {len(tools)} tools: "
                    + ", ".join(spoken[:-1])
                    + f", and {spoken[-1]}."
                )
            if len(tools) > len(spoken):
                reply = reply[:-1] + f", plus {len(tools) - len(spoken)} more."

        names = self.skills.names() if self.skills is not None else []
        if names:
            reply += (
                f" I also have {len(names)} short skill guide"
                f"{'' if len(names) == 1 else 's'} I can load when a task needs one."
            )
        return reply

    def _memory_reply(self, intent: str) -> str:
        """Answer "who am I" / "what do you know about me" from the file.

        Deterministic on purpose. The 0.8B reads "your name is Hamish" and says
        "I am Hamish" — it takes the user's facts as its own — so the one question
        that is entirely about attribution is answered without a model.
        """
        facts = self.memory.facts() if self.memory is not None else []
        if not facts:
            return (
                "You haven't told me anything about yourself yet. "
                "You can run setup again from Settings."
            )

        if intent == "identity":
            for fact in facts:
                lowered = fact.lower()
                for prefix in ("your name is ", "prefers to be called ", "call me "):
                    if lowered.startswith(prefix):
                        return f"You're {fact[len(prefix):].strip()}."
            return "You haven't told me your name. You can set it in Settings."

        listed = facts[:4]
        spoken = "; ".join(fact[0].lower() + fact[1:] if fact else fact for fact in listed)
        more = "" if len(facts) <= 4 else f", and {len(facts) - 4} more"
        return f"Here is what I know: {spoken}{more}."

    def _skill_note(self) -> str:
        """The body of a skill loaded this turn, and nothing else.

        There used to be an index here — "call load_skill with one of these" — on
        every turn. It is gone for two reasons: skills are matched lexically now,
        so the model is never asked to choose one, and advertising a tool the
        model does not have made a 0.8B narrate instead of answer.

        Consumed exactly once, so a guide cannot ride along for the rest of the
        conversation and eat the window.
        """
        body = self._skill_context
        self._skill_context = ""
        return body

    def _run_skill_tool(self, arguments: dict) -> str:
        """Resolve ``load_skill``. Returns a short note for the tool card.

        The body itself is not returned as the tool result: tool results are
        flattened into one line by ``ToolBatch.as_prompt`` and capped, which
        would mangle a 200-line guide. Instead the body is parked in
        ``_skill_context`` and injected as its own system note, where the model
        reads it as instructions rather than as a fact about the user.
        """
        if self.skills is None:
            return "error: skills are disabled"
        name = str(arguments.get("name") or "").strip()
        body = self.skills.load(name)
        if not body:
            available = ", ".join(self.skills.names()) or "none"
            return f"error: no skill named {name!r}. Installed: {available}"
        self._skill_context = self.skills.as_context(name)
        log.info("skill loaded: %s (%d chars)", name, len(body))
        return f"loaded the {name!r} skill into context ({len(body)} characters)"

    def _audio_event(self, sentence: str, audio_format: str = "wav") -> dict:
        """One spoken sentence.

        ``wav`` is a complete file the browser can feed straight into an
        ``<audio>`` element. ``pcm`` is raw samples, which is what the CLI wants
        when it is collecting a single file to write at the end.
        """
        if audio_format == "pcm":
            audio = b"".join(self.client.speech_pcm(sentence))
        else:
            audio = self.client.speech_wav(sentence)
        return {
            "type": "audio",
            "sentence": sentence,
            "sample_rate": self.settings.tts_sample_rate,
            "format": audio_format,
            "b64": base64.b64encode(audio).decode("ascii"),
        }

    def _done_event(
        self,
        *,
        reply: str,
        intent: str,
        control: ControlMatch,
        started: float,
        first_token: float = 0.0,
        first_audio: float = 0.0,
        truncated: bool = False,
        model: str = "",
        tool_names: list[str] | None = None,
        tool_confidence: float = 0.0,
        tool_calls: list[dict] | None = None,
        tool_seconds: float = 0.0,
    ) -> dict:
        compaction = self.agent.last_compaction
        return {
            "type": "done",
            "reply": reply,
            "intent": intent,
            "matched": control.matched,
            "model": model,
            "tool": tool_names or [],
            "tool_confidence": round(float(tool_confidence), 4),
            "tool_calls": tool_calls or [],
            "timings": {
                "route_ms": round(control.latency_s * 1000, 3),
                "tool_ms": round(tool_seconds * 1000, 1),
                "first_token_s": round(first_token, 3),
                "first_audio_s": round(first_audio, 3),
                "total_s": round(time.perf_counter() - started, 3),
            },
            "context_tokens": self.agent.history_tokens,
            "truncated": truncated,
            "compaction": {
                "messages_removed": compaction.messages_removed,
                "tokens_before": compaction.tokens_before,
                "tokens_after": compaction.tokens_after,
            }
            if compaction
            else None,
        }

    # ------------------------------------------------------------------
    # Convenience wrappers over stream_events
    # ------------------------------------------------------------------
    def respond(
        self,
        user_text: str,
        *,
        speak: bool = True,
        save_wav: Path | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_sentence: Callable[[str], None] | None = None,
        route: bool = True,
        use_tools: bool = True,
    ) -> TurnResult:
        """Route, maybe call tools, and answer one text turn."""
        return self._consume(
            user_text,
            speak=speak,
            save_wav=save_wav,
            on_delta=on_delta,
            on_sentence=on_sentence,
            route=route,
            use_tools=use_tools,
        )

    def reply(
        self,
        user_text: str,
        *,
        speak: bool = True,
        on_delta: Callable[[str], None] | None = None,
        on_sentence: Callable[[str], None] | None = None,
        save_wav: Path | None = None,
        think: bool = False,
        tool_batch: ToolBatch | None = None,
        images: list[str] | None = None,
    ) -> TurnResult:
        """Answer without routing — used by benchmarks and tests."""
        return self._consume(
            user_text,
            speak=speak,
            save_wav=save_wav,
            on_delta=on_delta,
            on_sentence=on_sentence,
            route=False,
            use_tools=False,
            think=think,
            tool_batch=tool_batch,
            images=images,
        )

    def _consume(
        self,
        user_text: str,
        *,
        speak: bool,
        save_wav: Path | None,
        on_delta: Callable[[str], None] | None,
        on_sentence: Callable[[str], None] | None,
        route: bool,
        use_tools: bool,
        think: bool = False,
        tool_batch: ToolBatch | None = None,
        images: list[str] | None = None,
    ) -> TurnResult:
        started = time.perf_counter()
        result = TurnResult(transcript=user_text)
        pcm_chunks: list[bytes] = []

        for event in self.stream_events(
            user_text,
            speak=speak,
            emit_audio=save_wav is not None,
            route=route,
            use_tools=use_tools,
            think=think,
            tool_batch=tool_batch,
            images=images,
            audio_format="pcm" if save_wav else "wav",
        ):
            kind = event["type"]
            if kind == "delta":
                if on_delta:
                    on_delta(event["text"])
            elif kind == "audio":
                pcm_chunks.append(base64.b64decode(event["b64"]))
                if on_sentence:
                    on_sentence(event["sentence"])
            elif kind == "route":
                result.intent = event["intent"]
                result.route_s = event["latency_ms"] / 1000
            elif kind == "tool_result":
                result.tool_result = event["result"]
            elif kind == "done":
                result.reply = event["reply"]
                result.intent = event["intent"]
                result.tool_confidence = event["tool_confidence"]
                result.tool_s = event["timings"]["tool_ms"] / 1000
                result.first_token_s = event["timings"]["first_token_s"]
                result.first_audio_s = event["timings"]["first_audio_s"]
                result.compaction = self.agent.last_compaction
                if event["tool"]:
                    result.tool = ", ".join(event["tool"])
                calls = event.get("tool_calls") or []
                if calls:
                    result.tool_args = calls[0].get("arguments", {})

        if save_wav and pcm_chunks:
            write_wav(save_wav, b"".join(pcm_chunks), self.settings.tts_sample_rate)
            log.info("wrote %s", save_wav)

        result.total_s = time.perf_counter() - started
        return result

    def say(
        self, text: str, *, speak: bool = True, save_wav: Path | None = None
    ) -> TurnResult:
        """Speak fixed text — no language model, no router."""
        started = time.perf_counter()
        if speak and text:
            captured: list[bytes] | None = [] if save_wav else None
            speaker = Speaker(
                self._tts_stream(captured), sample_rate=self.settings.tts_sample_rate
            )
            try:
                speaker.say(text)
                speaker.wait()
            finally:
                speaker.close()
            if save_wav and captured is not None:
                write_wav(save_wav, b"".join(captured), self.settings.tts_sample_rate)
        return TurnResult(reply=text, total_s=time.perf_counter() - started)

    # -- speech in, speech out -----------------------------------------
    def transcribe_audio(self, audio: np.ndarray) -> Transcript:
        """Transcribe a float32 mono 16 kHz array."""
        return self.transcriber.transcribe(audio)

    def transcribe_file(self, path: str | Path) -> Transcript:
        return self.transcriber.transcribe(str(path))

    def turn_from_audio(
        self,
        audio: np.ndarray,
        *,
        speak: bool = True,
        save_wav: Path | None = None,
    ) -> TurnResult:
        """Transcribe ``audio``, route it, and act on it."""
        started = time.perf_counter()
        transcript = self.transcribe_audio(audio)
        stt_s = time.perf_counter() - started
        if not transcript.text:
            return TurnResult(stt_s=stt_s, total_s=stt_s)

        result = self.respond(transcript.text, speak=speak, save_wav=save_wav)
        result.stt_s = stt_s
        result.total_s = time.perf_counter() - started
        return result

    # -- interactive loop ----------------------------------------------
    def listen_loop(self, *, max_turns: int | None = None) -> None:
        """Speak, pause, hear the answer. Press Ctrl+C to stop."""
        vad = VadConfig(
            threshold_db=self.settings.silence_db,
            silence_ms=self.settings.silence_ms,
            max_utterance_s=self.settings.max_utterance_s,
            min_utterance_ms=self.settings.min_utterance_ms,
        )
        turns = 0
        guard = MemoryGuard(self.settings.memory_budget_mb, self.client)
        guard.start()

        try:
            with Microphone(self.settings) as mic:
                recorder = UtteranceRecorder(vad, self.settings.sample_rate)
                print("Listening. Speak, then pause. Say 'goodbye' or press Ctrl+C to stop.\n")
                for frame in mic.frames():
                    utterance = recorder.push(frame)
                    if utterance is None:
                        continue
                    try:
                        result = self.turn_from_audio(utterance)
                    except StopLoop:
                        print("Goodbye.")
                        break
                    if not result.transcript:
                        print("(heard nothing intelligible)\n")
                        continue
                    print(result.summary() + "\n")
                    turns += 1
                    if max_turns is not None and turns >= max_turns:
                        break
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            guard.stop()
            print(
                f"Peak observed memory: {guard.peak_mb:.0f} MB "
                f"(budget {self.settings.memory_budget_mb} MB)"
            )

    # -- helpers -------------------------------------------------------
    def _tts_stream(
        self, captured: list[bytes] | None
    ) -> Callable[[str], Iterator[bytes]]:
        """Wrap ``speech_pcm`` so audio can optionally be saved as it plays."""

        def stream(text: str) -> Iterator[bytes]:
            for chunk in self.client.speech_pcm(text):
                if captured is not None:
                    captured.append(chunk)
                yield chunk

        return stream
