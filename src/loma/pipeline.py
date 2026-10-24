"""The end-to-end voice agent.

One turn is:

    microphone -> VAD -> faster-whisper -> control lexicon
                                          |-> fast action (no LLM)
                                          `-> Needle 3 -> MCP tools
                                                          `-> LLM -> Phonon -> speakers

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
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from loma.agent import Agent, Compaction
from loma.audio import Microphone, Speaker, UtteranceRecorder, VadConfig, write_wav
from loma.chunking import SentenceChunker
from loma.config import Settings
from loma.control import ControlMatch, ControlRouter
from loma.mcp_hub import MCPHub, ToolBatch, ToolCall, load_mcp_config
from loma.memory import MemoryGuard
from loma.ollama_client import OllamaClient
from loma.stt import Transcriber, Transcript
from loma.tool_router import NeedleRouter, ToolRoute

log = logging.getLogger(__name__)

#: Canned spoken replies for fast actions — no language model involved.
FAST_REPLIES = {
    "reset": "Okay, starting fresh.",
    "repeat_empty": "I haven't said anything yet.",
    "exit": "Goodbye.",
}


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
        self.transcriber = Transcriber(self.settings)
        self.controls = ControlRouter(self.settings.control_enabled)
        self.tool_router = NeedleRouter(self.settings)

        config = load_mcp_config(self.settings.mcp_config) if self.settings.mcp_enabled else None
        self.hub = MCPHub(config, max_tools=self.settings.tool_max_tools)
        self._tools_ready = False
        self.last_tool_route: ToolRoute | None = None
        self.last_reply = ""

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
        if not self.hub.configured:
            log.debug("no mcp.json configured; running without tools")
            return False
        if not self.hub.start():
            return False
        return self.tool_router.load(self.hub.tools)

    def warm_up(self, *, speech: bool = True) -> None:
        """Load every model, and connect to MCP, before the user speaks."""
        if self.hub.configured:
            log.info("connecting to MCP servers from %s", self.settings.mcp_config)
            self.start_tools()
        log.info("routing: control lexicon + Needle 3 (no decision model)")
        log.info(
            "warming up language model (%s, %d token window)",
            self.settings.llm_model,
            self.settings.num_ctx,
        )
        self.client.chat([{"role": "user", "content": "hi"}])
        if speech:
            log.info("warming up speech model (%s)", self.settings.tts_model)
            for _ in self.client.speech_pcm("ready"):
                pass
        log.info("warming up faster-whisper (%s)", self.settings.stt_model)
        self.transcriber.warm_up()
        # Keep the warm-up out of the real conversation.
        self.agent.reset()

    # -- text in, speech out -------------------------------------------
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
        started = time.perf_counter()
        control = self.controls.match(user_text) if route else ControlMatch()

        fast = self._fast_action(control, speak=speak, save_wav=save_wav)
        if fast is not None:
            fast.transcript = user_text
            fast.total_s = time.perf_counter() - started
            return fast

        batch = self._maybe_call_tools(user_text) if use_tools else None

        result = self.reply(
            user_text,
            speak=speak,
            save_wav=save_wav,
            on_delta=on_delta,
            on_sentence=on_sentence,
            tool_batch=batch,
        )
        result.transcript = user_text
        result.intent = control.intent
        result.route_s = control.latency_s
        result.total_s = time.perf_counter() - started
        return result

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
    ) -> TurnResult:
        """Run one LLM turn, optionally speaking it aloud."""
        started = time.perf_counter()
        chunker = SentenceChunker()
        captured: list[bytes] | None = [] if save_wav else None
        tts_stream = self._tts_stream(captured)

        speaker = Speaker(tts_stream, sample_rate=self.settings.tts_sample_rate) if speak else None
        parts: list[str] = []
        first_token = 0.0
        first_audio = 0.0

        extra_context = tool_batch.as_prompt() if tool_batch else None

        try:
            for delta in self.agent.respond(user_text, think=think, extra_context=extra_context):
                if not first_token:
                    first_token = time.perf_counter() - started
                parts.append(delta)
                if on_delta:
                    on_delta(delta)
                for sentence in chunker.push(delta):
                    if not first_audio:
                        first_audio = time.perf_counter() - started
                    if on_sentence:
                        on_sentence(sentence)
                    if speaker:
                        speaker.say(sentence)
            for sentence in chunker.flush():
                if not first_audio:
                    first_audio = time.perf_counter() - started
                if on_sentence:
                    on_sentence(sentence)
                if speaker:
                    speaker.say(sentence)
            if speaker:
                speaker.wait()
        finally:
            if speaker:
                speaker.close()

        reply_text = "".join(parts).strip()
        if reply_text:
            self.last_reply = reply_text
        if save_wav and captured is not None:
            write_wav(save_wav, b"".join(captured), self.settings.tts_sample_rate)
            log.info("wrote %s", save_wav)

        result = TurnResult(
            reply=reply_text,
            first_token_s=first_token,
            first_audio_s=first_audio,
            total_s=time.perf_counter() - started,
            compaction=self.agent.last_compaction,
        )
        if tool_batch and tool_batch.calls:
            result.tool = ", ".join(tool_batch.names)
            result.tool_args = tool_batch.calls[0].arguments
            result.tool_result = " | ".join(call.result for call in tool_batch.calls)
            result.tool_s = tool_batch.latency_s
            route = self.last_tool_route
            result.tool_confidence = route.confidence if route else 0.0
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

    # -- fast actions --------------------------------------------------
    def _fast_action(
        self, control: ControlMatch, *, speak: bool, save_wav: Path | None
    ) -> TurnResult | None:
        """Handle control intents without the language model."""
        if not control.is_fast_action:
            return None

        intent = control.intent
        log.debug("control: %s (matched %r)", intent, control.matched)

        if intent == "stop":
            # Half-duplex loop: by the time we hear "stop" nothing is playing.
            # The honest action is to end the turn silently.
            return TurnResult(intent=intent, route_s=control.latency_s)

        if intent == "reset":
            self.agent.reset()
            self.last_reply = ""
            result = self.say(FAST_REPLIES["reset"], speak=speak, save_wav=save_wav)
            result.intent = intent
            result.route_s = control.latency_s
            return result

        if intent == "repeat":
            text = self.last_reply or FAST_REPLIES["repeat_empty"]
            result = self.say(text, speak=speak, save_wav=save_wav)
            result.intent = intent
            result.route_s = control.latency_s
            return result

        if intent == "exit":
            self.say(FAST_REPLIES["exit"], speak=speak, save_wav=save_wav)
            raise StopLoop

        return None

    # -- tools ---------------------------------------------------------
    def _maybe_call_tools(self, user_text: str) -> ToolBatch | None:
        """Let Needle pick tools and fill arguments; execute them if confident."""
        if not self.settings.mcp_enabled:
            return None
        if not self.start_tools():
            return None

        route = self.tool_router.route(user_text)
        self.last_tool_route = route
        if not route.has_calls:
            log.debug("no tool: %s", route.reason)
            return None

        calls: list[ToolCall] = []
        for request in route.requests:
            started = time.perf_counter()
            result = self.hub.call(request.tool.name, request.arguments)
            latency = time.perf_counter() - started
            log.info(
                "tool %s(%s) -> %s",
                request.tool.name,
                json.dumps(request.arguments, ensure_ascii=False),
                result[:120],
            )
            calls.append(
                ToolCall(
                    tool=request.tool,
                    arguments=request.arguments,
                    result=result,
                    latency_s=latency,
                    error=result.startswith("error:"),
                )
            )
        return ToolBatch(calls=calls) if calls else None

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
