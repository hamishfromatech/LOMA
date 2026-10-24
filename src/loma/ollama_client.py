"""A tiny HTTP client for the two Ollama endpoints LOMA needs.

Ollama exposes:
  * ``POST /api/chat``          — chat completions (NDJSON streaming)
  * ``POST /v1/audio/speech``   — OpenAI-compatible text-to-speech

We talk to both with plain HTTP instead of pulling in an SDK, which keeps the
dependency tree and the memory footprint small — and makes the wire format
visible for students.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from loma.config import Settings

# TTS requests are capped by the server at 4096 characters.
MAX_SPEECH_CHARS = 4000

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error payload."""


class OllamaClient:
    """Thin, explicit wrapper around the Ollama HTTP API."""

    def __init__(self, settings: Settings, timeout: float = 300.0) -> None:
        self.settings = settings
        self._http = httpx.Client(
            base_url=settings.ollama_host,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    # -- lifecycle -----------------------------------------------------
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- diagnostics ---------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Return Ollama's version string, or raise a helpful error."""
        try:
            response = self._http.get("/api/version")
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.settings.ollama_host}. "
                "Is `ollama serve` running?"
            ) from exc
        _raise_for_status(response)
        return response.json()

    def loaded_models(self) -> list[dict[str, Any]]:
        """Models currently resident in memory (``/api/ps``)."""
        response = self._http.get("/api/ps")
        _raise_for_status(response)
        return response.json().get("models", [])

    def list_models(self) -> list[dict[str, Any]]:
        """Every model installed locally, smallest first."""
        response = self._http.get("/api/tags")
        _raise_for_status(response)
        models = []
        for model in response.json().get("models", []):
            name = model.get("name", "")
            details = model.get("details") or {}
            models.append(
                {
                    "name": name,
                    "size_mb": round(model.get("size", 0) / 1_000_000, 1),
                    "family": details.get("family", ""),
                    "parameters": details.get("parameter_size", ""),
                    "capabilities": self.capabilities(name),
                }
            )
        return sorted(models, key=lambda item: item["size_mb"])

    def capabilities(self, model: str) -> list[str]:
        """A model's capabilities: completion, vision, tools, thinking, speech…"""
        try:
            response = self._http.post("/api/show", json={"model": model}, timeout=10)
        except httpx.HTTPError:
            return []
        if not response.is_success:
            return []
        return list(response.json().get("capabilities") or [])

    def voices(self, model: str | None = None) -> list[str]:
        """Discover a speech model's voices.

        This Ollama build does not expose ``voices`` in ``/api/show``, but an
        unknown voice is rejected with the full list, so probe once and parse it.
        """
        model = model or self.settings.tts_model
        probe = {
            "model": model,
            "input": "hi",
            "voice": "__loma_probe__",
            "response_format": "wav",
        }
        try:
            response = self._http.post("/v1/audio/speech", json=probe)
        except httpx.HTTPError:
            return []
        if response.status_code == 200:
            # The server accepts arbitrary voices; offer the documented ones.
            return ["default", "Zoey", "Easton", "Freya"]
        try:
            message = str((response.json().get("error") or {}).get("message", ""))
        except ValueError:
            message = response.text
        marker = "available voices:"
        if marker in message:
            listed = message.split(marker, 1)[1]
            return [voice.strip() for voice in listed.split(",") if voice.strip()]
        return []

    def unload(self, model: str) -> None:
        """Ask Ollama to drop a model from memory now."""
        try:
            self._http.post("/api/generate", json={"model": model, "keep_alive": 0})
        except httpx.HTTPError as exc:
            log.debug("could not unload %s: %s", model, exc)

    def has_model(self, name: str) -> bool:
        """True when ``name`` appears in the local model list."""
        response = self._http.get("/api/tags")
        _raise_for_status(response)
        wanted = name.split(":")[0]
        for model in response.json().get("models", []):
            model_name = model.get("name", "")
            if model_name == name or model_name.split(":")[0] == wanted:
                return True
        return False

    # -- chat ----------------------------------------------------------
    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        think: bool = False,
        max_tokens: int | None = None,
        on_usage: Callable[[dict[str, int]], None] | None = None,
    ) -> Iterator[str]:
        """Yield assistant text fragments as the model produces them.

        ``on_usage`` receives Ollama's final accounting dict, which contains
        ``prompt_eval_count`` (the real size of the context we just used) and
        ``eval_count``. The agent uses that number to decide when to compact
        instead of guessing.
        """
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "stream": True,
            # Bonsai is a reasoning model. Thinking is wasted latency for a
            # spoken reply, so it is off by default.
            "think": think,
            "keep_alive": self.settings.keep_alive,
            "options": {
                "num_ctx": self.settings.num_ctx,
                "num_predict": max_tokens or self.settings.max_tokens,
                "temperature": self.settings.temperature,
            },
        }
        with self._http.stream("POST", "/api/chat", json=payload) as response:
            _raise_for_status(response, streamed=True)
            for line in response.iter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if error := chunk.get("error"):
                    raise OllamaError(str(error))
                message = chunk.get("message") or {}
                content = message.get("content")
                if content:
                    yield content
                if chunk.get("done"):
                    if on_usage is not None:
                        on_usage(
                            {
                                "prompt_tokens": int(chunk.get("prompt_eval_count") or 0),
                                "completion_tokens": int(chunk.get("eval_count") or 0),
                            }
                        )
                    break

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        think: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming convenience wrapper used by tests and benchmarks."""
        return "".join(self.chat_stream(messages, think=think, max_tokens=max_tokens))

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        max_tokens: int = 200,
    ) -> dict[str, Any]:
        """Constrained generation: Ollama forces the reply to match ``schema``.

        This is how LOMA turns a spoken sentence into tool arguments without
        asking the model to remember a JSON dialect.
        """
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "keep_alive": self.settings.keep_alive,
            "options": {
                "num_ctx": self.settings.num_ctx,
                "num_predict": max_tokens,
                "temperature": 0.0,
            },
        }
        response = self._http.post("/api/chat", json=payload)
        _raise_for_status(response)
        content = (response.json().get("message") or {}).get("content", "")
        try:
            return json.loads(content)
        except ValueError as exc:
            raise OllamaError(f"model did not return valid JSON: {content[:200]}") from exc

    # -- text to speech -------------------------------------------------
    def speech_pcm(self, text: str, *, voice: str | None = None) -> Iterator[bytes]:
        """Stream raw 16-bit little-endian mono 24 kHz PCM for ``text``.

        We ask for ``pcm`` rather than ``wav`` because the WAV header that
        Ollama writes while streaming has placeholder length fields — PCM is
        exactly what the sound card wants anyway.
        """
        text = text.strip()[:MAX_SPEECH_CHARS]
        if not text:
            return iter(())

        payload = {
            "model": self.settings.tts_model,
            "input": text,
            "voice": voice or self.settings.tts_voice,
            "response_format": "pcm",
        }
        with self._http.stream("POST", "/v1/audio/speech", json=payload) as response:
            _raise_for_status(response, streamed=True)
            yield from response.iter_bytes()

    def speech_wav(self, text: str, *, voice: str | None = None) -> bytes:
        """Return a complete WAV file for ``text`` (handy for saving output)."""
        text = text.strip()[:MAX_SPEECH_CHARS]
        if not text:
            return b""
        payload = {
            "model": self.settings.tts_model,
            "input": text,
            "voice": voice or self.settings.tts_voice,
            "response_format": "wav",
        }
        response = self._http.post("/v1/audio/speech", json=payload)
        _raise_for_status(response)
        return response.content

    # -- housekeeping ---------------------------------------------------
    def warm_up(self) -> None:
        """Load both models now so the first real turn feels instant."""
        self.chat([{"role": "user", "content": "hi"}])
        for _ in self.speech_pcm("ready"):
            pass


def _raise_for_status(response: httpx.Response, *, streamed: bool = False) -> None:
    """Turn an Ollama error body into a readable exception."""
    if response.is_success:
        return
    if streamed:
        response.read()
    detail = response.text.strip()
    try:
        parsed = json.loads(detail)
        detail = parsed.get("error") or detail
    except (ValueError, AttributeError):
        pass
    raise OllamaError(f"Ollama returned {response.status_code}: {detail[:500]}")
