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
import re
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from loma.config import Settings

# TTS requests are capped by the server at 4096 characters.
MAX_SPEECH_CHARS = 4000

#: Extra completion budget granted when thinking is requested.
THINKING_ALLOWANCE = 768


def human_parameters(raw: str) -> str:
    """Ollama reports ``parameter_size`` as "0.8B" on local models but as a raw
    count ("27356728560") on cloud ones. Either way the picker wants "27.4B"."""
    text = str(raw or "").strip()
    if not text:
        return ""
    if not text.isdigit():
        # Ollama already humanised it; just trim "852.88M" to "852.9M".
        match = re.fullmatch(r"([\d.]+)\s*([KMBT])", text)
        if match:
            return f"{round(float(match.group(1)), 1):g}{match.group(2)}"
        return text
    value = float(text)
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if value >= size:
            return f"{value / size:.1f}{unit}".replace(".0", "")
    return f"{int(value)}"


def _voices_from_payload(payload: dict[str, Any]) -> list[str]:
    """``voices`` is a list of names, or of objects carrying ``name``/``id``."""
    found: list[str] = []
    for voice in payload.get("voices") or []:
        if isinstance(voice, str):
            found.append(voice)
        elif isinstance(voice, dict):
            name = voice.get("name") or voice.get("id")
            if name:
                found.append(str(name))
    return _ordered_voices(found)


def _ordered_voices(voices: list[str]) -> list[str]:
    """Default first, then alphabetical and de-duplicated (case-insensitive)."""
    seen: dict[str, str] = {}
    for voice in voices:
        seen.setdefault(voice.casefold(), voice)
    ordered = sorted(seen.values(), key=lambda voice: voice.casefold())
    for index, voice in enumerate(ordered):
        if voice.casefold() == "default":
            ordered.insert(0, ordered.pop(index))
            break
    return ordered

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
        self._detail_cache: dict[str, dict[str, Any]] = {}

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

    def list_models(self, *, probe: bool = True) -> list[dict[str, Any]]:
        """Every model installed locally, smallest first.

        Ollama has two listings and they answer slightly different questions:

        * ``GET /api/tags`` — the native list; carries size, family, parameter
          count and quantisation. This is the one to prefer.
        * ``GET /v1/models`` — the OpenAI-compatible list; only ids and
          timestamps, but it exists on every build that speaks ``/v1``.

        We try ``/api/tags`` first and fall back to ``/v1/models`` so the model
        picker still fills when the native route is missing or proxied away.
        """
        try:
            models = self._models_from_tags()
        except (httpx.HTTPError, OllamaError):
            models = self._models_from_v1()
        if probe:
            for model in models:
                # Ollama 0.40.10 puts capabilities in /api/tags, and MLX builds
                # leave family and parameter_size blank there — /api/show is the
                # only place those two survive, so fill only what is missing.
                if not model["capabilities"] or not model["parameters"] or not model["family"]:
                    info = self.model_info(model["name"])
                    model["capabilities"] = model["capabilities"] or info["capabilities"]
                    model["parameters"] = model["parameters"] or info["parameters"]
                    model["family"] = model["family"] or info["family"]
                    model["thinking"] = model.get("thinking") or info["thinking"]
        return sorted(models, key=lambda item: item["size_mb"])

    def _models_from_tags(self) -> list[dict[str, Any]]:
        response = self._http.get("/api/tags")
        _raise_for_status(response)
        models = []
        for model in response.json().get("models", []):
            name = model.get("name") or model.get("model", "")
            if not name:
                continue
            details = model.get("details") or {}
            models.append(
                {
                    "name": name,
                    "size_mb": round(model.get("size", 0) / 1_000_000, 1),
                    "family": details.get("family", ""),
                    "parameters": human_parameters(details.get("parameter_size", "")),
                    "quantization": details.get("quantization_level", ""),
                    "modified_at": model.get("modified_at", ""),
                    # Ollama 0.40.10+ includes this; older builds do not.
                    "capabilities": list(model.get("capabilities") or []),
                    "thinking": model.get("thinking"),
                    "source": "api/tags",
                }
            )
        return models

    def _models_from_v1(self) -> list[dict[str, Any]]:
        response = self._http.get("/v1/models")
        _raise_for_status(response)
        models = []
        for model in response.json().get("data", []):
            name = model.get("id", "")
            if not name:
                continue
            models.append(
                {
                    "name": name,
                    "size_mb": 0.0,
                    "family": "",
                    "parameters": "",
                    "quantization": "",
                    "modified_at": str(model.get("created", "")),
                    "capabilities": [],
                    "thinking": None,
                    "source": "v1/models",
                }
            )
        return models

    def model_info(self, model: str) -> dict[str, Any]:
        """The useful half of ``/api/show``, cached per model.

        Returns capabilities plus whatever identifying detail the model carries:
        an MLX build reports ``parameter_size: ""`` in ``/api/tags`` but
        ``"852.88M"`` here, with ``general.parameter_count`` as a backstop.
        """
        cached = self._detail_cache.get(model)
        if cached is not None:
            return cached

        empty = {"capabilities": [], "parameters": "", "family": "", "voices": [], "thinking": None}
        try:
            response = self._http.post("/api/show", json={"model": model}, timeout=10)
        except httpx.HTTPError:
            return empty
        if not response.is_success:
            return empty

        payload = response.json()
        details = payload.get("details") or {}
        info = payload.get("model_info") or {}

        parameters = human_parameters(details.get("parameter_size", ""))
        if not parameters and info.get("general.parameter_count"):
            parameters = human_parameters(str(info["general.parameter_count"]))

        found = {
            "capabilities": list(payload.get("capabilities") or []),
            "parameters": parameters,
            "family": details.get("family", "") or details.get("families", "") or "",
            "voices": _voices_from_payload(payload),
            # ``thinking`` is Ollama's per-model control block:
            #   {"values": [false, true], "default": true}
            #   {"values": ["low", "high", "max"], "default": "max"}
            # ``values: [false]`` means the model cannot think; a missing block
            # means it may still think on its own, which is why the marker guard
            # in loma.text exists.
            "thinking": payload.get("thinking"),
        }
        self._detail_cache[model] = found
        return found

    def capabilities(self, model: str) -> list[str]:
        """A model's capabilities: completion, vision, tools, thinking, speech…

        An empty list means "unknown", never "unusable" — callers treat it as
        unproven rather than filtering the model out.
        """
        return self.model_info(model)["capabilities"]

    def voices(self, model: str | None = None) -> list[str]:
        """Discover a speech model's voices.

        Ollama 0.40.9+ lists them in ``/api/show`` under ``voices`` (the same
        payload that reports ``capabilities: ["speech"]``). On older builds that
        field is missing, so we fall back to the documented behaviour of an
        unknown voice: a 400 whose message lists the ones that do exist.
        """
        model = model or self.settings.tts_model

        listed = self.model_info(model)["voices"]
        if listed:
            return listed
        return self._voices_from_error(model)

    def _voices_from_error(self, model: str) -> list[str]:
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
            # The server accepts arbitrary voices; there is nothing to enumerate.
            return []
        try:
            message = str((response.json().get("error") or {}).get("message", ""))
        except ValueError:
            message = response.text
        marker = "available voices:"
        if marker in message:
            listed = message.split(marker, 1)[1]
            return _ordered_voices(
                [voice.strip() for voice in listed.split(",") if voice.strip()]
            )
        return []

    def supports_structured_outputs(self) -> bool:
        """Can this model be constrained to a JSON schema?

        Per Ollama's docs, the cloud does not support ``format``. A cloud model is
        identified by its tag rather than probed, because probing means sending a
        request that is guaranteed to fail.
        """
        return ":cloud" not in self.settings.llm_model.lower()

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
        think: bool | str = False,
        max_tokens: int | None = None,
        on_usage: Callable[[dict[str, int]], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        response_format: dict[str, Any] | str | None = None,
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
                # Thinking tokens are billed against the same budget, so asking
                # for reasoning has to buy room for it or the answer truncates.
                "num_predict": (max_tokens or self.settings.max_tokens)
                + (THINKING_ALLOWANCE if think else 0),
                "temperature": self.settings.temperature,
            },
        }
        if response_format is not None:
            if self.supports_structured_outputs():
                payload["format"] = response_format
            else:
                # Ollama's cloud does not implement `format`; asking anyway fails
                # the whole turn, so drop the constraint and keep the answer.
                log.warning(
                    "structured output requested for %s, which does not support it",
                    self.settings.llm_model,
                )

        with self._http.stream("POST", "/api/chat", json=payload) as response:
            _raise_for_status(response, streamed=True)
            for line in response.iter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if error := chunk.get("error"):
                    raise OllamaError(str(error))
                message = chunk.get("message") or {}
                # Reasoning and answer are separate fields. Thinking is never
                # yielded as content: LOMA speaks content sentence by sentence.
                reasoning = message.get("thinking")
                if reasoning and on_thinking is not None:
                    on_thinking(reasoning)
                content = message.get("content")
                if content:
                    yield content
                if chunk.get("done"):
                    if on_usage is not None:
                        on_usage(
                            {
                                "prompt_tokens": int(chunk.get("prompt_eval_count") or 0),
                                "completion_tokens": int(chunk.get("eval_count") or 0),
                                # "length" means generation stopped at the token
                                # budget, not at the end of a sentence. Without
                                # this an over-budget turn looks like a model that
                                # simply had nothing to say.
                                "done_reason": str(chunk.get("done_reason") or ""),
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
