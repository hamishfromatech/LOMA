"""Flask API for LOMA.

Endpoints
---------
GET  /api/status          models, memory, context, tools
GET  /api/tools           the MCP toolset
POST /api/chat            {"message": "..."} -> Server-Sent Events
POST /api/transcribe      multipart audio -> {"text": "..."}
POST /api/speak           {"text": "..."} -> audio/wav
POST /api/reset           clear the conversation

The chat endpoint streams the same events the CLI consumes, so the browser sees
the route decision, the tool call, the token stream and one spoken sentence at a
time — no second implementation of a turn.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shlex
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import HTTPException

from server.images import MAX_BODY_BYTES, read_images

from .runtime import get_runtime, peek_runtime

log = logging.getLogger(__name__)

WEB_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000"}

#: The only host names this server will answer to.
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

#: Audio containers the transcriber (ffmpeg under faster-whisper) can be relied
#: on to open. A filename comes from the client, and letting it be anything
#: meant an absurd suffix reached ``mkstemp`` and raised ``OSError``.
AUDIO_SUFFIXES = {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".mp4"}


def _host_is_local(header_value: str) -> bool:
    """True for a loopback host, with or without a port."""
    host = (header_value or "").strip().lower()
    if host.startswith("["):
        # Bracketed IPv6 literal, "[::1]:5050".
        return host.partition("]")[0].lstrip("[") in LOCAL_HOSTS
    return host.rpartition(":")[0] in LOCAL_HOSTS


def configure_logging() -> None:
    """Set up logging. Called by :func:`serve`, *not* by ``create_app``.

    Configuring the root logger at import time hijacks the logging of anything
    that imports this module — tests and any embedder included.
    """
    logging.basicConfig(
        level=os.getenv("LOMA_LOG_LEVEL", "INFO"),
        format="%(levelname)s %(name)s: %(message)s",
    )


def create_app() -> Flask:
    app = Flask(__name__)
    # Derived from the image limits rather than guessed at them. Four 8 MB
    # images are 42.7 MB once base64'd, so the old flat 32 MB made the documented
    # "up to 4 images" unreachable — Flask rejected the request before the
    # validator that would have given a readable error ever ran.
    app.config["MAX_CONTENT_LENGTH"] = max(32 * 1024 * 1024, MAX_BODY_BYTES)

    # -- who may talk to this server ----------------------------------
    # Binding to 127.0.0.1 is not the boundary people think it is. A page on the
    # internet can point its DNS at 127.0.0.1 (rebinding), and the browser then
    # treats http://evil.com as same-origin with the app: no CORS, no preflight,
    # and the Host header is whatever the attacker chose. From there it can read
    # /api/memory and /api/conversations, and — because POST /api/mcp/servers
    # writes a command that is then *executed* on reload — run arbitrary code.
    # So: only answer requests addressed to loopback, and only accept an Origin
    # that is itself loopback (that covers the Next dev server on :3000 and this
    # server's own origin, and rules out any site on the internet).
    @app.before_request
    def guard_loopback() -> Response | None:
        if not _host_is_local(request.host):
            log.warning("refused request with Host %r", request.host)
            return jsonify({"error": "this server only answers on localhost"}), 403

        origin = request.headers.get("Origin", "")
        if origin and not _host_is_local(urlparse(origin).hostname or ""):
            log.warning("refused request from Origin %r", origin)
            return jsonify({"error": f"origin not allowed: {origin}"}), 403
        return None

    # -- errors as JSON ------------------------------------------------
    # Every route here is consumed by ``response.json()``. A Flask HTML error
    # page arrives as "SyntaxError: Unexpected token '<'" in the UI, which is
    # how a perfectly clear "your body is too large" or "no such conversation"
    # becomes a mystery. Note that a failure *after* the SSE body has started
    # cannot change the status; those are reported as an ``error`` event.
    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> Response:
        payload = {"error": exc.description or exc.name}
        if exc.status_code == 413:
            limit = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
            payload["error"] = f"request body too large (limit {limit:.0f} MB)"
        return jsonify(payload), exc.status_code

    @app.errorhandler(Exception)
    def unexpected_error(exc: Exception) -> Response:
        # Logged in full, described in general: Ollama's messages and our own
        # tracebacks quote URLs and absolute paths, and they were being printed
        # verbatim into the chat bubble.
        log.exception("unhandled error on %s %s", request.method, request.path)
        return jsonify({"error": "the server could not complete that request"}), 500

    @app.after_request
    def cors(response: Response) -> Response:
        origin = request.headers.get("Origin", "")
        if origin in WEB_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.route("/api/<path:_any>", methods=["OPTIONS"])
    def preflight(_any: str) -> Response:
        return Response(status=204)

    # -- reads ---------------------------------------------------------
    @app.get("/api/status")
    def status() -> Response:
        return jsonify(get_runtime().status())

    @app.get("/api/tools")
    def tools() -> Response:
        runtime = get_runtime()
        return jsonify(
            [
                {"name": t.name, "description": t.short_description, "required": t.required}
                for t in runtime.pipeline.hub.tools
            ]
        )

    # -- conversations -------------------------------------------------
    # There is no /api/history: "the current conversation" is not a thing the
    # server knows. The client names the conversation it means, and reads it with
    # GET /api/conversations/<id>. The route that used to be here called
    # Runtime.history(), which has never existed — every request to it was a 500.
    @app.get("/api/conversations")
    def list_conversations() -> Response:
        return jsonify(get_runtime().list_conversations())

    @app.post("/api/conversations")
    def create_conversation() -> Response:
        payload = request.get_json(silent=True) or {}
        return jsonify(get_runtime().new_conversation(payload.get("title", "New chat"))), 201

    @app.get("/api/conversations/<conversation_id>")
    def get_conversation(conversation_id: str) -> Response:
        conversation = get_runtime().get_conversation(conversation_id)
        if conversation is None:
            return jsonify({"error": "no such conversation"}), 404
        return jsonify(conversation)

    @app.patch("/api/conversations/<conversation_id>")
    def patch_conversation(conversation_id: str) -> Response:
        payload = request.get_json(silent=True) or {}
        title = str(payload.get("title", "")).strip()
        if not title:
            return jsonify({"error": "title is required"}), 400
        conversation = get_runtime().rename_conversation(conversation_id, title)
        if conversation is None:
            return jsonify({"error": "no such conversation"}), 404
        return jsonify(conversation)

    @app.delete("/api/conversations/<conversation_id>")
    def delete_conversation(conversation_id: str) -> Response:
        if not get_runtime().delete_conversation(conversation_id):
            return jsonify({"error": "no such conversation"}), 404
        return jsonify({"ok": True})

    # -- the turn ------------------------------------------------------
    @app.post("/api/conversations/<conversation_id>/chat")
    def chat_in_conversation(conversation_id: str) -> Response:
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"error": "message is required"}), 400
        # Absent means "use the stored preference" rather than "yes".
        speak = payload.get("speak")
        speak = None if speak is None else bool(speak)
        images, problem = read_images(payload)
        if problem:
            return jsonify({"error": problem}), 400
        return _stream_turn(get_runtime(), conversation_id, message, speak, images)

    @app.post("/api/chat")
    def chat() -> Response:
        """Convenience endpoint: the newest conversation, or a new one."""
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"error": "message is required"}), 400
        speak = payload.get("speak")
        speak = None if speak is None else bool(speak)
        images, problem = read_images(payload)
        if problem:
            return jsonify({"error": problem}), 400

        runtime = get_runtime()
        existing = runtime.list_conversations()
        conversation_id = (
            str(payload.get("conversation_id") or "")
            or (existing[0]["id"] if existing else runtime.new_conversation()["id"])
        )
        return _stream_turn(runtime, conversation_id, message, speak, images)

    # -- first run -----------------------------------------------------
    @app.get("/api/onboarding")
    def get_onboarding() -> Response:
        return jsonify(get_runtime().onboarding_payload())

    @app.post("/api/onboarding")
    def post_onboarding() -> Response:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "expected an object of answers"}), 400
        return jsonify(get_runtime().complete_onboarding(payload))

    @app.post("/api/onboarding/skip")
    def skip_onboarding() -> Response:
        return jsonify(get_runtime().skip_onboarding())

    @app.post("/api/onboarding/reset")
    def reset_onboarding() -> Response:
        return jsonify(get_runtime().reset_onboarding())

    # -- memory --------------------------------------------------------
    @app.get("/api/memory")
    def get_memory() -> Response:
        return jsonify(get_runtime().memory_payload())

    @app.put("/api/memory")
    def put_memory() -> Response:
        payload = request.get_json(silent=True) or {}
        content = payload.get("content")
        if not isinstance(content, str):
            return jsonify({"error": "content is required"}), 400
        return jsonify(get_runtime().write_memory(content))

    @app.delete("/api/memory")
    def delete_memory() -> Response:
        return jsonify(get_runtime().clear_memory())

    # -- settings ------------------------------------------------------
    @app.get("/api/models")
    def list_models() -> Response:
        """What the picker should offer, straight from Ollama's own listing."""
        return jsonify(get_runtime().models_payload())

    @app.get("/api/settings")
    def get_settings() -> Response:
        return jsonify(get_runtime().settings_payload())

    @app.post("/api/settings")
    def post_settings() -> Response:
        payload = request.get_json(silent=True) or {}
        try:
            updated = get_runtime().update_settings(
                llm_model=payload.get("llm_model"),
                vlm_model=payload.get("vlm_model"),
                tts_voice=payload.get("tts_voice"),
                temperature=payload.get("temperature"),
                speak_replies=payload.get("speak_replies"),
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("could not update settings")
            return jsonify({"error": str(exc)}), 400
        return jsonify(updated)

    # -- MCP -----------------------------------------------------------
    @app.get("/api/mcp")
    def get_mcp() -> Response:
        return jsonify(get_runtime().mcp_overview())

    @app.post("/api/mcp/reload")
    def post_mcp_reload() -> Response:
        try:
            return jsonify(get_runtime().reload_mcp())
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/mcp/servers")
    def post_mcp_server() -> Response:
        payload = request.get_json(silent=True) or {}
        name = str(payload.get("name", "")).strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        if not all(char.isalnum() or char in "-_." for char in name):
            return jsonify({"error": "use letters, digits, dash, underscore or dot"}), 400

        url = str(payload.get("url", "")).strip()
        command = str(payload.get("command", "")).strip()
        raw_args = payload.get("args", [])
        raw_env = payload.get("env", {})

        spec: dict = {}
        if url:
            spec["url"] = url
        elif command:
            spec["command"] = command
            if isinstance(raw_args, str):
                spec["args"] = shlex.split(raw_args)
            elif isinstance(raw_args, list):
                spec["args"] = [str(item) for item in raw_args]
        else:
            return jsonify({"error": "a command or a url is required"}), 400

        if isinstance(raw_env, str) and raw_env.strip():
            try:
                raw_env = json.loads(raw_env)
            except ValueError:
                return jsonify({"error": "env must be a JSON object"}), 400
        if isinstance(raw_env, dict) and raw_env:
            spec["env"] = {str(k): str(v) for k, v in raw_env.items()}

        try:
            overview = get_runtime().add_mcp_server(name, spec)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400
        return jsonify(overview), 201

    @app.delete("/api/mcp/servers/<name>")
    def delete_mcp_server(name: str) -> Response:
        try:
            return jsonify(get_runtime().remove_mcp_server(name))
        except KeyError:
            return jsonify({"error": "no such server"}), 404
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400

    # -- speech --------------------------------------------------------
    @app.post("/api/transcribe")
    def transcribe() -> Response:
        upload = request.files.get("audio")
        if upload is None:
            return jsonify({"error": "audio file is required"}), 400

        suffix = (Path(upload.filename or "clip.webm").suffix or ".webm").lower()
        if suffix not in AUDIO_SUFFIXES:
            return jsonify({"error": f"unsupported audio type {suffix!r}"}), 400
        handle, path = tempfile.mkstemp(suffix=suffix)
        os.close(handle)
        try:
            upload.save(path)
            runtime = get_runtime()
            runtime.warm()
            transcript = runtime.pipeline.transcribe_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

        return jsonify(
            {"text": transcript.text, "language": transcript.language, "duration": transcript.duration}
        )

    @app.post("/api/speak")
    def speak() -> Response:
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        runtime = get_runtime()
        voice = payload.get("voice") or runtime.settings.tts_voice
        audio = runtime.pipeline.client.speech_wav(text[:4000], voice=voice)
        return Response(
            audio,
            mimetype="audio/wav",
            headers={"Cache-Control": "no-store"},
        )

    # -- state ---------------------------------------------------------
    @app.post("/api/reset")
    def reset() -> Response:
        get_runtime().reset()
        return jsonify({"ok": True})

    @app.post("/api/conversations/<id>/stop")
    def stop_turn(id: str) -> Response:
        """Stop the turn in this conversation.

        The browser sends this when the user clicks the stop button. The
        old ``stopTurn`` only called ``controller.abort()`` on the SSE
        reader — the model kept generating into a closed socket, holding
        the pipeline lock and GPU for the entire answer. Now the server
        receives a stop signal, the worker breaks from the pipeline loop,
        and ``Agent.finish_partial()`` keeps whatever text was streamed
        into the conversation record.
        """
        runtime = get_runtime()
        signaled = runtime.stop_turn(id)
        return jsonify({"ok": True, "signaled": signaled})

    @app.get("/healthz")
    def healthz() -> tuple[Response, int]:
        """Whether LOMA can actually answer, not whether the process is alive.

        The old unconditional ``{"ok": true}`` told a supervisor everything was
        fine while Ollama was down and every turn was failing.
        """
        runtime = get_runtime()
        try:
            version = runtime.pipeline.client.health().get("version", "")
        except Exception as exc:  # noqa: BLE001 - health must always answer
            return jsonify({"ok": False, "ollama": str(exc)}), 503
        return (
            jsonify(
                {
                    "ok": True,
                    "ollama": version,
                    "warmed": runtime.ready,
                    "tools": len(runtime.pipeline.hub.tools),
                }
            ),
            200,
        )

    return app


def serve() -> None:
    """Run the API. This is what ``python -m server.app`` and ``make api`` do."""
    configure_logging()

    # Ctrl-C on `make api` used to leave every MCP server process LOMA had
    # spawned behind, because Runtime.close() had no callers. atexit covers the
    # clean exit and the interpreter-shutting-down case alike.
    atexit.register(_shutdown)

    port = int(os.getenv("LOMA_PORT", "5050"))
    create_app().run(host="127.0.0.1", port=port, threaded=True, debug=False)


def _shutdown() -> None:
    runtime = peek_runtime()
    if runtime is None:
        return
    try:
        runtime.close()
    except Exception as exc:  # noqa: BLE001 - shutdown must never raise
        log.debug("runtime shutdown: %s", exc)


def _stream_turn(
    runtime,
    conversation_id: str,
    message: str,
    speak: bool | None,
    images: list[str] | None = None,
) -> Response:
    """Turn a conversation turn into an SSE response."""

    @stream_with_context
    def events():
        try:
            runtime.warm()
            for event in runtime.chat(conversation_id, message, speak=speak, images=images):
                yield sse(event)
        except KeyError:
            yield sse({"type": "error", "message": "no such conversation"})
        except Exception as exc:  # noqa: BLE001 - never kill the stream silently
            log.exception("chat turn failed")
            yield sse({"type": "error", "message": str(exc)})

    return Response(
        events(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


if __name__ == "__main__":
    serve()
