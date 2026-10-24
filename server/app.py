"""Flask API for LOMA.

Endpoints
---------
GET  /api/status          models, memory, context, tools
GET  /api/tools           the MCP toolset
GET  /api/history         the current conversation
POST /api/chat            {"message": "..."} -> Server-Sent Events
POST /api/transcribe      multipart audio -> {"text": "..."}
POST /api/speak           {"text": "..."} -> audio/wav
POST /api/reset           clear the conversation

The chat endpoint streams the same events the CLI consumes, so the browser sees
the route decision, the tool call, the token stream and one spoken sentence at a
time — no second implementation of a turn.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import tempfile
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context

from .runtime import get_runtime

log = logging.getLogger(__name__)

WEB_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000"}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB of audio

    logging.basicConfig(
        level=os.getenv("LOMA_LOG_LEVEL", "INFO"),
        format="%(levelname)s %(name)s: %(message)s",
    )

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

    @app.get("/api/history")
    def history() -> Response:
        return jsonify(get_runtime().history())

    # -- conversations -------------------------------------------------
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
        speak = bool(payload.get("speak", True))
        images, problem = _read_images(payload)
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
        speak = bool(payload.get("speak", True))
        images, problem = _read_images(payload)
        if problem:
            return jsonify({"error": problem}), 400

        runtime = get_runtime()
        existing = runtime.list_conversations()
        conversation_id = (
            str(payload.get("conversation_id") or "")
            or (existing[0]["id"] if existing else runtime.new_conversation()["id"])
        )
        return _stream_turn(runtime, conversation_id, message, speak, images)

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
                tts_voice=payload.get("tts_voice"),
                temperature=payload.get("temperature"),
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

        suffix = Path(upload.filename or "clip.webm").suffix or ".webm"
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

    @app.get("/healthz")
    def healthz() -> Response:
        return jsonify({"ok": True})

    return app


MAX_IMAGES = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _read_images(payload: dict) -> tuple[list[str], str | None]:
    """Validate the ``images`` field: base64 only, at most four, size-capped.

    The UI sends data URLs ("data:image/png;base64,…") because that is what a
    FileReader gives you, and Ollama wants the bare base64 payload.
    """
    raw = payload.get("images")
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return [], "images must be a list of base64 strings"
    if len(raw) > MAX_IMAGES:
        return [], f"at most {MAX_IMAGES} images per turn"

    cleaned: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item:
            return [], f"image {index} is not a base64 string"
        value = item.strip()
        if value.startswith("data:"):
            _, _, value = value.partition(",")
        value = "".join(value.split())
        if not value:
            return [], f"image {index} is empty"
        # 4/3 is the base64 expansion; this catches an oversized upload before
        # it reaches the model rather than after a 20-second wait.
        if len(value) * 3 / 4 > MAX_IMAGE_BYTES:
            return [], f"image {index} is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB"
        cleaned.append(value)
    return cleaned, None


def _stream_turn(
    runtime,
    conversation_id: str,
    message: str,
    speak: bool,
    images: list[str] | None = None,
) -> Response:
    """Turn a conversation turn into an SSE response."""

    @stream_with_context
    def events():
        try:
            runtime.warm()
            for event in runtime.chat(
                conversation_id, message, speak=speak, images=images
            ):
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


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("LOMA_PORT", "5050"))
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
