"""Smoke tests for the Flask API layer.

The tests use a stubbed Runtime so they never touch Ollama, the filesystem,
or the network. Each test verifies the route's shape (status code, JSON
response) rather than end-to-end behaviour.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest


def _make_stub(**kw):
    """Build a minimal, JSON-serialisable Runtime stub."""
    rt = MagicMock()
    for attr, value in kw.items():
        setattr(rt, attr, value)
    # Defaults for every route we test
    if not kw.get("hub"):
        rt.pipeline.hub = MagicMock(available=True, tools=[])
    if not kw.get("ready"):
        rt.ready = True
    if not kw.get("models_payload"):
        rt.models_payload.return_value = {"models": [], "current": "test"}
    if not kw.get("mcp_overview"):
        rt.mcp_overview.return_value = {"path": "/test", "connected": False, "tool_count": 0}
    if not kw.get("memory_payload"):
        rt.memory_payload.return_value = {
            "enabled": True,
            "auto_extract": False,
            "path": "/test",
            "content": "",
            "facts": [],
        }
    if not kw.get("settings_payload"):
        rt.settings_payload.return_value = {}
    if not kw.get("list_conversations"):
        rt.list_conversations.return_value = []
    if not kw.get("status"):
        rt.status.return_value = {
            "ready": True,
            "models": {},
            "memory_budget_mb": 0,
            "context": {},
            "tools": [],
            "config": {},
        }
    if not kw.get("pipeline"):
        rt.pipeline = MagicMock()
        rt.pipeline.client = MagicMock()
        rt.pipeline.client.health.return_value = {"version": "0.48"}
        rt.pipeline.client.list_models.return_value = []
    if not kw.get("reset"):
        rt.reset = MagicMock()
    return rt


HOST = {"Host": "127.0.0.1:5050"}

@pytest.fixture
def app():
    """Create a Flask app with a stubbed runtime."""
    with patch("server.app.get_runtime") as mock_gr:
        mock_gr.return_value = _make_stub(
            status=MagicMock(return_value={
                "ready": True,
                "models": {},
                "memory_budget_mb": 0,
                "context": {},
                "tools": [],
                "config": {},
            }),
            pipeline=MagicMock(
                hub=MagicMock(available=True, tools=[]),
                client=MagicMock(
                    health=MagicMock(return_value={"version": "0.48"}),
                    list_models=MagicMock(return_value=[]),
                ),
            ),
            reset=MagicMock(),
            list_conversations=MagicMock(return_value=[]),
            new_conversation=MagicMock(return_value={"id": "test", "title": "Test chat"}),
            delete_conversation=MagicMock(return_value=True),
        )

        from server.app import create_app

        flask_app = create_app()
        flask_app.config["TESTING"] = True

        with flask_app.test_client() as client:
            yield client


@pytest.fixture
def host_headers():
    """Fixture returning the Host header dict for localhost."""
    return HOST


def _get(client, path, host_headers=HOST):
    """GET helper that sets the Host header."""
    return client.get(path, headers=host_headers)


def _post(client, path, json=None, data=None, content_type=None, headers=HOST):
    """POST helper that sets the Host header."""
    kwargs = {"headers": headers}
    if json is not None:
        kwargs["json"] = json
    if data is not None:
        kwargs["data"] = data
    return client.post(path, **kwargs)


class TestStatus:
    def test_status_returns_json(self, app):
        resp = _get(app, "/api/status")
        assert resp.status_code == 200
        assert resp.is_json


class TestHealthz:
    def test_healthz_returns_detail(self, app):
        resp = _get(app, "/healthz")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "ok" in data
        assert "ollama" in data


class TestConversations:
    def test_list(self, app):
        resp = _get(app, "/api/conversations")
        assert resp.status_code == 200
        assert resp.is_json

    def test_create(self, app):
        resp = _post(app, "/api/conversations", json={"title": "Test chat"})
        assert resp.status_code == 201
        assert resp.is_json

    def test_delete(self, app):
        resp = app.delete("/api/conversations/fake-id", headers=HOST)
        assert resp.status_code == 200
        assert resp.is_json


class TestReset:
    def test_reset(self, app):
        resp = _post(app, "/api/reset")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True


class TestHostGuard:
    """The before_request guard only allows localhost hosts."""

    def test_localhost_allowed(self, app):
        resp = _get(app, "/api/status")
        assert resp.status_code == 200
        assert resp.is_json

    def test_loopback_allowed(self, app):
        resp = _get(app, "/api/status", host_headers={"Host": "localhost:5050"})
        assert resp.status_code == 200
        assert resp.is_json

    def test_external_blocked(self, app):
        resp = _get(app, "/api/status", host_headers={"Host": "evil.com"})
        assert resp.status_code == 403
        assert "error" in resp.get_json()

    def test_external_origin_blocked(self, app):
        resp = app.get(
            "/api/status",
            headers={
                "Host": "127.0.0.1:5050",
                "Origin": "http://evil.com",
            },
        )
        assert resp.status_code == 403
        assert "error" in resp.get_json()


class TestTranscribe:
    def test_no_audio_returns_error(self, app):
        resp = app.post("/api/transcribe", headers=HOST)
        assert resp.status_code == 400
        assert resp.is_json

    def test_invalid_suffix_returns_error(self, app):
        data = {"audio": (BytesIO(b"x"), "evil.exe")}
        resp = app.post(
            "/api/transcribe",
            data=data,
            headers={**HOST, "Content-Type": "multipart/form-data"},
        )
        assert resp.status_code == 400
        assert "unsupported" in resp.get_json()["error"].lower()

    def test_valid_suffix_accepted(self, app):
        # Stub the pipeline to return a valid transcript
        with patch("server.app.get_runtime") as mock_gr:
            rt = _make_stub()
            rt.pipeline.transcribe_file.return_value = MagicMock(
                text="test",
                language="en",
                duration=1.0,
            )
            mock_gr.return_value = rt

            data = {"audio": (BytesIO(b"fake audio"), "clip.webm")}
            resp = app.post(
                "/api/transcribe",
                data=data,
                headers={**HOST, "Content-Type": "multipart/form-data"},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert "text" in data
            assert data["text"] == "test"


class TestErrorShape:
    """Routes return JSON payloads instead of HTML pages."""

    def test_nonexistent_route_is_json(self, app):
        # 404 from Flask → our error handler → JSON
        resp = app.get("/api/nonexistent", headers=HOST)
        assert resp.is_json
