"""Model and voice discovery against Ollama's listings.

No network and no Ollama: the HTTP client is replaced by a stub that answers
from a dict, which is exactly the shape the two documented endpoints return.
"""

from __future__ import annotations

import httpx
import pytest

from loma.config import Settings
from loma.ollama_client import OllamaClient, _ordered_voices


class Response:
    def __init__(self, payload=None, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class StubHTTP:
    """Routes are keyed by (method, path). Anything unlisted raises ConnectError."""

    def __init__(self, routes: dict[tuple[str, str], Response]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str]] = []

    def get(self, path: str, **_kwargs):
        self.calls.append(("GET", path))
        if ("GET", path) not in self.routes:
            raise httpx.ConnectError("offline")
        return self.routes[("GET", path)]

    def post(self, path: str, **_kwargs):
        self.calls.append(("POST", path))
        if ("POST", path) not in self.routes:
            raise httpx.ConnectError("offline")
        return self.routes[("POST", path)]


def make_client(routes) -> tuple[OllamaClient, StubHTTP]:
    client = OllamaClient(Settings())
    stub = StubHTTP(routes)
    client._http = stub  # type: ignore[assignment]
    return client, stub


TAGS = {
    "models": [
        {
            "name": "qwen3.5:0.8b",
            "size": 1_493_000_000,
            "modified_at": "2026-10-20T10:00:00Z",
            "details": {
                "family": "qwen3.5",
                "parameter_size": "0.8B",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "phonon",
            "size": 121_000_000,
            "details": {"family": "phonon", "parameter_size": "31M"},
        },
    ]
}


def show(capabilities: list[str], voices=None) -> Response:
    payload = {"capabilities": capabilities}
    if voices is not None:
        payload["voices"] = voices
    return Response(payload)


class TestListModels:
    def test_reads_the_native_listing(self):
        client, _ = make_client(
            {
                ("GET", "/api/tags"): Response(TAGS),
                ("POST", "/api/show"): show(["completion"]),
            }
        )
        models = client.list_models()
        assert [model["name"] for model in models] == ["phonon", "qwen3.5:0.8b"]
        # Smallest first, and the size is reported in MB.
        assert models[0]["size_mb"] == pytest.approx(121.0, rel=0.01)
        assert models[1]["quantization"] == "Q4_K_M"
        assert models[1]["source"] == "api/tags"

    def test_falls_back_to_the_openai_listing(self):
        client, stub = make_client(
            {("GET", "/v1/models"): Response({"data": [{"id": "llama3.2:3b"}]})}
        )
        models = client.list_models()
        assert [model["name"] for model in models] == ["llama3.2:3b"]
        assert models[0]["source"] == "v1/models"
        assert ("GET", "/api/tags") in stub.calls

    def test_unknown_capabilities_are_kept(self):
        """A failed /api/show means unknown, never "cannot chat"."""
        client, _ = make_client({("GET", "/api/tags"): Response(TAGS)})
        models = client.list_models()
        assert len(models) == 2
        assert all(model["capabilities"] == [] for model in models)

    def test_capabilities_are_cached(self):
        client, stub = make_client(
            {
                ("GET", "/api/tags"): Response({"models": [TAGS["models"][0]]}),
                ("POST", "/api/show"): show(["completion", "tools"]),
            }
        )
        client.list_models()
        client.list_models()
        assert stub.calls.count(("POST", "/api/show")) == 1


class TestVoices:
    def test_reads_voices_from_api_show(self):
        client, stub = make_client({("POST", "/api/show"): show(["speech"], ["Zoey", "default"])})
        assert client.voices("phonon") == ["default", "Zoey"]
        # The 400-error probe is only a fallback.
        assert stub.calls == [("POST", "/api/show")]

    def test_accepts_voice_objects(self):
        client, _ = make_client(
            {("POST", "/api/show"): show(["speech"], [{"name": "Easton"}, {"id": "Freya"}])}
        )
        assert client.voices("phonon") == ["Easton", "Freya"]

    def test_falls_back_to_the_error_message(self):
        client, _ = make_client(
            {
                ("POST", "/api/show"): Response({}, status_code=404),
                ("POST", "/v1/audio/speech"): Response(
                    {"error": {"message": "invalid voice; available voices: Zoey, Easton"}},
                    status_code=400,
                ),
            }
        )
        assert client.voices("phonon") == ["Easton", "Zoey"]

    def test_returns_nothing_when_nothing_is_listed(self):
        client, _ = make_client({("POST", "/api/show"): show(["speech"])})
        assert client.voices("phonon") == []

    def test_ordering_dedupes_case_insensitively(self):
        assert _ordered_voices(["Zoey", "zoey", "default", "Easton"]) == [
            "default",
            "Easton",
            "Zoey",
        ]
