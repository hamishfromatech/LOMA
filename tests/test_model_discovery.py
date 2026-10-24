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


class TestMergingTagsAndShow:
    """MLX builds report no family and no parameter size in /api/tags."""

    def test_missing_fields_are_filled_from_api_show(self):
        thin = {
            "models": [
                {
                    "name": "qwen3.5:0.8b",
                    "size": 1_244_131_636,
                    "details": {"family": "", "parameter_size": ""},
                }
            ]
        }
        rich = Response(
            {
                "capabilities": ["completion", "tools"],
                "details": {"family": "qwen3_5", "parameter_size": "852.88M"},
                "model_info": {"general.parameter_count": 852_878_912},
            }
        )
        client, _ = make_client(
            {("GET", "/api/tags"): Response(thin), ("POST", "/api/show"): rich}
        )
        model = client.list_models()[0]
        assert model["family"] == "qwen3_5"
        assert model["parameters"] == "852.9M"
        assert model["capabilities"] == ["completion", "tools"]

    def test_parameter_count_is_the_backstop(self):
        thin = {
            "models": [
                {"name": "cloud:big", "size": 0, "details": {"parameter_size": ""}}
            ]
        }
        rich = Response({"model_info": {"general.parameter_count": 321_323_031_390}})
        client, _ = make_client(
            {("GET", "/api/tags"): Response(thin), ("POST", "/api/show"): rich}
        )
        assert client.list_models()[0]["parameters"] == "321.3B"

    def test_capabilities_in_the_listing_skip_the_probe(self):
        listing = {
            "models": [
                {
                    "name": "qwen3.5:0.8b",
                    "size": 1_200_000_000,
                    "details": {"family": "qwen3_5", "parameter_size": "0.8B"},
                    "capabilities": ["completion"],
                }
            ]
        }
        client, stub = make_client({("GET", "/api/tags"): Response(listing)})
        model = client.list_models()[0]
        assert model["capabilities"] == ["completion"]
        assert model["parameters"] == "0.8B"
        assert ("POST", "/api/show") not in stub.calls


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


class TestThinkingControls:
    """/api/show reports how a model thinks; the picker and the caller both need it."""

    def test_thinking_metadata_is_surfaced(self):
        thin = {"models": [{"name": "qwen3.5:0.8b", "size": 1, "details": {"parameter_size": ""}}]}
        rich = Response(
            {
                "capabilities": ["completion", "thinking"],
                "thinking": {"values": [False, True], "default": True},
            }
        )
        client, _ = make_client(
            {("GET", "/api/tags"): Response(thin), ("POST", "/api/show"): rich}
        )
        model = client.list_models()[0]
        assert model["thinking"] == {"values": [False, True], "default": True}

    def test_a_model_without_thinking_controls_is_null(self):
        thin = {"models": [{"name": "plain", "size": 1, "details": {"parameter_size": "7B"}}]}
        rich = Response({"capabilities": ["completion"]})
        client, _ = make_client(
            {("GET", "/api/tags"): Response(thin), ("POST", "/api/show"): rich}
        )
        assert client.list_models()[0]["thinking"] is None


class StreamStub(StubHTTP):
    """Adds the streaming context manager chat_stream uses."""

    class _Response:
        def __init__(self, lines):
            self._lines = lines
            self.is_success = True

        def iter_lines(self):
            return iter(self._lines)

    def __init__(self, lines):
        super().__init__({})
        self._lines = lines
        self.payload: dict | None = None

    class _Ctx:
        def __init__(self, outer):
            self.outer = outer

        def __enter__(self):
            return StreamStub._Response(self.outer._lines)

        def __exit__(self, *exc):
            return False

    def stream(self, _method, _path, json=None):
        self.payload = json
        return StreamStub._Ctx(self)


def run_stream(client, **kwargs):
    lines = [
        '{"message": {"content": "hello"}, "done": false}',
        '{"message": {"content": ""}, "done": true, "done_reason": "stop"}',
    ]
    stub = StreamStub(lines)
    client._http = stub  # type: ignore[assignment]
    text = "".join(client.chat_stream([{"role": "user", "content": "hi"}], **kwargs))
    return text, stub.payload


class TestStructuredOutputs:
    SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}

    def test_format_is_sent_when_supported(self):
        client, _ = make_client({})
        _, payload = run_stream(client, response_format=self.SCHEMA)
        assert payload is not None and payload["format"] == self.SCHEMA

    def test_no_format_field_without_a_request(self):
        client, _ = make_client({})
        _, payload = run_stream(client)
        assert payload is not None and "format" not in payload

    def test_cloud_models_are_not_constrained(self):
        """The docs are explicit: the cloud does not support `format`."""
        client, _ = make_client({})
        client.settings = Settings(llm_model="glm-5.3-flash:cloud")
        text, payload = run_stream(client, response_format=self.SCHEMA)
        assert payload is not None and "format" not in payload
        assert text == "hello"

    def test_done_reason_is_reported(self):
        client, _ = make_client({})
        usage: dict = {}
        run_stream(client, on_usage=usage.update)
        assert usage["done_reason"] == "stop"
