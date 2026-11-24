"""One turn chooses one chat model, and the pipeline uses it.

Constructs a real VoicePipeline (models load lazily) and replaces the two
network-facing pieces: the client and the model manager's view of it.
"""

from __future__ import annotations

from loma.agent import Agent
from loma.config import Settings
from loma.model_manager import ModelManager
from loma.pipeline import VoicePipeline


class FakeClient:
    def __init__(self, vision: bool = True) -> None:
        self.wants_vision = vision
        self.seen: list[dict] = []
        self.unloaded: list[str] = []
        self.prepared: list[str] = []

    def chat_stream(self, messages, *, think=False, on_usage=None, **kwargs):
        self.seen.append(kwargs)
        yield "a red square"
        if on_usage is not None:
            on_usage({"prompt_tokens": 12, "completion_tokens": 3, "done_reason": "stop"})

    def unload(self, model: str) -> None:
        self.unloaded.append(model)

    def capabilities(self, model: str) -> list[str]:
        return ["completion", "vision"] if self.wants_vision else ["completion"]

    def context_length(self, model: str) -> int:
        return {"lfm3:latest": 262_144, "lfm3-vl:2b": 128_000}.get(model, 0)

    def close(self) -> None:
        pass


def build(vision: bool = True):
    settings = Settings(
        llm_model="lfm3:latest",
        vlm_model="lfm3-vl:2b",
        dynamic_models=True,
        mcp_enabled=False,
        skills_enabled=False,
        num_ctx=262_144,
    )
    pipeline = VoicePipeline(settings)
    pipeline.client.close()
    client = FakeClient(vision=vision)
    pipeline.client = client
    pipeline.agent = Agent(client, settings)
    pipeline.models = ModelManager(client, settings)
    return pipeline, client


def run(pipeline, **kwargs):
    return list(
        pipeline.stream_events(
            "what is this?",
            speak=False,
            route=False,
            use_tools=False,
            **kwargs,
        )
    )


def test_an_image_turn_swaps_to_the_vision_model():
    pipeline, client = build()

    events = run(pipeline, images=["aGVsbG8="])

    done = next(event for event in events if event["type"] == "done")
    assert done["model"] == "lfm3-vl:2b"
    assert client.seen[-1]["model"] == "lfm3-vl:2b"
    # The VLM's own ceiling, not the machine-wide 256k.
    assert client.seen[-1]["num_ctx"] == 128_000
    assert pipeline.models.active == "lfm3-vl:2b"


def test_a_text_turn_swaps_back_and_unloads_the_vision_model():
    pipeline, client = build()

    run(pipeline, images=["aGVsbG8="])
    events = run(pipeline)

    done = next(event for event in events if event["type"] == "done")
    assert done["model"] == "lfm3:latest"
    assert client.seen[-1]["model"] == "lfm3:latest"
    assert client.unloaded == ["lfm3-vl:2b"]
    assert pipeline.models.swaps == 1


def test_an_unreadable_image_stops_before_the_request():
    pipeline, client = build(vision=False)

    events = run(pipeline, images=["aGVsbG8="])

    assert events[0]["type"] == "error"
    assert "cannot read images" in events[0]["message"]
    assert not any(event["type"] == "done" for event in events)
    assert client.seen == []
