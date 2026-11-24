"""Two turns on the same conversation must both survive.

Runtime.chat used to read the conversation *before* taking the turn lock. Two
overlapping requests therefore both started from the same old copy, and the
second save erased the first turn. The read, the turn and the write now happen
inside the lock, and the file itself is written atomically.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from loma.agent import Agent
from loma.config import Settings
from server.runtime import Runtime


class FakeClient:
    def chat_stream(self, messages, *, think=False, on_usage=None, **_):
        yield "ok"
        if on_usage is not None:
            on_usage({"prompt_tokens": 5, "completion_tokens": 1, "done_reason": "stop"})

    def chat_json(self, *args, **kwargs):  # memory extraction is disabled in these tests
        raise AssertionError("memory extraction should not run")


class SlowPipeline:
    """A turn that takes long enough for two requests to overlap."""

    def __init__(self) -> None:
        self.client = FakeClient()
        self.agent = Agent(self.client, Settings())
        self.last_reply = ""
        self.memory = None
        self.models = SimpleNamespace(active=None)

    def stream_events(self, message, **_kwargs):
        time.sleep(0.15)
        for delta in self.agent.respond(message):
            yield {"type": "delta", "text": delta}
        reply = self.agent.transcript()[-1]["content"]
        self.last_reply = reply
        yield {
            "type": "done",
            "reply": reply,
            "intent": "answer",
            "tool": [],
            "tool_confidence": 0.0,
            "timings": {"route_ms": 0, "tool_ms": 0, "first_token_s": 0,
                        "first_audio_s": 0, "total_s": 0},
            "context_tokens": 5,
        }


def test_overlapping_turns_are_both_persisted(tmp_path):
    runtime = Runtime(
        Settings(
            data_dir=tmp_path / "data",
            memory_enabled=False,
            mcp_enabled=False,
            skills_enabled=False,
        )
    )
    runtime.pipeline = SlowPipeline()
    conversation = runtime.new_conversation()
    conversation_id = conversation["id"]

    threads = [
        threading.Thread(target=lambda text=text: list(runtime.chat(conversation_id, text)))
        for text in ("alpha", "beta")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    path = tmp_path / "data" / "conversations" / f"{conversation_id}.json"
    saved = json.loads(path.read_text())

    users = [m["content"] for m in saved["messages"] if m["role"] == "user"]
    assert sorted(users) == ["alpha", "beta"]
    assert len([m for m in saved["messages"] if m["role"] == "assistant"]) == 2

    # Atomic write: no temporary sibling is left behind.
    assert not list(path.parent.glob("*.tmp"))
