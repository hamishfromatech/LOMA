"""The turn's wiring: Needle triages, the LLM calls, MCP executes.

The pipeline is real; the client, hub and router are stubs so the test needs no
Ollama and no subprocess.
"""

from __future__ import annotations

from loma.agent import Agent
from loma.config import Settings
from loma.mcp_hub import MCPTool
from loma.model_manager import ModelManager
from loma.pipeline import VoicePipeline
from loma.tool_router import ToolTriage

TOOL = MCPTool(
    name="roll_die",
    description="Roll a die.",
    input_schema={
        "type": "object",
        "properties": {"sides": {"type": "integer"}},
        "required": [],
    },
)
CALL = {
    "id": "call_1",
    "function": {"index": 0, "name": "roll_die", "arguments": {"sides": 20}},
}


class FakeClient:
    def __init__(self, call: bool = True) -> None:
        self.call = call
        self.rounds: list[dict] = []

    def chat_stream(self, messages, *, tools=None, on_usage=None, on_tool_calls=None, **_):
        self.rounds.append({"tools": tools, "messages": [dict(m) for m in messages]})
        if self.call and len(self.rounds) == 1:
            if on_tool_calls is not None:
                on_tool_calls([CALL])
            yield ""
        else:
            yield "You rolled a 14."
        if on_usage is not None:
            on_usage({"prompt_tokens": 30, "completion_tokens": 5, "done_reason": "stop"})

    def unload(self, model):
        pass

    def context_length(self, model):
        return 262_144

    def capabilities(self, model):
        return ["completion", "tools"]

    def close(self):
        pass


class FakeHub:
    def __init__(self) -> None:
        self.tools = [TOOL]
        self.calls: list[tuple[str, dict]] = []
        self.available = True

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return '{"roll": 14, "sides": 20}'


class FakeRouter:
    ready = True

    def __init__(self, triage: ToolTriage) -> None:
        self.triage = triage

    def classify(self, text):
        return self.triage

    def load(self, tools):
        return True

    def close(self):
        pass


def build(triage: ToolTriage, *, call: bool = True):
    settings = Settings(
        mcp_enabled=True,
        skills_enabled=False,
        triage_enabled=True,
        tool_rounds=1,
    )
    pipeline = VoicePipeline(settings)
    pipeline.client.close()
    client = FakeClient(call=call)
    pipeline.client = client
    pipeline.agent = Agent(client, settings)
    pipeline.models = ModelManager(client, settings)
    pipeline.hub = FakeHub()
    pipeline.tool_router = FakeRouter(triage)
    pipeline.start_tools = lambda: True  # type: ignore[method-assign]
    return pipeline, client


def test_triage_sends_schemas_and_the_model_calls_the_tool():
    pipeline, client = build(
        ToolTriage(needs_tools=True, candidates=["roll_die"], confidence=0.8)
    )

    events = list(pipeline.stream_events("roll a d20", speak=False))

    assert client.rounds[0]["tools"] is not None
    assert client.rounds[0]["tools"][0]["function"]["name"] == "roll_die"
    assert pipeline.hub.calls == [("roll_die", {"sides": 20})]

    kinds = [event["type"] for event in events]
    assert "tool" in kinds and "tool_result" in kinds
    done = next(event for event in events if event["type"] == "done")
    assert done["tool"] == ["roll_die"]
    assert done["tool_calls"] == [{"name": "roll_die", "arguments": {"sides": 20}}]
    assert done["reply"] == "You rolled a 14."


def test_a_no_tool_triage_offers_no_schemas():
    pipeline, client = build(ToolTriage(needs_tools=False, confidence=0.03))

    events = list(pipeline.stream_events("tell me a joke", speak=False))

    assert client.rounds[0]["tools"] is None
    done = next(event for event in events if event["type"] == "done")
    assert done["tool"] == []
    assert pipeline.hub.calls == []


def test_shortlisting_is_opt_in():
    second = MCPTool(
        name="calculate",
        description="Evaluate arithmetic.",
        input_schema={"type": "object", "properties": {"expression": {"type": "string"}}},
    )
    pipeline, client = build(
        ToolTriage(needs_tools=True, candidates=["calculate"], confidence=0.8),
        call=False,
    )
    pipeline.settings = pipeline.settings.with_(tool_shortlist=True)
    pipeline.hub.tools = [TOOL, second]

    list(pipeline.stream_events("17 times 24", speak=False))

    offered = client.rounds[0]["tools"]
    assert [spec["function"]["name"] for spec in offered] == ["calculate"]


def test_the_default_offers_every_tool():
    second = MCPTool(
        name="calculate",
        description="Evaluate arithmetic.",
        input_schema={"type": "object", "properties": {"expression": {"type": "string"}}},
    )
    pipeline, client = build(
        ToolTriage(needs_tools=True, candidates=["calculate"], confidence=0.8),
        call=False,
    )
    pipeline.hub.tools = [TOOL, second]

    list(pipeline.stream_events("17 times 24", speak=False))

    offered = client.rounds[0]["tools"]
    assert {spec["function"]["name"] for spec in offered} == {"roll_die", "calculate"}
