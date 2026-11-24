"""What the language model is allowed to know about tools.

The answer path never receives a tool schema. It receives a name + one line of
prose, and only when the user asks what LOMA can do. The common forms of that
question are answered deterministically, with no model at all.
"""

from __future__ import annotations

from loma.agent import Agent
from loma.config import Settings
from loma.control import ControlRouter
from loma.mcp_hub import MCPTool, tool_index
from loma.model_manager import ModelManager
from loma.pipeline import VoicePipeline

TOOLS = [
    MCPTool(
        name="local_time",
        description="Return the current date and time in a city or UTC.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
        },
    ),
    MCPTool(
        name="roll_die",
        description="Roll a single die with the given number of sides and return the result.",
        input_schema={
            "type": "object",
            "properties": {"sides": {"type": "integer"}},
        },
    ),
]


class FakeHub:
    def __init__(self, tools):
        self.tools = tools
        self.available = True


class FakeClient:
    def __init__(self):
        self.seen = []

    def chat_stream(self, messages, *, think=False, on_usage=None, **_):
        self.seen.append([dict(message) for message in messages])
        yield "ok"
        if on_usage is not None:
            on_usage({"prompt_tokens": 10, "completion_tokens": 1, "done_reason": "stop"})

    def unload(self, model):
        pass

    def context_length(self, model):
        return 262_144

    def capabilities(self, model):
        return ["completion"]


# -- the deterministic answer ------------------------------------------------


def test_common_capability_phrasings_are_a_control_intent():
    router = ControlRouter()
    for phrase in (
        "what tools do you have?",
        "what can you do?",
        "list your tools",
        "tell me about your capabilities",
        "what tools are available",
    ):
        match = router.match(phrase)
        assert match.intent == "tools", phrase
        assert match.is_fast_action


def test_a_tool_request_is_not_a_capability_question():
    router = ControlRouter()
    assert router.match("what tools should I use to cut wood?").intent == "answer"
    assert router.match("use your tools to check the weather").intent == "answer"


def test_the_reply_is_built_from_the_live_tool_list():
    pipeline = VoicePipeline(Settings(mcp_enabled=False, skills_enabled=False))
    pipeline.hub = FakeHub(TOOLS)
    pipeline.skills = None

    reply = pipeline._tools_reply()

    assert reply.startswith("I have two tools:")
    assert "current date and time" in reply
    assert "roll a single die" in reply
    # Names are for the router; the spoken answer describes capabilities.
    assert "local_time" not in reply
    pipeline.client.close()


def test_no_tools_is_still_a_sentence():
    pipeline = VoicePipeline(Settings(mcp_enabled=False, skills_enabled=False))
    pipeline.hub = FakeHub([])
    pipeline.skills = None
    assert pipeline._tools_reply() == "I don't have any tools connected right now."
    pipeline.client.close()


# -- the conditional index for the model -------------------------------------


def test_the_index_is_prose_not_a_schema():
    index = tool_index(TOOLS)

    assert "local_time" in index and "roll_die" in index
    assert "current date and time" in index
    # No parameter names, types or JSON: nothing the model could turn into a call.
    assert "properties" not in index
    assert "integer" not in index
    assert "{" not in index


def test_the_index_respects_its_character_budget():
    many = [
        MCPTool(name=f"tool_{i}", description="x " * 200, input_schema={}) for i in range(60)
    ]
    assert len(tool_index(many, max_chars=800)) <= 800


def test_the_index_reaches_the_model_only_when_it_is_set():
    client = FakeClient()
    agent = Agent(client, Settings())

    list(agent.respond("hello"))
    assert "Tools available" not in client.seen[-1][0]["content"]

    agent.tool_context = tool_index(TOOLS)
    list(agent.respond("what can you do?"))
    assert "local_time" in client.seen[-1][0]["content"]


# -- referring to the assistant's own tools ---------------------------------


def test_only_second_person_tool_talk_counts_as_capability():
    from loma.control import mentions_own_tools

    assert mentions_own_tools("No, use your tools, mate.")
    assert mentions_own_tools("use the tools")
    assert mentions_own_tools("your tools are useless")
    # "your tools to check the weather" is still LOMA's tools; the pipeline only
    # injects the index when no tool call was accepted for the turn.
    assert mentions_own_tools("use your tools to check the weather")
    # A task that happens to contain the word "tool" is not a capability talk.
    assert not mentions_own_tools("what tools should I use to cut wood?")


def _pipeline(tools):
    pipeline = VoicePipeline(Settings(mcp_enabled=False, skills_enabled=False))
    pipeline.client.close()
    client = FakeClient()
    pipeline.client = client
    pipeline.agent = Agent(client, pipeline.settings)
    pipeline.models = ModelManager(client, pipeline.settings)
    pipeline.hub = FakeHub(tools)
    pipeline.skills = None
    return pipeline, client


def test_a_deterministic_answer_is_kept_in_the_transcript():
    """The model must not contradict an answer it never saw."""
    pipeline, _ = _pipeline(TOOLS)

    list(pipeline.stream_events("what tools do you have?", speak=False))

    transcript = pipeline.agent.transcript()
    assert transcript[-2]["content"] == "what tools do you have?"
    assert "I have two tools" in transcript[-1]["content"]


def test_talking_about_your_tools_gives_the_model_the_index():
    pipeline, client = _pipeline(TOOLS)

    list(pipeline.stream_events("No, use your tools, mate.", speak=False))

    system = client.seen[-1][0]["content"]
    assert "local_time" in system
    assert "never claim you have no tools" in system
