import sys
import types

from loma.config import Settings
from loma.mcp_hub import MCPTool
from loma.tool_router import NeedleRouter, tool_schema


def make_tool(name="roll_dice", properties=None, required=("count",), description="Roll dice."):
    return MCPTool(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties if properties is not None else {
                "count": {"type": "integer", "description": "how many dice"},
                "sides": {"type": "integer", "description": "faces per die"},
            },
            "required": list(required),
        },
    )


class FakeAgent:
    def __init__(self, payload):
        self.payload = payload
        self.resets = 0
        self.queries = []

    def reset(self):
        self.resets += 1

    def complete(self, text):
        self.queries.append(text)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeNeedleModule(types.ModuleType):
    def __init__(self, payload):
        super().__init__("needle")
        self.payload = payload
        self.built: list[list[dict]] = []
        self.agents: list[FakeAgent] = []

    def Needle(self, tools=None):  # noqa: N802 - mirrors the real API
        self.built.append(list(tools or []))
        agent = FakeAgent(self.payload)
        self.agents.append(agent)
        return agent


def install(monkeypatch, payload) -> FakeNeedleModule:
    module = FakeNeedleModule(payload)
    monkeypatch.setitem(sys.modules, "needle", module)
    return module


def payload(calls=None, confidence=0.9, suppressed=None, reasoning=""):
    return {
        "type": "call",
        "function_calls": calls or [],
        "suppressed_calls": suppressed or [],
        "confidence": confidence,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


def test_schema_keeps_description_and_parameter_docs():
    schema = tool_schema(make_tool(), "roll_dice")
    assert schema["name"] == "roll_dice"
    assert schema["description"] == "Roll dice."
    assert schema["parameters"]["properties"]["sides"]["description"] == "faces per die"


def test_schema_handles_a_tool_with_no_parameters():
    tool = MCPTool(name="now", description="time", input_schema={})
    assert tool_schema(tool, "now")["parameters"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_builds_one_needle_agent(monkeypatch):
    module = install(monkeypatch, payload())
    router = NeedleRouter(Settings())
    assert router.load([make_tool()]) is True
    assert router.ready
    assert len(module.built) == 1
    assert module.built[0][0]["name"] == "roll_dice"


def test_load_is_idempotent_for_the_same_toolset(monkeypatch):
    module = install(monkeypatch, payload())
    router = NeedleRouter(Settings())
    tools = [make_tool()]
    router.load(tools)
    router.load(tools)
    assert len(module.built) == 1


def test_load_rebuilds_when_the_toolset_changes(monkeypatch):
    module = install(monkeypatch, payload())
    router = NeedleRouter(Settings())
    router.load([make_tool()])
    router.load([make_tool(), make_tool("get_time", properties={}, required=())])
    assert len(module.built) == 2


def test_load_without_needle_installed_fails_softly(monkeypatch):
    monkeypatch.setitem(sys.modules, "needle", None)
    router = NeedleRouter(Settings())
    assert router.load([make_tool()]) is False
    assert "not installed" in router.last_error


def test_load_with_no_tools_is_a_no_op(monkeypatch):
    install(monkeypatch, payload())
    assert NeedleRouter(Settings()).load([]) is False


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_a_confident_call_is_returned_with_its_tool(monkeypatch):
    install(monkeypatch, payload(
        calls=[{"name": "roll_dice", "arguments": {"count": 3, "sides": 20}}],
        confidence=0.86,
        reasoning="'three' -> count",
    ))
    router = NeedleRouter(Settings())
    router.load([make_tool()])

    route = router.route("roll three twenty sided dice")
    assert route.has_calls
    assert route.requests[0].tool.name == "roll_dice"
    assert route.requests[0].arguments == {"count": 3, "sides": 20}
    assert route.confidence == 0.86
    assert "count" in route.reasoning


def test_empty_calls_is_a_refusal(monkeypatch):
    install(monkeypatch, payload(calls=[], confidence=0.03))
    router = NeedleRouter(Settings())
    router.load([make_tool()])

    route = router.route("tell me a joke")
    assert not route.has_calls
    assert route.refused
    assert "refused" in route.reason


def test_low_confidence_call_is_ignored(monkeypatch):
    install(monkeypatch, payload(
        calls=[{"name": "roll_dice", "arguments": {"count": 2}}], confidence=0.20
    ))
    router = NeedleRouter(Settings())
    router.load([make_tool()])

    route = router.route("maybe dice")
    assert not route.has_calls
    assert "low confidence" in route.reason


def test_unknown_tool_names_are_dropped(monkeypatch):
    install(monkeypatch, payload(calls=[{"name": "hack_the_planet", "arguments": {}}]))
    router = NeedleRouter(Settings())
    router.load([make_tool()])
    assert not router.route("do something").has_calls


def test_arguments_are_filtered_to_the_schema(monkeypatch):
    install(monkeypatch, payload(
        calls=[{"name": "roll_dice", "arguments": {"count": 2, "bogus": "x", "sides": None}}]
    ))
    router = NeedleRouter(Settings())
    router.load([make_tool()])

    arguments = router.route("roll two dice").requests[0].arguments
    assert arguments == {"count": 2}


def test_multiple_calls_are_preserved_in_order(monkeypatch):
    install(monkeypatch, payload(calls=[
        {"name": "roll_dice", "arguments": {"count": 2}},
        {"name": "get_time", "arguments": {}},
    ]))
    router = NeedleRouter(Settings())
    router.load([make_tool(), make_tool("get_time", properties={}, required=())])

    route = router.route("roll two dice and tell me the time")
    assert route.names == ["roll_dice", "get_time"]


def test_each_utterance_resets_conversation_state(monkeypatch):
    module = install(monkeypatch, payload(calls=[]))
    router = NeedleRouter(Settings())
    router.load([make_tool()])
    router.route("first")
    router.route("second")
    assert [agent.resets for agent in module.agents] == [2]


def test_a_crashing_router_does_not_raise(monkeypatch):
    install(monkeypatch, RuntimeError("engine exploded"))
    router = NeedleRouter(Settings())
    router.load([make_tool()])

    route = router.route("hello")
    assert not route.has_calls
    assert "router error" in route.reason


def test_unloaded_router_returns_nothing():
    route = NeedleRouter(Settings()).route("hello")
    assert not route.has_calls
    assert "not loaded" in route.reason


def test_empty_text_short_circuits(monkeypatch):
    module = install(monkeypatch, payload())
    router = NeedleRouter(Settings())
    router.load([make_tool()])
    assert not router.route("   ").has_calls
    assert module.agents[0].queries == []


def test_suppressed_calls_are_counted(monkeypatch):
    install(monkeypatch, payload(
        calls=[], suppressed=[{"name": "roll_dice"}], confidence=0.04
    ))
    router = NeedleRouter(Settings())
    router.load([make_tool()])
    route = router.route("roll")
    assert route.suppressed == 1
    assert not route.has_calls


def test_tool_names_are_sanitised_for_needle(monkeypatch):
    module = install(monkeypatch, payload(
        calls=[{"name": "my_tool_v1", "arguments": {}}], confidence=0.9
    ))
    router = NeedleRouter(Settings())
    router.load([make_tool("my.tool-v1", properties={}, required=())])

    assert module.built[0][0]["name"] == "my_tool_v1"
    assert router.route("go").requests[0].tool.name == "my.tool-v1"


def test_route_serialises_for_the_cli(monkeypatch):
    install(monkeypatch, payload(
        calls=[{"name": "roll_dice", "arguments": {"count": 2}}], confidence=0.8
    ))
    router = NeedleRouter(Settings())
    router.load([make_tool()])
    data = router.route("roll two").as_dict()
    assert data["confidence"] == 0.8
    assert data["tools"][0]["tool"] == "roll_dice"
