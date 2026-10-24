from loma.config import Settings
from loma.mcp_hub import MCPTool
from loma.ollama_client import OllamaError
from loma.tool_router import NO_TOOL, ToolRouter


def tool(name="roll_dice", required=("count",), properties=None) -> MCPTool:
    return MCPTool(
        name=name,
        description=f"{name} does something",
        input_schema={
            "type": "object",
            "properties": properties if properties is not None else {"count": {"type": "integer"}},
            "required": list(required),
        },
    )


class FakeClient:
    def __init__(self, choice=NO_TOOL, confidence=0.0, probabilities=None, arguments=None, error=None):
        self.choice = choice
        self.confidence = confidence
        self.probabilities = probabilities or {choice: confidence}
        self.arguments = arguments if arguments is not None else {"count": 2}
        self.error = error
        self.systemone_calls = []
        self.json_calls = []

    def systemone(self, state, questions, *, keep_alive=None):
        self.systemone_calls.append((state, questions))
        if self.error:
            raise self.error
        return {
            "answers": {
                "tool": {
                    "type": "choice",
                    "choice": self.choice,
                    "probabilities": self.probabilities,
                    "confidence": self.confidence,
                }
            }
        }

    def chat_json(self, messages, schema, *, max_tokens=200):
        self.json_calls.append((messages, schema))
        if self.error:
            raise self.error
        return self.arguments


def test_confident_tool_is_selected():
    client = FakeClient("roll_dice", 0.92, {"roll_dice": 0.95, NO_TOOL: 0.04})
    decision = ToolRouter(client, Settings()).choose("roll two dice", [tool()])
    assert decision.tool is not None
    assert decision.tool.name == "roll_dice"
    assert decision.probability == 0.95


def test_none_choice_means_no_tool():
    client = FakeClient(NO_TOOL, 0.9, {NO_TOOL: 0.9, "roll_dice": 0.1})
    decision = ToolRouter(client, Settings()).choose("tell me a joke", [tool()])
    assert decision.tool is None
    assert "no tool" in decision.reason


def test_low_confidence_abstains():
    client = FakeClient("roll_dice", 0.30, {"roll_dice": 0.35, NO_TOOL: 0.6})
    decision = ToolRouter(client, Settings()).choose("maybe dice", [tool()])
    assert decision.tool is None
    assert "low confidence" in decision.reason


def test_router_error_is_not_fatal():
    client = FakeClient(error=OllamaError("model missing"))
    decision = ToolRouter(client, Settings()).choose("roll dice", [tool()])
    assert decision.tool is None
    assert "router error" in decision.reason


def test_no_tools_short_circuits():
    client = FakeClient()
    decision = ToolRouter(client, Settings()).choose("anything", [])
    assert decision.tool is None
    assert client.systemone_calls == []


def test_every_tool_appears_in_the_criteria_plus_none():
    client = FakeClient(NO_TOOL, 0.9)
    tools = [tool("roll_dice"), tool("get_time", required=()), tool("calc")]
    ToolRouter(client, Settings()).choose("hello", tools)
    _, questions = client.systemone_calls[0]
    criteria = questions["tool"]["criteria"]
    assert set(criteria) == {"roll_dice", "get_time", "calc", NO_TOOL}
    assert questions["tool"]["type"] == "choice"


def test_arguments_use_constrained_generation_with_one_schema():
    client = FakeClient(arguments={"count": 3, "bogus": "x"})
    router = ToolRouter(client, Settings())
    args = router.arguments(tool(), "roll three dice")
    assert args == {"count": 3}
    _, schema = client.json_calls[0]
    assert "count" in schema["properties"]


def test_arguments_missing_a_required_key_aborts_the_call():
    client = FakeClient(arguments={"other": 1})
    assert ToolRouter(client, Settings()).arguments(tool(), "roll dice") is None


def test_argument_failure_aborts_the_call():
    client = FakeClient(error=OllamaError("bad json"))
    assert ToolRouter(client, Settings()).arguments(tool(), "roll dice") is None


def test_tool_without_parameters_needs_no_generation():
    client = FakeClient()
    arguments = ToolRouter(client, Settings()).arguments(
        tool("get_time", required=(), properties={}), "time?"
    )
    assert arguments == {}
    assert client.json_calls == []


def test_decision_serialises_for_the_cli():
    client = FakeClient("roll_dice", 0.9, {"roll_dice": 0.93})
    data = ToolRouter(client, Settings()).choose("roll", [tool()]).as_dict()
    assert data["tool"] == "roll_dice"
    assert data["probabilities"]["roll_dice"] == 0.93
