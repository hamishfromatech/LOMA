import json
from types import SimpleNamespace

import pytest

from loma.mcp_hub import MCPHub, MCPTool, ToolBatch, ToolCall, _result_text, load_mcp_config


def make_tool(**overrides) -> MCPTool:
    base = dict(
        name="roll_dice",
        description="Roll `count` dice with `sides` sides each.",
        input_schema={
            "type": "object",
            "properties": {"sides": {"type": "integer"}, "count": {"type": "integer"}},
            "required": ["count"],
        },
    )
    base.update(overrides)
    return MCPTool(**base)


def test_tool_exposes_required_and_properties():
    tool = make_tool()
    assert tool.required == ["count"]
    assert set(tool.properties) == {"sides", "count"}


def test_short_description_is_one_line_and_bounded():
    tool = make_tool(description="line one\n" + "x" * 500)
    assert "\n" not in tool.short_description
    assert len(tool.short_description) <= 220


def test_tool_call_prompt_names_the_tool_and_forbids_a_second_call():
    call = ToolCall(tool=make_tool(), arguments={"count": 2}, result='{"rolls":[3,5]}')
    prompt = call.as_prompt()
    assert "roll_dice" in prompt
    # Facts, not process: the note must not invite commentary about tooling.
    assert "do not mention" in prompt.lower()
    assert "tools just ran" not in prompt.lower()
    # Flat and readable, not a nested JSON array.
    assert "count=2" in prompt
    assert "rolls=[3, 5]" in prompt


def test_error_tool_call_asks_for_an_honest_answer():
    call = ToolCall(tool=make_tool(), arguments={}, result="error: boom", error=True)
    assert "failed" in call.as_prompt().lower()


def test_result_text_prefers_text_blocks():
    result = SimpleNamespace(
        is_error=False,
        content=[SimpleNamespace(text='{"rolls":[1,2]}')],
        data={"rolls": [1, 2]},
        structured_content=None,
    )
    assert _result_text(result) == '{"rolls":[1,2]}'


def test_result_text_falls_back_to_structured_data():
    result = SimpleNamespace(is_error=False, content=[], data={"total": 8})
    assert json.loads(_result_text(result)) == {"total": 8}


def test_result_text_reports_errors():
    result = SimpleNamespace(
        is_error=True, content=[SimpleNamespace(text="permission denied")], data=None
    )
    assert _result_text(result).startswith("error: permission denied")


def test_result_text_handles_empty_output():
    result = SimpleNamespace(is_error=False, content=[], data=None, structured_content=None)
    assert "no output" in _result_text(result)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_missing_config_is_not_an_error(tmp_path):
    assert load_mcp_config(tmp_path / "nope.json") is None


def test_config_placeholders_are_expanded(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "command": "${LOMA_PYTHON}",
                        "args": ["${LOMA_ROOT}/examples/x.py", "plain"],
                    }
                }
            }
        )
    )
    config = load_mcp_config(path)
    command = config["mcpServers"]["local"]["command"]
    args = config["mcpServers"]["local"]["args"]
    assert command.endswith("python") or "python" in command
    assert "${" not in command
    assert "${" not in args[0]
    assert args[1] == "plain"


def test_config_without_servers_is_rejected(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {}}))
    assert load_mcp_config(path) is None


# ---------------------------------------------------------------------------
# Hub without a server
# ---------------------------------------------------------------------------


def test_hub_without_config_is_inert():
    hub = MCPHub(None)
    assert not hub.configured
    assert hub.start() is False
    assert hub.available is False
    assert hub.tools == []
    assert "not connected" in hub.call("anything", {})


def test_hub_stop_is_idempotent():
    hub = MCPHub(None)
    hub.stop()
    hub.stop()


@pytest.mark.parametrize("name", ["roll_dice", "get_current_time", "calculate"])
def test_find_returns_none_for_an_empty_hub(name):
    hub = MCPHub(None)
    assert hub.find(name) is None


# ---------------------------------------------------------------------------
# ToolBatch
# ---------------------------------------------------------------------------


def call(name="roll_dice", result='{"rolls":[3,5]}', error=False):
    return ToolCall(
        tool=make_tool(name=name), arguments={"count": 2}, result=result, error=error
    )


def test_batch_prompt_covers_every_call():
    batch = ToolBatch(calls=[call(), call("get_time", result="22:31")])
    prompt = batch.as_prompt()
    assert "roll_dice" in prompt
    assert "get_time" in prompt
    assert "do not mention" in prompt.lower()
    assert batch.names == ["roll_dice", "get_time"]


def test_batch_prompt_is_flat_not_nested_json():
    prompt = ToolBatch(calls=[call()]).as_prompt()
    assert prompt.count("\n") == 1  # header + one line
    assert "{" not in prompt
    assert "rolls=[3, 5]" in prompt


def test_empty_batch_has_no_prompt():
    assert ToolBatch().as_prompt() == ""


def test_batch_reports_failure():
    batch = ToolBatch(calls=[call(error=True, result="error: nope")])
    assert batch.failed
    assert "failed" in batch.as_prompt().lower()


def test_batch_latency_is_the_sum():
    a, b = call(), call("get_time")
    a.latency_s, b.latency_s = 0.1, 0.2
    assert ToolBatch(calls=[a, b]).latency_s == pytest.approx(0.3)


def test_batch_serialises_for_the_cli():
    data = ToolBatch(calls=[call()]).as_dict()
    assert data["calls"][0]["tool"] == "roll_dice"
    assert data["calls"][0]["arguments"] == {"count": 2}
