"""The LLM calls tools; the Agent runs the protocol and keeps only the answer.

Ollama's shape: an assistant message carrying ``tool_calls``, then one
``role: tool`` message per result, then another model turn. None of that is
conversation history — the next user turn must see the question and the final
answer only.
"""

from __future__ import annotations

from loma.agent import Agent
from loma.config import Settings

TOOL = {
    "type": "function",
    "function": {
        "name": "roll_die",
        "description": "Roll a die.",
        "parameters": {"type": "object", "properties": {"sides": {"type": "integer"}}},
    },
}
CALL = {
    "id": "call_1",
    "function": {"index": 0, "name": "roll_die", "arguments": {"sides": 20}},
}


class ToolClient:
    """First round asks for a tool, second round answers."""

    def __init__(self, answer: str = "You rolled a 14.") -> None:
        self.answer = answer
        self.rounds: list[dict] = []

    def chat_stream(
        self,
        messages,
        *,
        think=False,
        tools=None,
        on_usage=None,
        on_tool_calls=None,
        **_,
    ):
        self.rounds.append(
            {"tools": tools, "messages": [dict(message) for message in messages]}
        )
        if len(self.rounds) == 1:
            if on_tool_calls is not None:
                on_tool_calls([CALL])
            yield ""
        else:
            yield self.answer
        if on_usage is not None:
            on_usage(
                {"prompt_tokens": 20 * len(self.rounds), "completion_tokens": 5,
                 "done_reason": "stop"}
            )


def test_a_tool_call_is_executed_and_the_answer_is_streamed():
    client = ToolClient()
    agent = Agent(client, Settings(tool_rounds=1))
    seen: list[tuple[str, dict]] = []

    def execute(name, arguments):
        seen.append((name, arguments))
        return '{"roll": 14, "sides": 20}'

    events = list(
        agent.respond_events(
            "roll a d20", tools=[TOOL], execute_tool=execute
        )
    )

    kinds = [event["type"] for event in events]
    assert "tool" in kinds and "tool_result" in kinds
    assert events[-1] == {"type": "delta", "text": "You rolled a 14."}
    assert seen == [("roll_die", {"sides": 20})]

    # The second round saw the assistant tool-call turn and the tool result.
    second = client.rounds[1]["messages"]
    assert any(message.get("tool_calls") for message in second)
    assert any(message.get("role") == "tool" for message in second)
    # With a one-round budget the follow-up is forced to answer: tools are
    # offered on the call that may use them, not on the one that must conclude.
    assert client.rounds[0]["tools"] == [TOOL]
    assert client.rounds[1]["tools"] is None


def test_the_tool_protocol_is_not_kept_in_the_transcript():
    agent = Agent(ToolClient(), Settings())
    list(
        agent.respond_events(
            "roll a d20", tools=[TOOL], execute_tool=lambda name, args: "roll=14"
        )
    )

    transcript = agent.transcript()
    assert [message["role"] for message in transcript] == ["user", "assistant"]
    assert transcript[0]["content"] == "roll a d20"
    assert transcript[1]["content"] == "You rolled a 14."
    assert all("tool_calls" not in message for message in transcript)


def test_a_failing_tool_still_produces_an_answer():
    class Failing(ToolClient):
        def chat_stream(self, messages, **kwargs):
            if len(self.rounds) == 0:
                self.rounds.append({})
                if kwargs.get("on_tool_calls") is not None:
                    kwargs["on_tool_calls"]([CALL])
                yield ""
            else:
                self.rounds.append({})
                yield "I could not roll the die."

    agent = Agent(Failing(), Settings())

    def explode(name, arguments):
        raise RuntimeError("MCP went away")

    events = list(
        agent.respond_events("roll a d20", tools=[TOOL], execute_tool=explode)
    )
    result = next(event for event in events if event["type"] == "tool_result")
    assert result["error"] is True
    assert "MCP went away" in result["result"]
    assert events[-1]["text"] == "I could not roll the die."


def test_the_last_tool_trace_is_persisted_without_results():
    agent = Agent(ToolClient(), Settings(tool_rounds=1))
    list(
        agent.respond_events(
            "roll a d20", tools=[TOOL], execute_tool=lambda name, args: "roll=14"
        )
    )

    answer = agent.transcript()[-1]
    assert answer["content"] == "You rolled a 14."
    assert answer["tool_trace"] == [{"name": "roll_die", "arguments": {"sides": 20}}]


def test_a_loaded_trace_replays_the_protocol_before_the_next_turn():
    """A recent tool-call shape keeps the model calling tools on a follow-up.

    Without it, lfm3 answered "what time is it?" from memory after a roll turn
    five times out of five.
    """
    client = ToolClient(answer="ok")
    agent = Agent(client, Settings())
    agent.load(
        [
            {"role": "user", "content": "roll a d20"},
            {
                "role": "assistant",
                "content": "You rolled a 14.",
                "tool_trace": [{"name": "roll_die", "arguments": {"sides": 20}}],
            },
        ]
    )
    # The follow-up question is the only new message; the loaded protocol is
    # already in the prompt.
    list(agent.respond("what time is it?", tools=[TOOL]))

    messages = client.rounds[0]["messages"]
    assert any(message.get("tool_calls") for message in messages)
    assert any(message.get("role") == "tool" for message in messages)
