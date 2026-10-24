import contextlib

from loma.agent import Agent
from loma.config import Settings


class FakeClient:
    """Stands in for OllamaClient so tests never touch the network."""

    def __init__(self, reply: str = "Sure thing.") -> None:
        self.reply = reply
        self.seen: list[list[dict[str, str]]] = []

    def chat_stream(self, messages, *, think: bool = False, on_usage=None, **_):
        self.seen.append(list(messages))
        for word in self.reply.split(" "):
            yield word + " "
        if on_usage is not None:
            on_usage({"prompt_tokens": 42, "completion_tokens": 7})


def test_respond_streams_and_records_history():
    client = FakeClient("Two plus two is four.")
    agent = Agent(client, Settings())

    chunks = list(agent.respond("what is 2+2?"))
    assert "".join(chunks).strip() == "Two plus two is four."

    messages = agent.messages
    assert messages[0]["role"] == "system"
    assert messages[-2] == {"role": "user", "content": "what is 2+2?"}
    assert messages[-1] == {"role": "assistant", "content": "Two plus two is four."}


def test_history_is_trimmed_to_the_configured_window():
    client = FakeClient("ok")
    agent = Agent(client, Settings(history_turns=2))

    for index in range(5):
        list(agent.respond(f"message {index}"))

    messages = agent.messages
    assert messages[0]["role"] == "system"
    # 2 turns = at most 4 messages plus the system prompt.
    assert len(messages) <= 5
    assert messages[1]["role"] == "user"


def test_empty_prompt_is_ignored():
    client = FakeClient()
    agent = Agent(client, Settings())
    assert list(agent.respond("   ")) == []
    assert len(agent.messages) == 1


def test_reset_clears_conversation():
    client = FakeClient()
    agent = Agent(client, Settings())
    list(agent.respond("hello"))
    agent.reset()
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"


def test_failed_turn_does_not_leave_a_dangling_user_message():
    class Broken(FakeClient):
        def chat_stream(self, messages, *, think: bool = False, on_usage=None, **_):
            if False:
                yield ""
            raise RuntimeError("boom")

    agent = Agent(Broken(), Settings())
    with contextlib.suppress(RuntimeError):
        list(agent.respond("hello"))
    assert all(m["role"] != "user" for m in agent.messages)
