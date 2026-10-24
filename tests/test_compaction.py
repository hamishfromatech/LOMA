import pytest

from loma.agent import SUMMARY_PREFIX, Agent
from loma.config import Settings


class FakeClient:
    """Reports a configurable prompt size, and summarises on demand."""

    def __init__(self, prompt_tokens: int = 0, summary: str = "we discussed apples"):
        self.prompt_tokens = prompt_tokens
        self.summary = summary
        self.summarise_calls: list[list[dict]] = []

    def chat_stream(self, messages, *, think=False, on_usage=None, **_):
        if on_usage is not None:
            on_usage({"prompt_tokens": self.prompt_tokens, "completion_tokens": 5})
        yield "ok"

    def chat(self, messages, *, think=False, max_tokens=None):
        self.summarise_calls.append(messages)
        return self.summary


def small_window(**overrides) -> Settings:
    base = dict(
        num_ctx=1000,
        compact_at=0.5,
        compact_keep_turns=1,
        history_turns=10,
        compact_enabled=True,
    )
    base.update(overrides)
    return Settings(**base)


def fill(agent, turns: int) -> None:
    for index in range(turns):
        list(agent.respond(f"question {index}"))


def test_compaction_fires_above_the_threshold():
    client = FakeClient(prompt_tokens=800)  # above 50% of 1000
    agent = Agent(client, small_window())

    fill(agent, 3)

    assert agent.last_compaction is not None
    assert agent.last_compaction.messages_removed > 0
    assert agent.last_compaction.tokens_before == 800
    assert agent.compactions == 1
    assert any(
        m["role"] == "system" and m["content"].startswith(SUMMARY_PREFIX)
        for m in agent.messages
    )


def test_compaction_keeps_recent_turns_verbatim():
    client = FakeClient(prompt_tokens=800)
    agent = Agent(client, small_window(compact_keep_turns=1))

    fill(agent, 4)
    contents = [m["content"] for m in agent.messages]

    assert any("question 3" in c for c in contents)
    assert not any("question 0" in c for c in contents)


def test_no_compaction_below_the_threshold():
    client = FakeClient(prompt_tokens=100)  # well under 500
    agent = Agent(client, small_window())

    fill(agent, 4)

    assert agent.last_compaction is None
    assert agent.compactions == 0
    assert client.summarise_calls == []


def test_compaction_does_nothing_without_enough_history():
    client = FakeClient(prompt_tokens=900)
    agent = Agent(client, small_window())

    fill(agent, 1)  # only one exchange on the books
    list(agent.respond("second question"))

    assert agent.last_compaction is None


def test_compaction_can_be_disabled():
    client = FakeClient(prompt_tokens=900)
    agent = Agent(client, small_window(compact_enabled=False))

    fill(agent, 4)

    assert agent.compactions == 0


def test_failed_summary_falls_back_to_dropping_old_turns():
    class BrokenSummary(FakeClient):
        def chat(self, messages, *, think=False, max_tokens=None):
            from loma.ollama_client import OllamaError

            raise OllamaError("nope")

    client = BrokenSummary(prompt_tokens=900)
    agent = Agent(client, small_window())

    fill(agent, 3)

    assert agent.compactions == 1
    assert any("dropped" in m["content"] for m in agent.messages)


def test_history_trim_never_starts_on_an_assistant_turn():
    client = FakeClient(prompt_tokens=0)
    agent = Agent(client, Settings(history_turns=1, compact_enabled=False))

    fill(agent, 4)

    assert agent.messages[1]["role"] == "user"


def test_system_prompt_survives_compaction():
    client = FakeClient(prompt_tokens=900)
    settings = small_window()
    agent = Agent(client, settings)

    fill(agent, 4)

    assert agent.messages[0]["content"] == settings.system_prompt


def test_extra_context_is_transient():
    client = FakeClient(prompt_tokens=0)
    agent = Agent(client, Settings(compact_enabled=False))

    list(agent.respond("roll two dice", extra_context="tool result: 3 and 5"))

    contents = " ".join(m["content"] for m in agent.messages)
    assert "tool result" not in contents
    assert any(m["content"] == "roll two dice" for m in agent.messages)


@pytest.mark.parametrize("turns", [1, 2, 3, 5])
def test_token_accounting_is_recorded(turns):
    client = FakeClient(prompt_tokens=777)
    agent = Agent(client, Settings(compact_enabled=False))
    fill(agent, turns)
    assert agent.last_prompt_tokens == 777
