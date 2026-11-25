"""That a file survives, and that a value means what it says.

These are the bugs whose failure the user never sees until the damage is done: a
memory file truncated to nothing by a crash mid-write, a conversation that got
shorter on every turn, a preference that reads as its own opposite. Each test
here exists because the behaviour it checks was broken and silent.
"""

import base64
import json
import os
import stat
import sys
import threading
from pathlib import Path

import pytest

from loma.agent import Agent
from loma.config import Settings
from loma.conversations import MAX_TRANSCRIPT_MESSAGES, ConversationStore
from loma.fs import atomic_write_text
from loma.mcp_config import add_server, describe, empty, read_raw
from loma.preferences import coerce, load_preferences, save_preferences

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.images import MAX_BODY_BYTES, MAX_IMAGE_BYTES, MAX_IMAGES, read_images  # noqa: E402


# ---------------------------------------------------------------------------
# atomic writes
# ---------------------------------------------------------------------------
def test_atomic_write_replaces_the_file(tmp_path):
    path = tmp_path / "MEMORY.md"
    path.write_text("old")
    atomic_write_text(path, "new")
    assert path.read_text() == "new"


def test_atomic_write_creates_missing_parents(tmp_path):
    path = tmp_path / "a" / "b" / "prefs.json"
    atomic_write_text(path, "{}\n")
    assert path.read_text() == "{}\n"


def test_a_new_file_is_not_world_readable(tmp_path):
    """These files hold personal facts; a fresh one defaults to 0600."""
    path = tmp_path / "MEMORY.md"
    atomic_write_text(path, "prefers tea\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_an_existing_mode_is_respected(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text("{}")
    os.chmod(path, 0o644)
    atomic_write_text(path, '{"a": 1}')
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_a_failed_write_leaves_the_old_file_and_no_scratch(tmp_path, monkeypatch):
    """The whole point: a crash cannot cost the user the file they already had.

    ``path.write_text`` truncates first, so the old version of this test would
    have found an empty file here.
    """
    path = tmp_path / "MEMORY.md"
    path.write_text("everything I know")

    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError):
        atomic_write_text(path, "something new")

    assert path.read_text() == "everything I know"
    assert [item.name for item in tmp_path.iterdir()] == ["MEMORY.md"]


def test_concurrent_appends_both_survive(tmp_path):
    """Memory append is read-modify-write; racing writers used to eat a fact."""
    from loma.memory_store import MemoryStore

    store = MemoryStore(tmp_path / "MEMORY.md")
    barrier = threading.Barrier(8)

    def write(index: int) -> None:
        barrier.wait()
        store.append(f"fact number {index}")

    threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(store.facts()) == 8


# ---------------------------------------------------------------------------
# mcp_config: the empty default must not be shared
# ---------------------------------------------------------------------------
def test_empty_config_is_not_shared():
    first = empty()
    first["mcpServers"]["leaked"] = {"command": "x"}
    assert empty() == {"mcpServers": {}}


def test_adding_a_server_to_one_path_does_not_appear_on_another(tmp_path):
    """`dict(EMPTY)` was a shallow copy, so every missing path shared one inner
    dict and a server added anywhere showed up everywhere."""
    path = tmp_path / "mcp.json"
    add_server(path, "local", {"command": "python"})

    assert describe(Path("/does/not/exist.json")) == []
    assert read_raw(Path("/also/not/here.json")) == {"mcpServers": {}}


# ---------------------------------------------------------------------------
# preferences: a stored value has to mean what it says
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        (False, False),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("off", False),
        ("", False),
        (True, True),
        ("true", True),
        ("yes", True),
        ("1", True),
    ],
)
def test_speak_replies_reads_as_english_not_python_truthiness(raw, expected):
    # bool("false") is True, and this file is edited by hand as well as by the
    # UI. Getting it wrong means an assistant that speaks when told not to.
    assert coerce("speak_replies", raw) is expected


def test_a_junk_preference_is_dropped_rather_than_passed_through(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text(
        json.dumps(
            {
                "speak_replies": "false",
                "temperature": "warm",
                "llm_model": 12345,
                "vlm_model": "vision:2b",
            }
        )
    )
    loaded = load_preferences(path)
    assert loaded["speak_replies"] is False
    assert loaded["vlm_model"] == "vision:2b"
    # "warm" and 12345 must not travel to Ollama as options.temperature and the
    # model name — that made every subsequent turn fail with a 400.
    assert "temperature" not in loaded
    assert "llm_model" not in loaded


def test_temperature_is_clamped_to_a_usable_range():
    assert coerce("temperature", 3) == 2.0
    assert coerce("temperature", -5) == 0.0
    assert coerce("temperature", 0.4) == 0.4
    assert coerce("temperature", float("nan")) is None


def test_save_will_not_write_a_value_a_later_read_cannot_use(tmp_path):
    path = tmp_path / "prefs.json"
    save_preferences(path, {"temperature": "scorching", "speak_replies": False})
    assert load_preferences(path) == {"speak_replies": False}


# ---------------------------------------------------------------------------
# compaction must not delete the conversation
# ---------------------------------------------------------------------------
class Summariser:
    def __init__(self, summary: str = "we discussed apples then pears") -> None:
        self.summary = summary

    def chat_stream(self, messages, *, think=False, on_usage=None, **_):
        if on_usage is not None:
            on_usage({"prompt_tokens": 900, "completion_tokens": 5})
        yield "ok"

    def chat(self, messages, *, think=False, max_tokens=None, **_):
        return self.summary


def window_settings(**overrides) -> Settings:
    base = dict(
        num_ctx=1000,
        compact_at=0.5,  # threshold = 500 tokens
        compact_keep_turns=1,
        history_turns=2,
    )
    base.update(overrides)
    return Settings(**base)


def seeded_agent(settings: Settings, turns: int = 20) -> Agent:
    agent = Agent(Summariser(), settings)
    archive = []
    for index in range(turns):
        archive.append({"role": "user", "content": f"question {index}"})
        archive.append({"role": "assistant", "content": f"answer {index}"})
    agent.load(archive)
    agent.last_prompt_tokens = settings.compact_threshold_tokens + 1
    return agent


def test_compaction_folds_the_prompt_but_not_the_record(tmp_path):
    """The bug: compaction rewrote the working list, transcript() returned that
    list, and every save wrote less of the chat than the save before it."""
    settings = window_settings()
    agent = seeded_agent(settings)
    assert len(agent.transcript()) == 40

    list(agent.respond("and one more question"))

    assert len(agent._messages) < 10, "the model's window should still be small"
    assert len(agent.transcript()) == 42, "the record must still have everything"


def test_the_summary_survives_to_the_file_and_back(tmp_path):
    settings = window_settings()
    agent = seeded_agent(settings)
    list(agent.respond("and one more question"))

    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create("long chat")
    conversation.messages = agent.transcript()
    conversation.summary = agent.summary
    store.save(conversation)

    reloaded = store.get(conversation.id)
    assert reloaded is not None
    assert reloaded.summary == "we discussed apples then pears"

    # A second agent reading the same file gets the summary in front of its
    # window, so the next compaction folds it rather than re-deriving it.
    second = Agent(Summariser(), settings)
    second.load(reloaded.messages, summary=reloaded.summary)
    assert any("Earlier in this conversation" in message["content"] for message in second.messages)


def test_the_record_is_capped_and_the_cap_is_reported(tmp_path):
    settings = window_settings()
    agent = seeded_agent(settings, turns=(MAX_TRANSCRIPT_MESSAGES // 2) + 25)

    record = agent.transcript()
    assert len(record) == MAX_TRANSCRIPT_MESSAGES
    assert agent.trimmed_messages == 50


def test_an_empty_reply_rolls_the_question_back_from_the_record():
    class Silent(Summariser):
        def chat_stream(self, messages, *, think=False, on_usage=None, **_):
            return iter(())

    agent = Agent(Silent(), window_settings())
    list(agent.respond("hello?"))
    assert agent.transcript() == []


# ---------------------------------------------------------------------------
# the body cap and the image validator must agree
# ---------------------------------------------------------------------------
def test_four_legal_images_fit_in_the_body_cap():
    """4 x 8 MB is 42.7 MB once base64'd. MAX_CONTENT_LENGTH was 32 MB, so the
    documented limit could never be reached — Flask answered with an HTML 413."""
    images = [
        base64.b64encode(b"\xff" * (MAX_IMAGE_BYTES // 2)).decode("ascii")
        for _ in range(MAX_IMAGES)
    ]
    payload = json.dumps({"images": images})
    assert len(payload.encode()) < MAX_BODY_BYTES


def test_a_non_base64_image_is_rejected_before_the_model():
    cleaned, problem = read_images({"images": ["data:image/png;base64,not base64 at!!"]})
    assert cleaned == []
    assert problem is not None and "base64" in problem


def test_a_valid_data_url_is_stripped_to_bare_base64():
    value = base64.b64encode(b"\x89PNG\r\n").decode("ascii")
    cleaned, problem = read_images({"images": [f"data:image/png;base64,{value}"]})
    assert problem is None
    assert cleaned == [value]
