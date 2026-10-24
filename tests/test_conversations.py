import json

from loma.conversations import Conversation, ConversationStore


def test_create_writes_a_file(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create()
    assert conversation.title == "New chat"
    assert (tmp_path / "conversations" / f"{conversation.id}.json").exists()


def test_list_is_newest_first(tmp_path):
    import time

    store = ConversationStore(tmp_path / "conversations")
    first = store.create("first")
    time.sleep(0.005)
    second = store.create("second")
    time.sleep(0.005)
    # save() refreshes updated_at, so touching the first one moves it to the top
    store.save(first)

    ids = [item["id"] for item in store.list()]
    assert ids == [first.id, second.id]


def test_round_trip_preserves_messages(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create("Greeting")
    conversation.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    conversation.last_reply = "hi"
    store.save(conversation)

    loaded = store.get(conversation.id)
    assert loaded is not None
    assert loaded.messages == conversation.messages
    assert loaded.last_reply == "hi"


def test_get_returns_none_for_unknown_id(tmp_path):
    assert ConversationStore(tmp_path).get("nope") is None


def test_delete_removes_the_file(tmp_path):
    store = ConversationStore(tmp_path)
    conversation = store.create()
    assert store.delete(conversation.id) is True
    assert store.get(conversation.id) is None
    assert store.delete(conversation.id) is False


def test_rename_persists(tmp_path):
    store = ConversationStore(tmp_path)
    conversation = store.create()
    renamed = store.rename(conversation.id, "  A better title  ")
    assert renamed is not None
    assert renamed.title == "A better title"
    assert store.get(conversation.id).title == "A better title"


def test_title_from_truncates_long_messages():
    title = ConversationStore.title_from("word " * 40)
    assert len(title) <= 52
    assert title.endswith("…")


def test_title_from_keeps_short_messages():
    assert ConversationStore.title_from("  hello   there ") == "hello there"


def test_unknown_roles_are_dropped_on_load(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(
        json.dumps(
            {
                "id": "x",
                "title": "t",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "messages": [
                    {"role": "system", "content": "nope"},
                    {"role": "user", "content": "yes"},
                ],
            }
        )
    )
    conversation = Conversation.from_dict(json.loads(path.read_text()))
    assert conversation.messages == [{"role": "user", "content": "yes"}]


def test_corrupt_file_is_skipped_when_listing(tmp_path):
    store = ConversationStore(tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    store.create("fine")
    titles = [item["title"] for item in store.list()]
    assert titles == ["fine"]


def test_ids_cannot_escape_the_directory(tmp_path):
    store = ConversationStore(tmp_path)
    path = store._path("../../etc/passwd")
    assert path.parent == tmp_path
