from datetime import date

from loma.memory_store import HEADER, MemoryStore


def test_missing_file_is_created_with_a_header(tmp_path):
    store = MemoryStore(tmp_path / "memory" / "MEMORY.md")
    content = store.read()
    assert content.startswith("# LOMA memory")
    assert store.path.exists()


def test_append_writes_one_bullet(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    assert store.append("Keeps a note file by hand") is True
    facts = store.facts()
    assert facts == ["Keeps a note file by hand"]
    assert date.today().isoformat() in store.read()


def test_append_is_idempotent(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    assert store.append("Prefers tea") is True
    assert store.append("Prefers tea") is False
    assert store.facts() == ["Prefers tea"]


def test_append_ignores_tiny_facts(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    assert store.append("  ok ") is False
    assert store.facts() == []


def test_facts_ignore_prose_and_read_only_bullets(tmp_path):
    path = tmp_path / "MEMORY.md"
    path.write_text(
        "# Memory\n\nSome prose that is not a fact.\n- One fact\n* Another fact\n"
    )
    assert MemoryStore(path).facts() == ["One fact", "Another fact"]


def test_context_is_empty_when_nothing_is_remembered(tmp_path):
    assert MemoryStore(tmp_path / "MEMORY.md").as_context() == ""


def test_context_lists_facts_and_asks_for_restraint(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    store.append("Keeps a note file by hand")
    store.append("Prefers tea")
    context = store.as_context()
    assert "Keeps a note file by hand" in context
    assert "Prefers tea" in context
    assert "do not recite" in context.lower()


def test_context_respects_the_limit(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    for index in range(10):
        store.append(f"fact number {index}")
    context = store.as_context(limit=3)
    assert "fact number 9" in context
    assert "fact number 6" not in context


def test_write_replaces_the_file(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    store.append("old fact")
    store.write("# Custom\n- new fact\n")
    assert store.facts() == ["new fact"]


def test_clear_returns_the_header_only(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    store.append("something")
    store.clear()
    assert store.facts() == []
    assert store.read().startswith(HEADER.splitlines()[0])


def test_write_adds_a_trailing_newline(tmp_path):
    store = MemoryStore(tmp_path / "MEMORY.md")
    store.write("# no newline")
    assert store.path.read_text().endswith("\n")
