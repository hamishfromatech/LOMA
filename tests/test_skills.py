"""Skills: frontmatter, discovery, loading and the always-on index.

No network, no models — the store is just files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loma.skills import SkillStore


def write_skill(root: Path, name: str, *, frontmatter: str | None, body: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    text = f"{frontmatter}\n\n{body}" if frontmatter else body
    path.write_text(text, encoding="utf-8")
    return path


# -- discovery ---------------------------------------------------------
def test_reads_name_and_description_from_frontmatter(tmp_path):
    write_skill(
        tmp_path,
        "chunk",
        frontmatter="---\nname: chunking\ndescription: Split a file into chunks.\n---",
        body="# Chunking\n\n1. Count the lines.",
    )
    skills = SkillStore(tmp_path).scan()
    assert [s.name for s in skills] == ["chunking"]
    assert skills[0].description == "Split a file into chunks."


def test_folder_name_is_the_fallback_name(tmp_path):
    write_skill(tmp_path, "my-skill", frontmatter=None, body="# A heading\n\nBody.")
    skill = SkillStore(tmp_path).scan()[0]
    assert skill.name == "my-skill"
    # No frontmatter: the first heading stands in for the description.
    assert skill.description == "A heading"


def test_skill_without_any_text_is_skipped(tmp_path):
    write_skill(tmp_path, "empty", frontmatter="---\nname: empty\n---", body="")
    assert SkillStore(tmp_path).scan() == []


def test_unterminated_frontmatter_is_not_treated_as_metadata(tmp_path):
    """A malformed block must not become a skill whose description is "---"."""
    write_skill(tmp_path, "broken", frontmatter=None, body="---\nname: broken")
    assert SkillStore(tmp_path).scan() == []


def test_folded_description_lines_are_joined(tmp_path):
    frontmatter = (
        "---\nname: folded\ndescription: Build a server LOMA can call,\n"
        "  with arguments a small router handles.\n---"
    )
    write_skill(tmp_path, "folded", frontmatter=frontmatter, body="Body.")
    assert "handles." in SkillStore(tmp_path).scan()[0].description


def test_flat_markdown_files_are_skills_too(tmp_path):
    (tmp_path / "loose.md").write_text("---\nname: loose\ndescription: A flat file.\n---\nBody")
    assert SkillStore(tmp_path).names() == ["loose"]


def test_underscore_and_dot_folders_are_ignored(tmp_path):
    write_skill(tmp_path, "_template", frontmatter="---\nname: t\ndescription: d\n---", body="x")
    write_skill(tmp_path, ".hidden", frontmatter="---\nname: h\ndescription: d\n---", body="x")
    write_skill(tmp_path, "real", frontmatter="---\nname: real\ndescription: d\n---", body="x")
    assert SkillStore(tmp_path).names() == ["real"]


def test_missing_directory_is_not_an_error(tmp_path):
    store = SkillStore(tmp_path / "nope")
    assert store.scan() == []
    assert store.as_index() == ""


# -- lookup ------------------------------------------------------------
@pytest.mark.parametrize("query", ["my-skill", "My Skill", "MY_SKILL", "myskill"])
def test_lookup_tolerates_case_spaces_and_underscores(tmp_path, query):
    write_skill(tmp_path, "my-skill", frontmatter="---\nname: my-skill\ndescription: d\n---", body="B")
    assert SkillStore(tmp_path).get(query) is not None


def test_unknown_skill_returns_none_and_empty_body(tmp_path):
    store = SkillStore(tmp_path)
    write_skill(tmp_path, "real", frontmatter="---\nname: real\ndescription: d\n---", body="x")
    store.refresh()
    assert store.get("ghost") is None
    assert store.load("ghost") == ""


# -- loading -----------------------------------------------------------
def test_load_returns_the_body_without_frontmatter(tmp_path):
    write_skill(
        tmp_path,
        "s",
        frontmatter="---\nname: s\ndescription: d\n---",
        body="# Heading\n\nStep one.",
    )
    body = SkillStore(tmp_path).load("s")
    assert body.startswith("# Heading")
    assert "description:" not in body


def test_long_bodies_are_truncated_with_a_pointer(tmp_path):
    write_skill(
        tmp_path,
        "long",
        frontmatter="---\nname: long\ndescription: d\n---",
        body="\n".join(f"line {i}" for i in range(500)),
    )
    body = SkillStore(tmp_path, max_chars=200).load("long")
    assert len(body) < 400
    assert "skill truncated" in body
    assert "SKILL.md" in body


# -- prompt shapes -----------------------------------------------------
def test_index_lists_every_skill_and_mentions_the_tool(tmp_path):
    write_skill(tmp_path, "a", frontmatter="---\nname: alpha\ndescription: First one.\n---", body="x")
    write_skill(tmp_path, "b", frontmatter="---\nname: beta\ndescription: Second one.\n---", body="x")
    index = SkillStore(tmp_path).as_index()
    assert "load_skill" in index
    # Names only: hand a 0.8B model a paragraph about its own skills and it
    # starts answering that instead of the question. Descriptions live in the
    # tool schema, where they are only paid for when the tool is considered.
    assert "alpha, beta" in index
    assert "First one." not in index


def test_context_frames_the_body_as_instructions(tmp_path):
    write_skill(
        tmp_path,
        "s",
        frontmatter="---\nname: s\ndescription: d\n---",
        body="1. Do the thing.",
    )
    context = SkillStore(tmp_path).as_context("s")
    assert "skill" in context.lower()
    assert "1. Do the thing." in context


def test_short_description_is_one_trimmed_line(tmp_path):
    write_skill(
        tmp_path,
        "s",
        frontmatter="---\nname: s\ndescription: A very\n  folded   description.\n---",
        body="Body.",
    )
    short = SkillStore(tmp_path).scan()[0].short_description
    assert "\n" not in short
    assert "  " not in short


# -- lexical matching --------------------------------------------------
def test_match_finds_the_skill_a_request_is_asking_for(tmp_path):
    write_skill(
        tmp_path,
        "write-a-skill",
        frontmatter="---\nname: write-a-skill\ndescription: Teach LOMA a procedure.\n---",
        body="x",
    )
    write_skill(
        tmp_path,
        "write-an-mcp-server",
        frontmatter="---\nname: write-an-mcp-server\ndescription: Build a new tool.\n---",
        body="x",
    )
    store = SkillStore(tmp_path)
    assert store.match("write me a skill for summarising PDFs").name == "write-a-skill"
    assert store.match("build an MCP server for my calendar").name == "write-an-mcp-server"


def test_match_returns_nothing_for_an_ordinary_question(tmp_path):
    write_skill(tmp_path, "write-a-skill", frontmatter="---\nname: write-a-skill\n---", body="x")
    store = SkillStore(tmp_path)
    assert store.match("what is on this machine?") is None
    assert store.match("what time is it?") is None
    assert store.match("") is None


def test_match_needs_more_than_one_shared_word(tmp_path):
    """One overlapping word is a coincidence, and a wrong guide is worse than none."""
    write_skill(
        tmp_path,
        "tune-a-modelfile",
        frontmatter="---\nname: tune-a-modelfile\ndescription: Adjust sampling for a model.\n---",
        body="x",
    )
    store = SkillStore(tmp_path)
    assert store.match("is the model ready?") is None
