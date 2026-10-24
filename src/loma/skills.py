"""Skills — reusable instructions the model can load on demand.

A skill is a directory with a ``SKILL.md`` in it:

    skills/
      write-an-mcp-server/
        SKILL.md
      tune-a-modelfile/
        SKILL.md

The file starts with YAML frontmatter, exactly like a Claude or A-Coder skill:

    ---
    name: write-an-mcp-server
    description: Build a FastMCP server LOMA can call, with arguments that
      survive a 35 MB router. Use when the user wants a new tool.
    ---

    # Write an MCP server
    ...

Two things are read, at different times, and the distinction is the whole
design:

* **The index** — every skill's ``name`` and ``description`` — is injected as a
  system note on every turn. It costs about 20 tokens per skill, so a dozen
  skills is still cheaper than one tool schema.
* **The body** is injected **only when the model asks for it**, by calling the
  built-in ``load_skill`` tool. That keeps a 4,000-token guide out of the window
  until the moment it is relevant, which is what makes a 128k window go a long
  way.

The body is fetched by the router, not by the model: ``load_skill`` takes a
``name`` that Needle fills from an enum of the installed skills, so a skill
cannot be hallucinated and a near-miss cannot silently load the wrong guide.

Frontmatter is optional. A bare ``SKILL.md`` with no frontmatter still works —
its first heading becomes the description — because a skill you cannot be
bothered to annotate is better than no skill at all.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"

#: Characters of a skill body that may enter the context at once.
DEFAULT_MAX_CHARS = 6000

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_FIELD = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$")
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_WORD = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True, slots=True)
class Skill:
    """One installed skill, as the index and the loader see it."""

    name: str
    description: str
    path: Path

    @property
    def short_description(self) -> str:
        """One line, trimmed — this is what enters the context every turn."""
        text = " ".join(self.description.split())
        return text[:200]


def _strip_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a SKILL.md into ``(frontmatter, body)``.

    Deliberately not a YAML parser: skills are hand-written, frontmatter is a
    flat map of strings, and ``pyyaml`` is not a dependency of an offline
    assistant that should stay small. Folded lines (``description: a`` then an
    indented ``b``) are joined, which is the only YAML feature that matters here.
    """
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text

    fields: dict[str, str] = {}
    key = ""
    for raw in match.group(1).splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        field = _FIELD.match(raw.strip())
        if field and not raw.startswith((" ", "\t")):
            key = field.group(1).lower()
            fields[key] = field.group(2).strip().strip("'\"")
        elif key:
            # A folded continuation line for the previous key.
            fields[key] = f"{fields[key]} {raw.strip()}".strip()
    return fields, text[match.end() :]


class SkillStore:
    """The ``skills/`` directory, scanned once and reloaded on demand."""

    def __init__(self, root: Path, *, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self.root = Path(root)
        self.max_chars = max_chars
        self._skills: list[Skill] = []
        self._loaded = False

    # -- discovery -----------------------------------------------------
    def scan(self, *, refresh: bool = False) -> list[Skill]:
        if self._loaded and not refresh:
            return self._skills

        found: list[Skill] = []
        if self.root.is_dir():
            for entry in sorted(self.root.iterdir()):
                if entry.name.startswith((".", "_")):
                    continue
                path = entry / SKILL_FILENAME if entry.is_dir() else entry
                if not path.is_file() or path.suffix.lower() != ".md":
                    continue
                skill = self._read(entry.name if entry.is_dir() else path.stem, path)
                if skill is not None:
                    found.append(skill)

        self._skills = found
        self._loaded = True
        if found:
            log.info("skills: %d installed (%s)", len(found), ", ".join(s.name for s in found))
        return found

    def _read(self, fallback_name: str, path: Path) -> Skill | None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("could not read skill %s: %s", path, exc)
            return None

        fields, body = _strip_frontmatter(text)
        name = (fields.get("name") or fallback_name).strip()
        description = (fields.get("description") or "").strip()

        if not description:
            # No frontmatter: the first heading or first non-empty line will do.
            heading = _HEADING.search(body)
            if heading:
                description = heading.group(1)
            else:
                # Skip lines with no words (a stray "---" from an unterminated
                # frontmatter block) and lines that are just a metadata field
                # ("name: broken"), which describe nothing a reader can use.
                first = next(
                    (
                        line
                        for line in body.splitlines()
                        if _WORD.search(line) and not _FIELD.match(line.strip())
                    ),
                    "",
                )
                description = first.lstrip("# ").strip()

        if not description:
            log.warning("skill %s has no description; skipping", path)
            return None

        return Skill(name=name, description=description, path=path)

    def refresh(self) -> list[Skill]:
        return self.scan(refresh=True)

    # -- access --------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self.scan())

    def names(self) -> list[str]:
        return [skill.name for skill in self.scan()]

    def get(self, name: str) -> Skill | None:
        """Find a skill by name, tolerating case, spaces and underscores.

        The router fills ``name`` from an enum, but the CLI and the browser can
        both send something a person typed, so be forgiving where it is free.
        """
        wanted = _normalise(name)
        if not wanted:
            return None
        for skill in self.scan():
            if _normalise(skill.name) == wanted:
                return skill
        return None

    def load(self, name: str) -> str:
        """The skill's body, frontmatter stripped and capped. Empty if unknown."""
        skill = self.get(name)
        if skill is None:
            return ""
        try:
            text = skill.path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("could not read skill %s: %s", skill.path, exc)
            return ""

        _, body = _strip_frontmatter(text)
        body = body.strip()
        if len(body) > self.max_chars:
            body = body[: self.max_chars].rsplit("\n", 1)[0]
            body += f"\n\n[skill truncated — the full file is at {skill.path}]"
        return body

    # -- prompt shapes -------------------------------------------------
    def as_index(self) -> str:
        """The always-on note: what skills exist and when to reach for one.

        Returns an empty string when nothing is installed, so a project without
        skills carries no tokens for them.
        """
        skills = self.scan()
        if not skills:
            return ""
        lines = [
            "You have skills installed: short guides written for this machine. "
            "The full text of a skill is not loaded — call load_skill with its "
            "name before answering anything it covers.",
        ]
        lines += [f"- {skill.name}: {skill.short_description}" for skill in skills]
        return "\n".join(lines)

    def as_context(self, name: str) -> str:
        """The loaded skill, framed so a 0.8B model follows it as instructions."""
        body = self.load(name)
        if not body:
            return ""
        skill = self.get(name)
        title = skill.name if skill else name
        return (
            f"The user asked for the {title!r} skill. Follow it for this reply. "
            "It describes procedure, not facts about the user.\n\n" + body
        )


def _normalise(name: str) -> str:
    return re.sub(r"[\s_-]+", "", (name or "").strip().lower())
