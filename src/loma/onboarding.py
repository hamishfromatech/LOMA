"""First-run setup: a handful of questions, turned into memory facts.

The point is not a form. It is that LOMA is far more useful on turn one if it
already knows what to call you and what you are trying to get done — and that
this knowledge lives in the same ``MEMORY.md`` the model already reads, rather
than in some parallel profile the prompt never sees.

State lives in a small JSON file so that "skipped" is remembered too. Deriving
"has the user onboarded?" from "is memory empty?" would ask again on every visit
to anyone who declined.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STATE_FILENAME = "onboarding.json"

#: Longer answers are the user writing an essay; the fact is the first sentence.
MAX_ANSWER_CHARS = 240


def state_path(data_dir: Path) -> Path:
    return data_dir / STATE_FILENAME


def read_state(data_dir: Path) -> dict[str, Any]:
    path = state_path(data_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def write_state(data_dir: Path, state: dict[str, Any]) -> None:
    path = state_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def _clean(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split()).strip(" .")
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0]
    return text


def facts_from(answers: dict[str, Any]) -> list[str]:
    """Turn the wizard's answers into the bullet facts MEMORY.md holds.

    Each one is phrased the way a person would say it and the way the model can
    act on: a name, a job, what they want help with, how they like to be answered.
    Empty answers produce nothing — a skipped question is not a fact.
    """
    name = _clean(answers.get("name"))
    role = _clean(answers.get("role"))
    goals = answers.get("goals")
    style = _clean(answers.get("style"))
    notes = _clean(answers.get("notes"))

    facts: list[str] = []
    if name:
        facts.append(f"Prefers to be called {name}")
    if role:
        facts.append(f"Work and situation: {role}")
    if isinstance(goals, list):
        chosen = [_clean(goal) for goal in goals]
        chosen = [goal for goal in chosen if goal]
        if chosen:
            facts.append(f"Wants help with {', '.join(chosen)}")
    elif isinstance(goals, str) and _clean(goals):
        facts.append(f"Wants help with {_clean(goals)}")
    if style:
        facts.append(f"Likes answers that are {style}")
    if notes:
        facts.append(f"Context to keep in mind: {notes}")
    return facts


def summarise(answers: dict[str, Any]) -> dict[str, Any]:
    """What the UI shows back on the recap screen."""
    return {
        "name": _clean(answers.get("name")),
        "role": _clean(answers.get("role")),
        "goals": [
            _clean(goal)
            for goal in (answers.get("goals") or [])
            if isinstance(goal, str) and _clean(goal)
        ]
        if isinstance(answers.get("goals"), list)
        else [],
        "style": _clean(answers.get("style")),
        "notes": _clean(answers.get("notes")),
    }


def completed_state(answers: dict[str, Any], stored: int) -> dict[str, Any]:
    return {
        "onboarded": True,
        "skipped": False,
        "name": _clean(answers.get("name")),
        "completed_at": date.today().isoformat(),
        "facts_stored": stored,
    }


def skipped_state() -> dict[str, Any]:
    return {
        "onboarded": True,
        "skipped": True,
        "name": "",
        "completed_at": date.today().isoformat(),
        "facts_stored": 0,
    }
