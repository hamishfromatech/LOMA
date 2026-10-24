"""Preferences the user can change from the web UI.

Environment variables stay the first-class configuration for everything a
developer would script. Two values are different: the language model and the TTS
voice are things a *user* picks in the Settings page, so they live in a small
JSON file next to the project and override the environment for those two keys.

    loma.prefs.json
    { "llm_model": "qwen3.5:0.8b", "tts_voice": "default" }

Delete the file to fall back to `.env` / defaults.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Keys the Settings page may write.
EDITABLE = ("llm_model", "tts_voice", "temperature")


def load_preferences(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key in EDITABLE}


def save_preferences(path: Path, values: dict[str, Any]) -> dict[str, Any]:
    """Merge ``values`` into the file and return the stored result."""
    current = load_preferences(path)
    for key, value in values.items():
        if key in EDITABLE and value is not None:
            current[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2) + "\n")
    return current
