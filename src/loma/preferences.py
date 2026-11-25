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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .fs import atomic_write_text

log = logging.getLogger(__name__)

#: Keys the Settings page may write.
EDITABLE = ("llm_model", "vlm_model", "tts_voice", "temperature", "speak_replies")

#: Strings a person plausibly writes by hand for a boolean, in either sense.
_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off", ""}


def _model_name(value: Any) -> str | None:
    """A non-empty model reference, or None to reject it."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _voice(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _temperature(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN arrives from a slipped keyboard, never on purpose.
        return None
    return min(2.0, max(0.0, number))


def _boolean(value: Any) -> bool | None:
    """A real bool.

    ``bool("false")`` is True, and this file is edited by hand as well as by the
    UI, so the string has to be read as English rather than as Python truthiness.
    Getting this backwards means an assistant that speaks out loud when told not
    to, which is not a mistake to make twice.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    return None


#: One caster per editable key. None means "this value is not usable".
CASTS: dict[str, Callable[[Any], Any]] = {
    "llm_model": _model_name,
    "vlm_model": _model_name,
    "tts_voice": _voice,
    "temperature": _temperature,
    "speak_replies": _boolean,
}


def coerce(key: str, value: Any) -> Any:
    """The typed value for ``key``, or None when ``value`` cannot be honoured."""
    cast = CASTS.get(key)
    if cast is None or key not in EDITABLE:
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def load_preferences(path: Path) -> dict[str, Any]:
    """The stored preferences, typed.

    A value that cannot be read is dropped rather than passed through. The
    alternative is worse than a silent default: an unusable ``temperature`` used
    to travel straight into Ollama's ``options.temperature`` and make *every*
    subsequent turn fail with a 400, with nothing in the UI pointing at the
    preference file as the cause.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}

    result: dict[str, Any] = {}
    for key, value in data.items():
        if key not in EDITABLE:
            continue
        coerced = coerce(key, value)
        if coerced is None:
            log.warning("ignoring %s=%r in %s: not a valid %s", key, value, path, key)
            continue
        result[key] = coerced
    return result


def save_preferences(path: Path, values: dict[str, Any]) -> dict[str, Any]:
    """Merge ``values`` into the file and return the stored result.

    Values are coerced on the way in as well as on the way out, so the file on
    disk is always something a later read can use.
    """
    current = load_preferences(path)
    for key, value in values.items():
        if key not in EDITABLE or value is None:
            continue
        coerced = coerce(key, value)
        if coerced is None:
            log.warning("refused to save %s=%r", key, value)
            continue
        current[key] = coerced
    atomic_write_text(path, json.dumps(current, indent=2) + "\n")
    return current
