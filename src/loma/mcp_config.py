"""Read and write ``mcp.json`` from the Settings page.

The file is kept in the Claude-Desktop shape so it is portable between tools.
LOMA's ``${LOMA_ROOT}`` / ``${LOMA_PYTHON}`` placeholders are preserved on disk
and only expanded at connect time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

EMPTY: dict[str, Any] = {
    "mcpServers": {},
}


def read_raw(path: Path) -> dict[str, Any]:
    """The file exactly as written, placeholders and comments intact."""
    if not path.exists():
        return dict(EMPTY)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return dict(EMPTY)
    if not isinstance(data, dict):
        return dict(EMPTY)
    data.setdefault("mcpServers", {})
    if not isinstance(data["mcpServers"], dict):
        data["mcpServers"] = {}
    return data


def write_raw(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def servers(path: Path) -> dict[str, dict[str, Any]]:
    return dict(read_raw(path)["mcpServers"])


def add_server(path: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("server name is required")
    if not spec.get("command") and not spec.get("url"):
        raise ValueError("a server needs a command (stdio) or a url (http)")

    data = read_raw(path)
    data["mcpServers"][clean_name] = spec
    write_raw(path, data)
    return spec


def remove_server(path: Path, name: str) -> bool:
    data = read_raw(path)
    if name not in data["mcpServers"]:
        return False
    del data["mcpServers"][name]
    write_raw(path, data)
    return True


def describe(path: Path) -> list[dict[str, Any]]:
    """Configured servers, without expanding placeholders."""
    described = []
    for name, spec in servers(path).items():
        described.append(
            {
                "name": name,
                "transport": "http" if spec.get("url") else "stdio",
                "command": spec.get("command", ""),
                "args": spec.get("args", []),
                "url": spec.get("url", ""),
                "env": spec.get("env", {}),
            }
        )
    return described
