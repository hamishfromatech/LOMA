"""Conversations on disk.

Every chat is one JSON file under ``data/conversations/``. That keeps the format
inspectable — students can open the folder and see exactly what LOMA stored —
and it means the sidebar survives a restart without a database.

The file holds only what is needed to resume:

    {
      "id": "20261025-020717-4f2a",
      "title": "What time is it?",
      "created_at": "...",
      "updated_at": "...",
      "last_reply": "...",
      "messages": [{"role": "user", "content": "..."}, ...]
    }
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MAX_TITLE = 52
UNSAVED_TITLE = "New chat"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[dict[str, str]] = field(default_factory=list)
    last_reply: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_reply": self.last_reply,
            "messages": self.messages,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title") or UNSAVED_TITLE),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
            messages=[
                {"role": str(m.get("role")), "content": str(m.get("content", ""))}
                for m in data.get("messages", [])
                if m.get("role") in {"user", "assistant"}
            ],
            last_reply=str(data.get("last_reply") or ""),
        )


class ConversationStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    # -- paths ---------------------------------------------------------
    def _path(self, conversation_id: str) -> Path:
        safe = "".join(c for c in conversation_id if c.isalnum() or c in "-_")
        return self.directory / f"{safe}.json"

    def ensure(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- reads ---------------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        self.ensure()
        conversations: list[Conversation] = []
        for path in self.directory.glob("*.json"):
            try:
                conversations.append(Conversation.from_dict(json.loads(path.read_text())))
            except (OSError, ValueError) as exc:
                log.warning("skipping %s: %s", path.name, exc)
        conversations.sort(key=lambda item: item.updated_at, reverse=True)
        return [item.summary() for item in conversations]

    def get(self, conversation_id: str) -> Conversation | None:
        path = self._path(conversation_id)
        if not path.exists():
            return None
        try:
            return Conversation.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            log.warning("could not read %s: %s", path.name, exc)
            return None

    # -- writes --------------------------------------------------------
    def create(self, title: str = UNSAVED_TITLE) -> Conversation:
        now = _now()
        conversation = Conversation(
            id=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}",
            title=title,
            created_at=now,
            updated_at=now,
        )
        self.save(conversation)
        return conversation

    def save(self, conversation: Conversation) -> None:
        self.ensure()
        conversation.updated_at = _now()
        self._path(conversation.id).write_text(
            json.dumps(conversation.to_dict(), indent=2, ensure_ascii=False) + "\n"
        )

    def delete(self, conversation_id: str) -> bool:
        path = self._path(conversation_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def rename(self, conversation_id: str, title: str) -> Conversation | None:
        conversation = self.get(conversation_id)
        if conversation is None:
            return None
        conversation.title = title.strip()[:MAX_TITLE] or UNSAVED_TITLE
        self.save(conversation)
        return conversation

    # -- helpers -------------------------------------------------------
    @staticmethod
    def title_from(text: str) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= MAX_TITLE:
            return cleaned
        return cleaned[: MAX_TITLE - 1].rstrip() + "…"
