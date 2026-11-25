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

from .fs import atomic_write_text

log = logging.getLogger(__name__)

MAX_TITLE = 52
UNSAVED_TITLE = "New chat"

#: The archive is the record, not the prompt, so it is allowed to be far longer
#: than anything that goes near the context window. It still has to stop
#: somewhere: past this many messages the oldest are dropped, and the count is
#: recorded on the file so the drop is at least visible rather than silent.
MAX_TRANSCRIPT_MESSAGES = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _messages_from(raw: list[Any]) -> list[dict[str, Any]]:
    """Keep the two roles and, for an assistant turn, a bounded tool trace.

    The trace is names and arguments only. It exists so the next model turn can
    see the tool-call shape again; results are never persisted.
    """
    messages: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        message: dict[str, Any] = {
            "role": role,
            "content": str(item.get("content", "")),
        }
        trace = item.get("tool_trace")
        if role == "assistant" and isinstance(trace, list):
            clean = []
            for entry in trace[:8]:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "")[:120]
                arguments = entry.get("arguments")
                if name:
                    clean.append(
                        {
                            "name": name,
                            "arguments": arguments if isinstance(arguments, dict) else {},
                        }
                    )
            if clean:
                message["tool_trace"] = clean
        messages.append(message)
    return messages


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_reply: str = ""
    #: What the older, un-windowed turns added up to. Compaction rewrites the
    #: working context into this sentence, and it has to survive the turn that
    #: produced it — otherwise the next compaction summarises an already-shortened
    #: history and the beginning of the conversation is gone twice over.
    summary: str = ""
    #: How many of the oldest messages were trimmed to stay under the cap.
    trimmed_messages: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_reply": self.last_reply,
            "summary": self.summary,
            "trimmed_messages": self.trimmed_messages,
            "messages": self.messages,
        }

    def list_entry(self) -> dict[str, Any]:
        """The sidebar row: enough to draw the list without shipping the chat.

        This used to be called ``summary``, which collided with the compaction
        summary field added below — the method shadowed the dataclass default, so
        every save tried to serialise a function.
        """
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
            messages=_messages_from(data.get("messages", [])),
            last_reply=str(data.get("last_reply") or ""),
            summary=str(data.get("summary") or ""),
            trimmed_messages=_count(data.get("trimmed_messages")),
        )


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


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
        return [item.list_entry() for item in conversations]

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
        payload = json.dumps(conversation.to_dict(), indent=2, ensure_ascii=False) + "\n"
        atomic_write_text(self._path(conversation.id), payload)

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
