"""One shared LOMA engine, many conversations.

The models, the MCP subprocess and the Needle router are expensive and
single-instance, so they are shared. A *conversation* is just an :class:`Agent`
plus a JSON file, built fresh for each turn from what is on disk. That keeps the
sidebar durable, the memory footprint flat, and the code honest about what is
actually stateful.

Turns are serialised with a lock: one voice, one conversation at a time.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator

from loma import mcp_config
from loma.agent import Agent
from loma.config import Settings
from loma.context import describe_policy
from loma.conversations import ConversationStore
from loma.memory import snapshot
from loma.memory_evidence import (
    MEMORY_PROMPT,
    MEMORY_SCHEMA,
    looks_like_fact,
    quote_appears,
)
from loma.memory_store import MemoryStore
from loma.onboarding import (
    completed_state,
    facts_from,
    read_state,
    skipped_state,
    state_path,
    summarise,
    write_state,
)
from loma.pipeline import StopLoop, VoicePipeline
from loma.preferences import save_preferences

log = logging.getLogger(__name__)

#: Cheap gate before spending a generation on memory extraction.
_PERSONAL = re.compile(r"\b(i|i'm|im|i've|my|me|mine|we|our|us)\b", re.IGNORECASE)


#: A fact is a statement. These are the shapes that are not statements.
class Runtime:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.pipeline = VoicePipeline(self.settings)
        self.lock = threading.Lock()
        self.store = ConversationStore(self.settings.conversations_dir)
        self.memory = MemoryStore(self.settings.memory_file)
        self._warmed = False
        self._warm_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------
    def warm(self) -> None:
        with self._warm_lock:
            if self._warmed:
                return
            log.info("warming up LOMA")
            self.pipeline.memory = self.memory if self.settings.memory_enabled else None
            self.pipeline.warm_up()
            self._warmed = True

    def close(self) -> None:
        self.pipeline.close()

    @property
    def ready(self) -> bool:
        return self._warmed

    # -- settings ------------------------------------------------------
    def _apply(self, settings: Settings) -> None:
        self.settings = settings
        pipeline = self.pipeline
        pipeline.settings = settings
        pipeline.client.settings = settings
        pipeline.agent.settings = settings
        pipeline.transcriber.settings = settings
        pipeline.tool_router.settings = settings
        pipeline.controls.enabled = settings.control_enabled
        pipeline.hub.max_tools = settings.tool_max_tools
        pipeline.memory = self.memory if settings.memory_enabled else None

    def update_settings(
        self,
        *,
        llm_model: str | None = None,
        tts_voice: str | None = None,
        temperature: float | None = None,
    ) -> dict:
        if temperature is not None:
            temperature = min(1.5, max(0.0, float(temperature)))

        save_preferences(
            self.settings.prefs_path,
            {"llm_model": llm_model, "tts_voice": tts_voice, "temperature": temperature},
        )

        previous_model = self.settings.llm_model
        updated = Settings.from_env()
        self._apply(updated)

        if llm_model and llm_model != previous_model:
            log.info("language model: %s -> %s", previous_model, llm_model)
            with self.lock:
                self.pipeline.client.unload(previous_model)

        return self.settings_payload()

    # -- first run -----------------------------------------------------
    def onboarding_payload(self) -> dict:
        """Whether to show setup, and the name to greet with if we know it."""
        state = read_state(self.settings.data_dir)
        name = str(state.get("name") or "")
        if not name:
            # A name can also arrive from an older memory file, or from someone
            # who skipped the wizard and later said "remember that my name is …".
            for fact in self.memory.facts():
                lowered = fact.lower()
                if "called " in lowered:
                    name = fact.split("called ", 1)[1].strip()
                    break
                if lowered.startswith("my name is "):
                    name = fact[len("my name is ") :].strip()
                    break
        return {
            "needed": not bool(state.get("onboarded")),
            "skipped": bool(state.get("skipped")),
            "name": name,
        }

    def complete_onboarding(self, answers: dict) -> dict:
        stored = 0
        for fact in facts_from(answers):
            if self.memory.append(fact, source="onboarding"):
                stored += 1

        state = completed_state(answers, stored)
        write_state(self.settings.data_dir, state)
        log.info("onboarding complete: %s fact(s) stored", stored)
        return {
            "state": state,
            "summary": summarise(answers),
            "memory": self.memory_payload(),
        }

    def skip_onboarding(self) -> dict:
        write_state(self.settings.data_dir, skipped_state())
        return {"state": skipped_state(), "summary": {}, "memory": self.memory_payload()}

    def reset_onboarding(self) -> dict:
        """Forget that setup ran, so the wizard can be walked again.

        Facts already in memory are kept: re-running setup is about the answers,
        not about erasing what LOMA learned since.
        """
        path = state_path(self.settings.data_dir)
        if path.exists():
            path.unlink()
        return self.onboarding_payload()

    def models_payload(self) -> dict:
        """Every model the picker may offer, discovered from Ollama itself.

        A model whose capabilities could not be probed is kept: an unreachable
        ``/api/show`` means "unknown", not "cannot chat", and filtering on it is
        what empties the picker on a fresh machine.
        """
        client = self.pipeline.client
        error: str | None = None
        try:
            discovered = client.list_models()
        except Exception as exc:  # noqa: BLE001 - the picker must still render
            log.warning("could not list models: %s", exc)
            discovered = []
            error = str(exc)

        usable = [
            model
            for model in discovered
            if not model["capabilities"] or "completion" in model["capabilities"]
        ]

        # The configured model must always be selectable, even before it is
        # pulled — otherwise the select renders blank while Ollama catches up.
        if not any(model["name"] == self.settings.llm_model for model in usable):
            usable.insert(
                0,
                {
                    "name": self.settings.llm_model,
                    "size_mb": 0.0,
                    "family": "",
                    "parameters": "",
                    "quantization": "",
                    "modified_at": "",
                    "capabilities": [],
                    "source": "preference",
                },
            )

        return {
            "models": usable,
            "current": self.settings.llm_model,
            "source": discovered[0]["source"] if discovered else ("ollama" if not error else ""),
            "error": error,
        }

    def settings_payload(self) -> dict:
        models = self.models_payload()["models"]

        try:
            voices = self.pipeline.client.voices()
        except Exception:  # noqa: BLE001
            voices = []

        if not voices:
            voices = [self.settings.tts_voice]
        elif self.settings.tts_voice not in voices:
            voices = [self.settings.tts_voice, *voices]

        return {
            "llm_model": self.settings.llm_model,
            "tts_voice": self.settings.tts_voice,
            "temperature": self.settings.temperature,
            "options": {"llm_models": models, "voices": voices},
            "mcp": self.mcp_overview(),
            "memory": self.memory_payload(),
            "context": {"window": self.settings.num_ctx, "policy": describe_policy()},
        }

    # -- conversations -------------------------------------------------
    def list_conversations(self) -> list[dict]:
        return self.store.list()

    def new_conversation(self, title: str = "New chat") -> dict:
        conversation = self.store.create(title)
        return conversation.to_dict()

    def get_conversation(self, conversation_id: str) -> dict | None:
        conversation = self.store.get(conversation_id)
        return conversation.to_dict() if conversation else None

    def rename_conversation(self, conversation_id: str, title: str) -> dict | None:
        conversation = self.store.rename(conversation_id, title)
        return conversation.to_dict() if conversation else None

    def delete_conversation(self, conversation_id: str) -> bool:
        return self.store.delete(conversation_id)

    def chat(
        self,
        conversation_id: str,
        message: str,
        *,
        speak: bool = True,
        images: list[str] | None = None,
    ) -> Iterator[dict]:
        """Run one turn inside a conversation and persist the result."""
        conversation = self.store.get(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)

        agent = Agent(self.pipeline.client, self.settings)
        agent.load(conversation.messages)

        stored_memory = False
        try:
            with self.lock:
                self.pipeline.agent = agent
                self.pipeline.last_reply = conversation.last_reply
                self.pipeline.memory = self.memory if self.settings.memory_enabled else None

                for event in self.pipeline.stream_events(
                    message, speak=False, emit_audio=speak, images=images
                ):
                    if event.get("type") == "memory":
                        stored_memory = True
                    yield event
        except StopLoop:
            yield {"type": "done", "reply": "", "intent": "exit", "tool": [], "timings": {}}
        finally:
            reply = self.pipeline.last_reply
            conversation.messages = agent.transcript()
            conversation.last_reply = reply
            if conversation.title in {"", "New chat"} and conversation.messages:
                conversation.title = ConversationStore.title_from(
                    next(
                        (m["content"] for m in conversation.messages if m["role"] == "user"),
                        message,
                    )
                )
            self.store.save(conversation)

        # Outside the lock: a short, separate generation that must not delay the
        # next turn's audio.
        if not stored_memory:
            learned = self._extract_memory(message)
            if learned:
                yield {"type": "memory", "fact": learned, "stored": True}

    # -- memory --------------------------------------------------------
    def memory_payload(self) -> dict:
        return {
            "enabled": self.settings.memory_enabled,
            "auto_extract": self.settings.memory_auto_extract,
            "path": str(self.settings.memory_file),
            "content": self.memory.as_markdown(),
            "facts": self.memory.facts(),
        }

    def write_memory(self, content: str) -> dict:
        self.memory.write(content)
        return self.memory_payload()

    def clear_memory(self) -> dict:
        self.memory.clear()
        return self.memory_payload()

    def _extract_memory(self, user_text: str) -> str | None:
        """Ask the model whether the user said something worth keeping."""
        if not self.settings.memory_enabled or not self.settings.memory_auto_extract:
            return None
        if not _PERSONAL.search(user_text):
            return None

        try:
            result = self.pipeline.client.chat_json(
                [
                    {"role": "system", "content": MEMORY_PROMPT},
                    {"role": "user", "content": user_text[:2000]},
                ],
                MEMORY_SCHEMA,
                max_tokens=80,
            )
        except Exception as exc:  # noqa: BLE001 - memory must never break a turn
            log.debug("memory extraction skipped: %s", exc)
            return None

        if not result.get("worth_remembering"):
            return None
        fact = str(result.get("fact", "")).strip()
        if len(fact) < 8:
            return None

        quote = " ".join(str(result.get("quote", "")).split()).strip(" .\u201c\u201d\"'")
        if not quote or not quote_appears(quote, user_text):
            log.info("memory candidate rejected: no verbatim evidence (%r)", fact[:60])
            return None

        if not looks_like_fact(fact, quote):
            log.info("memory candidate rejected: not a statement (%r)", fact[:60])
            return None

        return fact if self.memory.append(fact, source="auto") else None

    # -- MCP -----------------------------------------------------------
    def ensure_tools(self) -> None:
        if self.pipeline.hub.available:
            return
        with self.lock:
            self.pipeline.start_tools()

    def mcp_overview(self) -> dict:
        self.ensure_tools()
        path = self.settings.mcp_config
        configured = mcp_config.describe(path)
        hub = self.pipeline.hub
        tools = hub.tools

        multi = len(configured) > 1
        for server in configured:
            name = server["name"]
            if multi:
                owned = [t.name for t in tools if t.name.startswith(f"{name}_")]
                server["connected"] = bool(owned)
            else:
                owned = [t.name for t in tools]
                server["connected"] = hub.available
            server["tools"] = owned

        return {
            "path": str(path),
            "connected": hub.available,
            "tool_count": len(tools),
            "servers": configured,
        }

    def add_mcp_server(self, name: str, spec: dict) -> dict:
        path = self.settings.mcp_config
        before = mcp_config.read_raw(path)
        mcp_config.add_server(path, name, spec)

        with self.lock:
            ok = self.pipeline.reload_tools()

        if not ok:
            mcp_config.write_raw(path, before)
            with self.lock:
                self.pipeline.reload_tools()
            error = self.pipeline.tool_router.last_error or self.pipeline.hub.last_error
            raise RuntimeError(error or "the server did not start")

        return self.mcp_overview()

    def remove_mcp_server(self, name: str) -> dict:
        path = self.settings.mcp_config
        before = mcp_config.read_raw(path)
        if not mcp_config.remove_server(path, name):
            raise KeyError(name)

        with self.lock:
            self.pipeline.reload_tools()

        if not self.pipeline.hub.available and before["mcpServers"]:
            mcp_config.write_raw(path, before)
            with self.lock:
                self.pipeline.reload_tools()
            raise RuntimeError("the remaining servers could not start")

        return self.mcp_overview()

    def reload_mcp(self) -> dict:
        with self.lock:
            self.pipeline.reload_tools()
        return self.mcp_overview()

    # -- status --------------------------------------------------------
    def status(self) -> dict:
        snap = snapshot(self.pipeline.client)
        return {
            "ready": self._warmed,
            "models": snap.as_dict(),
            "memory_budget_mb": self.settings.memory_budget_mb,
            "context": {
                "window": self.settings.num_ctx,
                "policy": describe_policy(),
                "live_tokens": self.pipeline.agent.history_tokens,
                "compact_threshold": self.settings.compact_threshold_tokens,
            },
            "tools": [
                {"name": t.name, "description": t.short_description, "required": t.required}
                for t in self.pipeline.hub.tools
            ],
            "config": {
                "llm_model": self.settings.llm_model,
                "tts_model": self.settings.tts_model,
                "stt_model": self.settings.stt_model,
                "tts_voice": self.settings.tts_voice,
                "temperature": self.settings.temperature,
            },
        }

    def reset(self) -> None:
        self.pipeline.agent.reset()
        self.pipeline.last_reply = ""


_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime
