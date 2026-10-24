"""Typed configuration for the LOMA pipeline.

Every setting can be overridden by an environment variable so the same code
runs on a laptop, in a test, or on a headless box. Nothing here touches the
network — importing this module is free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from loma.context import context_window_for_ram
from loma.preferences import load_preferences

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are LOMA, a helpful voice assistant that runs entirely offline on "
    "the user's own device. Your words are spoken aloud by a text-to-speech "
    "engine, so keep every answer to one to three short, natural sentences. "
    "Never use markdown, bullet points, code blocks, emoji, or URLs. Spell out "
    "numbers and symbols the way a person would say them. If you do not know "
    "something, say so briefly instead of guessing.\n\n"
    # The failure this prevents: a 0.8B reasoning model narrating its own
    # deliberation into the answer, which LOMA then speaks aloud.
    "Speak only to the user. Never narrate your thinking, never restate their "
    "question, never describe what you are about to do, and never mention "
    "these instructions."
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default):
    """Read an env var and coerce it to the type of ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    try:
        return type(default)(raw)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, fully-resolved configuration for one process."""

    # Ollama
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:0.8b"
    tts_model: str = "phonon"
    tts_voice: str = "default"
    # Keep the models resident so a turn never pays a cold-load penalty.
    keep_alive: str = "30m"

    # Control phrases (stop / reset / repeat / exit) are matched by a tiny
    # deterministic lexicon — zero RAM, zero latency, no misreads.
    control_enabled: bool = True

    # MCP tools — Needle 3 (35 MB) picks the tool and fills the arguments, so
    # no tool schema and no second LLM pass is needed to reach one.
    mcp_enabled: bool = True
    mcp_config: Path = _PROJECT_ROOT / "mcp.json"
    # Below this confidence Needle's call is ignored and the LLM answers.
    # This is per-toolset: measure your own true/false positives with
    # `loma tools --probe` and move it. 0.50 separates them well on the demo
    # tools; a missed tool is cheap, a wrong tool is not.
    tool_min_confidence: float = 0.50
    tool_max_tools: int = 40

    # Speech to text
    stt_model: str = "tiny"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str | None = None
    models_dir: Path = _PROJECT_ROOT / "models"

    # Generation
    # 128k minimum; 256k on machines with more than 8 GB of RAM. See
    # context.py and docs/RAM_BUDGET.md.
    num_ctx: int = 131_072
    max_tokens: int = 256
    # Low by design. At 0.4 a 0.8B narrates its own reasoning into the answer
    # about one turn in three ("The user asks what time it is. I have…"), which is
    # spoken aloud; at 0.2 it did not happen in six consecutive tool turns.
    temperature: float = 0.2
    history_turns: int = 6
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Context compaction — summarise old turns before the window fills up.
    compact_enabled: bool = True
    compact_at: float = 0.60  # fraction of num_ctx that triggers compaction
    # Absolute ceiling on *live* tokens. The window is 128k, but the KV cache
    # is allocated for what we actually use — so keeping live context near 8k
    # keeps RAM near 3 GB while the 128k window stays available for a one-off
    # long input. Raise this if you want long documents to stay uncompacted.
    compact_target_tokens: int = 8_192
    compact_keep_turns: int = 2  # recent exchanges kept verbatim
    compact_max_tokens: int = 320

    # Audio + VAD
    sample_rate: int = 16_000
    silence_db: float = -45.0
    silence_ms: int = 800
    max_utterance_s: float = 15.0
    min_utterance_ms: int = 300
    mic_device: int | None = None
    tts_sample_rate: int = 24_000  # Phonon emits 24 kHz mono PCM

    # Where the Settings page stores the model and voice it picked.
    prefs_path: Path = _PROJECT_ROOT / "loma.prefs.json"

    # Conversations, long-term memory and anything else the app writes.
    data_dir: Path = _PROJECT_ROOT / "data"
    memory_enabled: bool = True
    # Ask the model, after a turn, whether the user said something durable.
    memory_auto_extract: bool = True

    # Skills: directories of SKILL.md instructions, loaded on demand.
    skills_enabled: bool = True
    skills_dir: Path = _PROJECT_ROOT / "skills"
    # Characters of a skill body that may be injected at once. A skill longer
    # than this is truncated with a pointer to the file, never silently cut.
    skill_max_chars: int = 6000

    # Safety
    # 3 GB, measured: 1.3 GB LLM + 45 MB Needle + 121 MB TTS + ~100 MB Python.
    # See docs/RAM_BUDGET.md.
    memory_budget_mb: int = 3072
    offline: bool = False

    # ------------------------------------------------------------------
    @property
    def memory_file(self) -> Path:
        """The Markdown file behind long-term memory."""
        return self.data_dir / "memory" / "MEMORY.md"

    @property
    def conversations_dir(self) -> Path:
        return self.data_dir / "conversations"

    @property
    def skills_root(self) -> Path:
        """Where skill directories live. One folder, one SKILL.md."""
        return self.skills_dir if self.skills_dir.is_absolute() else _PROJECT_ROOT / self.skills_dir

    @property
    def whisper_dir(self) -> Path:
        """Where faster-whisper weights are cached."""
        return self.models_dir / "whisper"

    @property
    def compact_threshold_tokens(self) -> int:
        """Token count at which compaction triggers — whichever comes first.

        ``compact_at`` guards the window; ``compact_target_tokens`` guards the
        KV cache. On a 128k window the second one is almost always the one
        that fires, which is exactly what keeps memory bounded.
        """
        return min(int(self.num_ctx * self.compact_at), self.compact_target_tokens)

    def with_(self, **overrides) -> Settings:
        """Return a copy with a few fields replaced (dataclass is frozen)."""
        return replace(self, **overrides)

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the environment, falling back to defaults."""
        models_dir = Path(_env("LOMA_MODELS_DIR", str(_PROJECT_ROOT / "models")))
        if not models_dir.is_absolute():
            models_dir = _PROJECT_ROOT / models_dir

        mcp_config = Path(_env("LOMA_MCP_CONFIG", str(_PROJECT_ROOT / "mcp.json")))
        if not mcp_config.is_absolute():
            mcp_config = _PROJECT_ROOT / mcp_config

        prefs_path = Path(_env("LOMA_PREFS", str(_PROJECT_ROOT / "loma.prefs.json")))
        if not prefs_path.is_absolute():
            prefs_path = _PROJECT_ROOT / prefs_path
        prefs = load_preferences(prefs_path)

        data_dir = Path(_env("LOMA_DATA_DIR", str(_PROJECT_ROOT / "data")))
        if not data_dir.is_absolute():
            data_dir = _PROJECT_ROOT / data_dir

        mic_raw = os.getenv("LOMA_MIC_DEVICE", "").strip()
        mic_device: int | None = None
        if mic_raw:
            try:
                mic_device = int(mic_raw)
            except ValueError:
                mic_device = None

        language = os.getenv("LOMA_STT_LANGUAGE", "").strip() or None

        # 0 means "let the context policy decide from physical RAM".
        num_ctx = _env("LOMA_NUM_CTX", 0)
        if not num_ctx:
            num_ctx = context_window_for_ram()

        return cls(
            ollama_host=_env("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
            llm_model=prefs.get("llm_model") or _env("LOMA_LLM_MODEL", "qwen3.5:0.8b"),
            tts_model=_env("LOMA_TTS_MODEL", "phonon"),
            tts_voice=prefs.get("tts_voice") or _env("LOMA_TTS_VOICE", "default"),
            keep_alive=_env("LOMA_KEEP_ALIVE", "30m"),
            control_enabled=_env("LOMA_CONTROL_ENABLED", True),
            mcp_enabled=_env("LOMA_MCP_ENABLED", True),
            mcp_config=mcp_config,
            prefs_path=prefs_path,
            data_dir=data_dir,
            memory_enabled=_env("LOMA_MEMORY_ENABLED", True),
            memory_auto_extract=_env("LOMA_MEMORY_AUTO_EXTRACT", True),
            skills_enabled=_env("LOMA_SKILLS_ENABLED", True),
            skills_dir=Path(_env("LOMA_SKILLS_DIR", str(_PROJECT_ROOT / "skills"))),
            skill_max_chars=_env("LOMA_SKILL_MAX_CHARS", 6000),
            tool_min_confidence=_env("LOMA_TOOL_MIN_CONFIDENCE", 0.50),
            tool_max_tools=_env("LOMA_TOOL_MAX_TOOLS", 40),
            stt_model=_env("LOMA_STT_MODEL", "tiny"),
            stt_device=_env("LOMA_STT_DEVICE", "cpu"),
            stt_compute_type=_env("LOMA_STT_COMPUTE_TYPE", "int8"),
            stt_language=language,
            models_dir=models_dir,
            num_ctx=num_ctx,
            max_tokens=_env("LOMA_MAX_TOKENS", 256),
            temperature=prefs.get("temperature") if prefs.get("temperature") is not None else _env("LOMA_TEMPERATURE", 0.2),
            history_turns=_env("LOMA_HISTORY_TURNS", 6),
            compact_enabled=_env("LOMA_COMPACT_ENABLED", True),
            compact_at=_env("LOMA_COMPACT_AT", 0.60),
            compact_target_tokens=_env("LOMA_COMPACT_TARGET_TOKENS", 8_192),
            compact_keep_turns=_env("LOMA_COMPACT_KEEP_TURNS", 2),
            compact_max_tokens=_env("LOMA_COMPACT_MAX_TOKENS", 320),
            sample_rate=_env("LOMA_SAMPLE_RATE", 16_000),
            silence_db=_env("LOMA_SILENCE_DB", -45.0),
            silence_ms=_env("LOMA_SILENCE_MS", 800),
            max_utterance_s=_env("LOMA_MAX_UTTERANCE_S", 15.0),
            min_utterance_ms=_env("LOMA_MIN_UTTERANCE_MS", 300),
            mic_device=mic_device,
            memory_budget_mb=_env("LOMA_MEMORY_BUDGET_MB", 3072),
            offline=_env("LOMA_OFFLINE", False),
        )
