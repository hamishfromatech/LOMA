"""RAM accounting against the project's 3 GB promise.

Two processes hold memory in this system:

  * the Python process (faster-whisper + numpy + audio buffers)
  * the Ollama server (the LLM and the TTS model)

``snapshot()`` reports both so you can prove the budget instead of trusting it.
"""

from __future__ import annotations

import logging
import resource
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

log = logging.getLogger(__name__)


@dataclass(slots=True)
class MemorySnapshot:
    python_rss_mb: float
    ollama_models: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ollama_mb(self) -> float:
        return sum(m.get("size", 0) for m in self.ollama_models) / 1_000_000

    @property
    def total_mb(self) -> float:
        return self.python_rss_mb + self.ollama_mb

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_rss_mb": round(self.python_rss_mb, 1),
            "ollama_mb": round(self.ollama_mb, 1),
            "total_mb": round(self.total_mb, 1),
            "models": [
                {
                    "name": m.get("name"),
                    "size_mb": round(m.get("size", 0) / 1_000_000, 1),
                    "runner": m.get("runner"),
                    "context": m.get("context_length"),
                }
                for m in self.ollama_models
            ],
        }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def process_rss_mb(include_children: bool = True) -> float:
    """Resident set size of this process (plus children) in megabytes."""
    process = psutil.Process()
    total = process.memory_info().rss
    if include_children:
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
    return total / 1_000_000


def peak_rss_mb() -> float:
    """High-water mark for this process, in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return usage / 1_000_000 if sys.platform == "darwin" else usage / 1024


def snapshot(client=None) -> MemorySnapshot:
    """Current Python RSS plus whatever Ollama has loaded."""
    models: list[dict[str, Any]] = []
    if client is not None:
        try:
            models = client.loaded_models()
        except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
            log.debug("could not read /api/ps: %s", exc)
    return MemorySnapshot(python_rss_mb=process_rss_mb(), ollama_models=models)


def format_snapshot(snap: MemorySnapshot, budget_mb: int | None = None) -> str:
    lines = [
        f"Python process : {snap.python_rss_mb:7.1f} MB",
        f"Ollama models  : {snap.ollama_mb:7.1f} MB",
    ]
    for model in snap.ollama_models:
        lines.append(
            f"    {model.get('name', '?'):<20} {model.get('size', 0) / 1_000_000:7.1f} MB"
            f"  ({model.get('runner', '?')})"
        )
    lines.append(f"{'Total':<15}  {snap.total_mb:7.1f} MB")
    if budget_mb:
        status = "OK" if snap.total_mb <= budget_mb else "OVER BUDGET"
        lines.append(f"{'Budget':<15}  {budget_mb:7.1f} MB  -> {status}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runtime guard
# ---------------------------------------------------------------------------


class MemoryGuard:
    """Background sampler that warns when the pipeline leaves the budget."""

    def __init__(self, budget_mb: int, client=None, interval_s: float = 5.0) -> None:
        self.budget_mb = budget_mb
        self.client = client
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="loma-memguard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _loop(self) -> None:
        warned = False
        while not self._stop.wait(self.interval_s):
            snap = snapshot(self.client)
            self.peak_mb = max(self.peak_mb, snap.total_mb)
            if snap.total_mb > self.budget_mb:
                if not warned:
                    log.warning(
                        "memory %.0f MB exceeds the %d MB budget",
                        snap.total_mb,
                        self.budget_mb,
                    )
                    warned = True
            else:
                warned = False

    def __enter__(self) -> MemoryGuard:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def wait_for_settle(seconds: float = 1.0) -> None:
    """Give the OS a moment before sampling memory."""
    time.sleep(seconds)
