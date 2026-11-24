"""One resident chat model at a time — Ollama's ``keep_alive`` as a swap policy.

LOMA has two chat models: a text model (``llm_model``) and a vision model
(``vlm_model``). Both are useful and neither is small, so keeping them both
loaded doubles the part of the footprint the user actually notices. Ollama
already knows how to load a model on demand and unload it after
``keep_alive``; this module just makes the choice explicit per turn:

    no image  -> llm_model   (and the VLM is unloaded with keep_alive=0)
    an image  -> vlm_model   (and the LLM is unloaded with keep_alive=0)

The unload is a real HTTP call that returns only after Ollama has dropped the
model, so the next request starts with the memory already freed. Swapping
costs a cold load (one to three seconds for a 1.7B model); leaving both loaded
costs their full weights for the whole session. On a laptop the second trade is
the expensive one.

Everything here is deterministic and side-effect free apart from the two
Ollama calls, so the policy is testable with a stub client.

See docs/ARCHITECTURE.md#two-chat-models-one-resident.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from loma.config import Settings
from loma.ollama_client import OllamaClient

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Swap:
    """One model exchange, kept for `loma status` and the benchmark.

    ``unload_seconds`` covers dropping the old model (Ollama blocks until it is
    gone). The *load* happens on the next request, so its cost shows up as that
    turn's first-token latency — `loma bench --swap` prints both.
    """

    previous: str
    current: str
    unload_seconds: float


class ModelManager:
    """Choose the model for a turn and keep only that one resident."""

    def __init__(self, client: OllamaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self._active: str | None = None
        #: Turns are serialised by the pipeline, but the memory extractor runs
        #: outside that lock; the unload/load pair must not interleave.
        self._lock = threading.Lock()
        self.swaps = 0
        self.last_swap: Swap | None = None

    # -- state ----------------------------------------------------------
    @property
    def active(self) -> str | None:
        """The model LOMA itself last asked Ollama for, if any."""
        return self._active

    def model_for(self, *, has_images: bool = False) -> str:
        """Which chat model should answer this turn?

        With ``dynamic_models`` off, the vision model is never chosen
        automatically: the selected model answers everything, and the caller's
        capability check decides whether an image is acceptable.
        """
        if has_images and self.settings.dynamic_models:
            return self.settings.vlm_model
        return self.settings.llm_model

    # -- swapping -------------------------------------------------------
    def prepare(self, model: str) -> str:
        """Make ``model`` the resident chat model; unload the other one.

        Returns the model to use, which matters when dynamic loading is off:
        the caller's pick is left untouched and Ollama decides for itself.
        """
        if not self.settings.dynamic_models:
            self._active = model
            return model

        with self._lock:
            if self._active == model:
                return model

            previous = self._active
            started = time.perf_counter()
            if previous and previous != model:
                log.info("swapping chat model: %s -> %s", previous, model)
                # keep_alive=0 is Ollama's "unload now". It blocks until the
                # model is gone, which is the point: the load below then has
                # the memory to itself.
                self.client.unload(previous)

            self._active = model
            if previous and previous != model:
                self.swaps += 1
                self.last_swap = Swap(
                    previous=previous,
                    current=model,
                    unload_seconds=time.perf_counter() - started,
                )
            return model

    def forget(self) -> None:
        """Drop our belief about what is resident (settings changed)."""
        with self._lock:
            self._active = None

    # -- per-model limits ----------------------------------------------
    def num_ctx(self, model: str) -> int:
        """The configured window, clamped to what ``model`` actually supports.

        ``lfm3`` carries 256k, ``lfm3-vl`` 125k. Sending the larger number to the
        smaller model is not harmless — Ollama can refuse the request outright —
        so the policy is applied per model rather than once at startup.
        """
        limit = self.client.context_length(model)
        if limit and limit < self.settings.num_ctx:
            return limit
        return self.settings.num_ctx

    def supports_vision(self, model: str) -> bool:
        """Does ``model`` read images? Unknown counts as no."""
        try:
            return "vision" in self.client.capabilities(model)
        except Exception:  # noqa: BLE001 - never block a turn on discovery
            return False

    # -- reporting ------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "active": self._active,
            "llm_model": self.settings.llm_model,
            "vlm_model": self.settings.vlm_model,
            "dynamic": self.settings.dynamic_models,
            "swaps": self.swaps,
            "last_swap": {
                "from": self.last_swap.previous,
                "to": self.last_swap.current,
                "unload_seconds": round(self.last_swap.unload_seconds, 3),
            }
            if self.last_swap
            else None,
        }
