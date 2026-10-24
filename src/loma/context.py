"""Context-window policy and token accounting.

LOMA's rules:

* The language model must run with **at least 128k tokens** of context.
* On a device with **more than 8 GB** of system RAM, use **256k**.

A 128k window is not free: Bonsai 2 is a hybrid-attention model (only 6 of its
24 layers keep a full KV cache), which works out to roughly 12.8 KB per token
at f16 and about 6.3 KB at q4_0. See ``docs/RAM_BUDGET.md`` for the measured
numbers and the ``OLLAMA_KV_CACHE_TYPE=q4_0`` recommendation that makes 128k
affordable.
"""

from __future__ import annotations

import psutil

#: The floor. LOMA never configures less than this.
MIN_CONTEXT_WINDOW = 131_072
#: Used when the machine has more than 8 GB of RAM.
LARGE_CONTEXT_WINDOW = 262_144
#: Machines at or below this get the 128k window.
LARGE_MEMORY_THRESHOLD_GB = 8.0

GIB = 1024**3


def total_ram_gb() -> float:
    """Physical memory of this machine, in GiB."""
    return psutil.virtual_memory().total / GIB


def context_window_for_ram(ram_gb: float | None = None) -> int:
    """Pick the context window for a machine with ``ram_gb`` of memory."""
    ram = total_ram_gb() if ram_gb is None else ram_gb
    return LARGE_CONTEXT_WINDOW if ram > LARGE_MEMORY_THRESHOLD_GB else MIN_CONTEXT_WINDOW


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used before real usage numbers are available.

    Deliberately conservative: 4 characters per token over-estimates English,
    which biases the compactor toward running early rather than too late.
    """
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate the token cost of a whole message list."""
    # Every message carries a small role/framing overhead.
    return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages)


def describe_policy(ram_gb: float | None = None) -> str:
    """One-line explanation for `loma context` and `loma doctor`."""
    ram = total_ram_gb() if ram_gb is None else ram_gb
    window = context_window_for_ram(ram)
    tier = ">8 GB machine" if ram > LARGE_MEMORY_THRESHOLD_GB else "<=8 GB machine"
    return f"{window // 1024}k context window chosen for a {ram:.1f} GB machine ({tier})"
