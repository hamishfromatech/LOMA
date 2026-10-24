#!/usr/bin/env python3
"""Prove the 3 GB claim by measuring the whole system, not just this process.

`loma bench` reports what Ollama says it is using. This script samples the
actual resident set size of every Ollama process plus this process while a
conversation runs, and reports the true peak.

    .venv/bin/python scripts/bench_memory.py --turns 3
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loma.config import Settings  # noqa: E402
from loma.pipeline import VoicePipeline  # noqa: E402

PROMPTS = [
    "In one sentence, what is a local language model?",
    "Give me a short fun fact about bees.",
    "What is seven times eight? Answer briefly.",
]


def ollama_rss_mb() -> float:
    total = 0
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (process.info["name"] or "").lower()
            cmdline = " ".join(process.info["cmdline"] or "")
            if "ollama" in name or "llama-server" in name or "ollama" in cmdline:
                total += process.memory_info().rss
        except psutil.Error:
            continue
    return total / 1_000_000


def own_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1_000_000


class Sampler(threading.Thread):
    def __init__(self, interval_s: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.own_peak = 0.0
        self.ollama_peak = 0.0
        self.total_peak = 0.0

    def run(self) -> None:
        while not self.stop_event.is_set():
            own = own_rss_mb()
            ollama = ollama_rss_mb()
            self.own_peak = max(self.own_peak, own)
            self.ollama_peak = max(self.ollama_peak, ollama)
            self.total_peak = max(self.total_peak, own + ollama)
            self.stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--budget", type=int, default=None, help="override the MB budget")
    args = parser.parse_args()

    settings = Settings.from_env()
    budget = args.budget or settings.memory_budget_mb

    print(f"budget: {budget} MB\n")
    sampler = Sampler()

    with VoicePipeline(settings) as pipeline:
        print("warming up all three models ...")
        pipeline.warm_up()
        time.sleep(1.5)

        sampler.start()
        try:
            for index in range(args.turns):
                prompt = PROMPTS[index % len(PROMPTS)]
                result = pipeline.reply(prompt, speak=False)
                print(f"  turn {index + 1}: {result.total_s:5.2f}s  {result.reply[:60]}")
            time.sleep(0.5)
        finally:
            sampler.stop()

    print()
    print(f"python peak RSS : {sampler.own_peak:7.1f} MB")
    print(f"ollama peak RSS : {sampler.ollama_peak:7.1f} MB")
    print(f"system peak     : {sampler.total_peak:7.1f} MB   (budget {budget} MB)")

    if sampler.total_peak <= budget:
        headroom = budget - sampler.total_peak
        print(f"\nPASS — {headroom:.0f} MB of headroom left.")
        return 0

    print(f"\nFAIL — over budget by {sampler.total_peak - budget:.0f} MB.")
    print("See docs/RAM_BUDGET.md for the levers to pull.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
