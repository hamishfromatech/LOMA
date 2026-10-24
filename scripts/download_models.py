#!/usr/bin/env python3
"""Cache the faster-whisper weights inside the repo.

Run once during setup. Afterwards `LOMA_OFFLINE=1` makes the whole pipeline
work with no network access at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loma.config import Settings  # noqa: E402
from loma.stt import Transcriber  # noqa: E402


def preload_needle() -> None:
    """Fetch Needle's engine and 35 MB of weights so the first turn is offline."""
    try:
        import needle
    except ImportError:
        print("cactus-needle is not installed; skipping the tool router")
        return

    @needle.tool
    def loma_warmup(value: int = 0):
        """Warm-up tool. Does nothing.

        Args:
            value: ignored
        """
        return {"value": value}

    print("downloading Needle 3 (engine + weights) ...")
    needle.Needle(tools=[loma_warmup])
    cache = Path.home() / ".cache" / "cactus-needle"
    size_mb = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    print(f"cached at {cache} ({size_mb / 1_000_000:.0f} MB)")


def main() -> int:
    settings = Settings.from_env()
    settings.whisper_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading faster-whisper '{settings.stt_model}' ...")
    transcriber = Transcriber(settings)
    transcriber.warm_up()

    size_mb = sum(f.stat().st_size for f in settings.whisper_dir.rglob("*") if f.is_file())
    print(f"cached at {settings.whisper_dir} ({size_mb / 1_000_000:.0f} MB)")

    preload_needle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
