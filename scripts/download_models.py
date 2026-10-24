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


def main() -> int:
    settings = Settings.from_env()
    settings.whisper_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading faster-whisper '{settings.stt_model}' ...")
    transcriber = Transcriber(settings)
    transcriber.warm_up()

    size_mb = sum(f.stat().st_size for f in settings.whisper_dir.rglob("*") if f.is_file())
    print(f"cached at {settings.whisper_dir} ({size_mb / 1_000_000:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
