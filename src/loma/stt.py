"""Speech-to-text with faster-whisper.

``tiny`` + ``int8`` on CPU is the sweet spot for this project: roughly 80 MB
of weights, well under a second of latency for a short utterance, and no GPU
required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from loma.config import Settings

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Transcript:
    text: str
    language: str | None = None
    duration: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class Transcriber:
    """Lazily-loaded faster-whisper wrapper.

    The model is only constructed on first use so that commands like
    ``loma speak`` never pay the whisper load cost.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    # -- loading -------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            whisper_dir = self.settings.whisper_dir
            whisper_dir.mkdir(parents=True, exist_ok=True)
            log.info(
                "loading faster-whisper '%s' (%s/%s)",
                self.settings.stt_model,
                self.settings.stt_device,
                self.settings.stt_compute_type,
            )
            self._model = WhisperModel(
                self.settings.stt_model,
                device=self.settings.stt_device,
                compute_type=self.settings.stt_compute_type,
                download_root=str(whisper_dir),
                local_files_only=self.settings.offline,
            )
        return self._model

    def warm_up(self) -> None:
        """Force the load and a dummy inference so the first turn is fast."""
        silence = np.zeros(self.settings.sample_rate // 10, dtype=np.float32)
        self.transcribe(silence)

    # -- inference -----------------------------------------------------
    def transcribe(self, audio: np.ndarray | str | Path) -> Transcript:
        """Transcribe float32 mono audio (or an audio file path)."""
        segments, info = self.model.transcribe(
            audio,
            language=self.settings.stt_language,
            beam_size=1,          # greedy: ~2x faster, fine for short turns
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,  # avoids hallucination loops
            vad_filter=False,     # our own VAD already trimmed the audio
            without_timestamps=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Transcript(
            text=text,
            language=getattr(info, "language", None),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
        )

    @staticmethod
    def available() -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True
