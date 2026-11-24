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


def _ignore_invalid_frames(frames):
    """Skip frames PyAV cannot decode instead of aborting the whole file."""
    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except Exception:  # noqa: BLE001 - av.error.InvalidDataError and friends
            continue


def _group_frames(frames, num_samples: int | None = None):
    """Batch frames so the resampler sees a decent chunk at a time."""
    import av

    fifo = av.audio.fifo.AudioFifo()
    for frame in frames:
        frame.pts = None  # ignore the timestamp check across container formats
        fifo.write(frame)
        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()
    if fifo.samples > 0:
        yield fifo.read()


def decode_audio(source: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    """Decode an audio file to mono float32 at ``sample_rate``.

    This mirrors ``faster_whisper.audio.decode_audio`` with one difference: it
    does not pass ``metadata_errors`` to ``av.open``. PyAV 19 removed that
    keyword and faster-whisper 1.2.1 still passes it, so letting faster-whisper
    decode a file path raises ``TypeError`` — every upload to /api/transcribe
    became a 500. Decoding here works on both old and new PyAV, and keeps the
    model call on a plain numpy array (which is also the microphone path).
    """
    import io
    import itertools

    import av

    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=sample_rate
    )
    raw = io.BytesIO()
    dtype = None

    with av.open(str(source), mode="r") as container:
        frames = _ignore_invalid_frames(container.decode(audio=0))
        grouped = _group_frames(frames, 500_000)
        # ``None`` flushes whatever is still buffered in the resampler.
        for frame in itertools.chain(grouped, [None]):
            for out in resampler.resample(frame):
                array = out.to_ndarray()
                dtype = array.dtype
                raw.write(array)

    if dtype is None:
        return np.zeros(0, dtype=np.float32)

    audio = np.frombuffer(raw.getbuffer(), dtype=dtype)
    return audio.astype(np.float32) / 32768.0


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
        if isinstance(audio, (str, Path)):
            audio = decode_audio(audio, self.settings.sample_rate)
        if audio.size == 0:
            return Transcript("")

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
