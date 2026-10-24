"""Microphone capture, voice-activity detection, and speaker playback.

The VAD is deliberately a simple energy gate rather than a neural model: it
is a few dozen lines of numpy, costs almost no RAM, and is easy to reason
about. ``UtteranceRecorder`` is a pure state machine over fixed-size frames,
so it can be unit-tested without a sound card.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FRAME_MS = 30


# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------


def dbfs(frame: np.ndarray) -> float:
    """Root-mean-square level of a float32 frame, in dBFS (0 dB = full scale)."""
    if frame.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-8))


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Convert little-endian 16-bit PCM bytes to float32 in [-1, 1]."""
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(pcm, dtype="<i2")
    return samples.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Voice activity detection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VadConfig:
    threshold_db: float = -45.0
    silence_ms: int = 800
    max_utterance_s: float = 15.0
    min_utterance_ms: int = 300
    preroll_ms: int = 300
    frame_ms: int = FRAME_MS

    def __post_init__(self) -> None:
        if self.silence_ms < self.frame_ms:
            raise ValueError("silence_ms must be at least one frame")


class UtteranceRecorder:
    """Turn a stream of audio frames into finished utterances.

    Feed frames with :meth:`push`; it returns a float32 array once speech has
    started and then stopped again for ``silence_ms``. Returns ``None`` while
    the utterance is still in progress.
    """

    def __init__(self, config: VadConfig, sample_rate: int) -> None:
        self.config = config
        self.sample_rate = sample_rate
        frame_len = int(sample_rate * config.frame_ms / 1000)
        self._preroll_frames = max(1, config.preroll_ms // config.frame_ms)
        self._silence_frames = max(1, config.silence_ms // config.frame_ms)
        self._max_frames = max(1, int(config.max_utterance_s * 1000 / config.frame_ms))
        self._min_frames = max(1, config.min_utterance_ms // config.frame_ms)
        self._frame_len = frame_len

        self._preroll: deque[np.ndarray] = deque(maxlen=self._preroll_frames)
        self._frames: list[np.ndarray] = []
        self._speaking = False
        self._silent_run = 0

    # -- state ---------------------------------------------------------
    @property
    def speaking(self) -> bool:
        return self._speaking

    def reset(self) -> None:
        self._preroll.clear()
        self._frames.clear()
        self._speaking = False
        self._silent_run = 0

    # -- core ----------------------------------------------------------
    def push(self, frame: np.ndarray) -> np.ndarray | None:
        """Consume one frame. Returns a finished utterance or ``None``."""
        level = dbfs(frame)
        loud = level >= self.config.threshold_db

        if not self._speaking:
            if not loud:
                self._preroll.append(frame)
                return None
            # Speech just started: replay a little lead-in so the first
            # phoneme is not clipped.
            self._speaking = True
            self._frames = list(self._preroll)
            self._preroll.clear()
            self._silent_run = 0
            self._frames.append(frame)
            return None

        self._frames.append(frame)
        self._silent_run = 0 if loud else self._silent_run + 1

        enough_silence = self._silent_run >= self._silence_frames
        too_long = len(self._frames) >= self._max_frames

        if enough_silence or too_long:
            return self._finish()
        return None

    def _finish(self) -> np.ndarray | None:
        # Copy before reset(): reset() clears the live buffer in place.
        frames = list(self._frames)
        self.reset()
        if len(frames) < self._min_frames:
            log.debug("dropping %d-frame blip", len(frames))
            return None
        audio = np.concatenate(frames).astype(np.float32, copy=False)
        log.debug(
            "utterance: %.2fs, peak %.1f dBFS",
            audio.size / self.sample_rate,
            dbfs(audio),
        )
        return audio


# ---------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------


class Microphone:
    """Blocking mono microphone stream that yields fixed-size frames."""

    def __init__(self, settings, frame_ms: int = FRAME_MS) -> None:
        self.settings = settings
        self.frame_ms = frame_ms
        self._frame_len = int(settings.sample_rate * frame_ms / 1000)
        self._stream = None

    def __enter__(self) -> Microphone:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.settings.sample_rate,
            blocksize=self._frame_len,
            channels=1,
            dtype="float32",
            device=self.settings.mic_device,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self) -> np.ndarray:
        """Block until one frame is available and return it as mono float32."""
        if self._stream is None:
            raise RuntimeError("Microphone is not open — use it as a context manager")
        data, overflowed = self._stream.read(self._frame_len)
        if overflowed:
            log.debug("input overflow: the CPU could not keep up")
        return data[:, 0].astype(np.float32, copy=True)

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            yield self.read()


def list_devices() -> str:
    """Human-readable audio device table."""
    import sounddevice as sd

    return str(sd.query_devices())


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------


class Speaker:
    """Background TTS worker.

    ``say(text)`` enqueues a sentence. A single worker thread fetches the
    audio from Ollama and plays it, so language-model generation continues
    while the previous sentence is being spoken. This overlap is what makes
    the pipeline feel responsive instead of lock-step.
    """

    def __init__(
        self,
        tts_stream: Callable[[str], Iterable[bytes]],
        sample_rate: int = 24_000,
        device: int | None = None,
    ) -> None:
        self._tts_stream = tts_stream
        self._sample_rate = sample_rate
        self._device = device
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="loma-speaker", daemon=True)
        self._error: BaseException | None = None
        self._thread.start()

    # -- public API ----------------------------------------------------
    def say(self, text: str) -> None:
        text = text.strip()
        if text:
            self._queue.put(text)

    def wait(self) -> None:
        """Block until every queued sentence has been played."""
        self._queue.join()
        if self._error is not None:
            raise RuntimeError("speech playback failed") from self._error

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)

    def __enter__(self) -> Speaker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- worker --------------------------------------------------------
    def _run(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                pcm = b"".join(self._tts_stream(text))
                play_pcm(pcm, sample_rate=self._sample_rate, device=self._device)
            except BaseException as exc:  # noqa: BLE001 - surfaced by wait()
                self._error = exc
                log.error("speech worker error: %s", exc)
            finally:
                self._queue.task_done()


def play_pcm(pcm: bytes, sample_rate: int = 24_000, device: int | None = None) -> None:
    """Play raw 16-bit mono PCM through the default (or given) output device."""
    audio = pcm16_to_float32(pcm)
    if audio.size == 0:
        return
    import sounddevice as sd

    sd.play(audio, samplerate=sample_rate, device=device, blocking=True)


def write_wav(path: str | Path, pcm: bytes, sample_rate: int = 24_000) -> Path:
    """Write raw 16-bit mono PCM to a WAV file."""
    import wave

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return target
