"""File-to-transcript decoding.

PyAV 19 removed the ``metadata_errors`` keyword that faster-whisper 1.2.1 still
passes, which broke /api/transcribe for every upload. LOMA decodes the file
itself and hands the model a numpy array instead. These tests cover that seam
without loading Whisper.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

pytest.importorskip("av")

from loma.config import Settings  # noqa: E402
from loma.stt import Transcriber, decode_audio  # noqa: E402

SR = 16_000


def write_tone(path, seconds: float = 0.5) -> None:
    samples = np.arange(int(SR * seconds)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * samples) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(tone.tobytes())


def test_a_wav_decodes_to_mono_float32_at_16k(tmp_path):
    path = tmp_path / "tone.wav"
    write_tone(path, 0.25)

    audio = decode_audio(path)

    assert audio.dtype == np.float32
    assert 0.24 * SR <= len(audio) <= 0.26 * SR
    assert float(np.abs(audio).max()) == pytest.approx(0.3, abs=0.02)


def test_an_empty_decode_is_an_empty_transcript(tmp_path):
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(b"")

    assert decode_audio(path).size == 0
    assert Transcriber(Settings()).transcribe(path).text == ""


class FakeModel:
    """Records what the model was given instead of running Whisper."""

    def __init__(self) -> None:
        self.seen = None

    def transcribe(self, audio, **kwargs):
        self.seen = audio

        class Segment:
            text = " hello there "

        class Info:
            language = "en"
            duration = 0.25

        return [Segment()], Info()


def test_a_path_is_decoded_before_the_model_is_called(tmp_path):
    path = tmp_path / "tone.wav"
    write_tone(path, 0.25)

    transcriber = Transcriber(Settings())
    fake = FakeModel()
    transcriber._model = fake

    transcript = transcriber.transcribe(path)

    assert isinstance(fake.seen, np.ndarray)
    assert fake.seen.dtype == np.float32
    assert transcript.text == "hello there"
    assert transcript.language == "en"
