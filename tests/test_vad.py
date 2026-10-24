import numpy as np

from loma.audio import UtteranceRecorder, VadConfig, dbfs, pcm16_to_float32

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000


def frame(amplitude: float) -> np.ndarray:
    """A 30 ms frame; amplitude 0 is silence, 0.3 is clearly speech-level."""
    if amplitude <= 0:
        return np.zeros(FRAME_LEN, dtype=np.float32)
    t = np.arange(FRAME_LEN) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_dbfs_of_silence_is_very_low():
    assert dbfs(np.zeros(FRAME_LEN, dtype=np.float32)) < -100


def test_dbfs_of_full_scale_sine_is_near_minus_three():
    assert -4.0 < dbfs(frame(1.0)) < -2.0


def test_pcm_roundtrip():
    original = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2")
    scaled = pcm16_to_float32(original.tobytes())
    assert scaled.shape == (5,)
    assert scaled[0] == 0.0
    assert 0.49 < scaled[1] < 0.51


def test_detects_an_utterance_after_silence():
    recorder = UtteranceRecorder(
        VadConfig(threshold_db=-45, silence_ms=210, min_utterance_ms=60), SAMPLE_RATE
    )
    assert recorder.push(frame(0.0)) is None
    assert recorder.push(frame(0.3)) is None
    assert recorder.speaking

    utterance = None
    for _ in range(20):
        utterance = recorder.push(frame(0.0))
        if utterance is not None:
            break

    assert utterance is not None
    assert utterance.dtype == np.float32
    assert utterance.size >= FRAME_LEN


def test_preroll_includes_audio_before_speech_started():
    recorder = UtteranceRecorder(
        VadConfig(threshold_db=-45, silence_ms=90, min_utterance_ms=30, preroll_ms=90),
        SAMPLE_RATE,
    )
    for _ in range(3):
        recorder.push(frame(0.0))
    recorder.push(frame(0.3))
    utterance = None
    for _ in range(10):
        utterance = recorder.push(frame(0.0))
        if utterance is not None:
            break
    # 3 preroll frames + 1 speech + 3 silence frames (90 ms / 30 ms).
    assert utterance is not None
    assert utterance.size / FRAME_LEN >= 7


def test_short_blip_is_rejected():
    recorder = UtteranceRecorder(
        VadConfig(threshold_db=-45, silence_ms=60, min_utterance_ms=300), SAMPLE_RATE
    )
    recorder.push(frame(0.3))
    utterance = None
    for _ in range(10):
        utterance = recorder.push(frame(0.0))
        if utterance is not None:
            break
    assert utterance is None
    assert not recorder.speaking


def test_max_utterance_cuts_off():
    recorder = UtteranceRecorder(
        VadConfig(threshold_db=-45, silence_ms=3000, max_utterance_s=0.09, min_utterance_ms=30),
        SAMPLE_RATE,
    )
    finished = None
    for _ in range(10):
        finished = recorder.push(frame(0.3))
        if finished is not None:
            break
    assert finished is not None
