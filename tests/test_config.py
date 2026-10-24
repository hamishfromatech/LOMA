from pathlib import Path

from loma.config import Settings


def test_defaults_match_the_documented_budget():
    settings = Settings()
    assert settings.llm_model == "qwen3.5:0.8b"
    assert settings.tts_model == "phonon"
    assert settings.stt_model == "tiny"
    assert settings.stt_compute_type == "int8"
    # 128k is the documented floor; 256k is chosen on >8 GB machines.
    assert settings.num_ctx >= 131_072
    # The measured cost of 128k with all three models resident.
    assert settings.memory_budget_mb == 3072


def test_env_overrides(monkeypatch, tmp_path):
    # Preferences written by the Settings page would otherwise win.
    monkeypatch.setenv("LOMA_PREFS", str(tmp_path / "prefs.json"))
    monkeypatch.setenv("LOMA_LLM_MODEL", "qwen3.5:2b")
    monkeypatch.setenv("LOMA_MEMORY_BUDGET_MB", "2048")
    monkeypatch.setenv("LOMA_TEMPERATURE", "0.1")
    monkeypatch.setenv("LOMA_OFFLINE", "true")
    monkeypatch.setenv("LOMA_MIC_DEVICE", "3")

    settings = Settings.from_env()
    assert settings.llm_model == "qwen3.5:2b"
    assert settings.memory_budget_mb == 2048
    assert settings.temperature == 0.1
    assert settings.offline is True
    assert settings.mic_device == 3


def test_blank_env_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("LOMA_PREFS", str(tmp_path / "prefs.json"))
    monkeypatch.setenv("LOMA_LLM_MODEL", "   ")
    assert Settings.from_env().llm_model == "qwen3.5:0.8b"


def test_with_returns_a_modified_copy():
    base = Settings()
    other = base.with_(temperature=0.0)
    assert other.temperature == 0.0
    assert base.temperature == 0.4
    assert other is not base


def test_models_dir_is_absolute():
    settings = Settings.from_env()
    assert isinstance(settings.models_dir, Path)
    assert settings.models_dir.is_absolute()
    assert settings.whisper_dir.name == "whisper"


def test_system_prompt_forbids_markdown():
    # The reply is spoken, so markdown would be read out literally.
    prompt = Settings().system_prompt.lower()
    assert "markdown" in prompt
    assert "spoken" in prompt


def test_preferences_beat_the_environment(monkeypatch, tmp_path):
    prefs = tmp_path / "prefs.json"
    prefs.write_text('{"llm_model": "picked-in-the-ui"}')
    monkeypatch.setenv("LOMA_PREFS", str(prefs))
    monkeypatch.setenv("LOMA_LLM_MODEL", "from-the-env")
    assert Settings.from_env().llm_model == "picked-in-the-ui"


def test_missing_preferences_fall_back_to_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LOMA_PREFS", str(tmp_path / "nope.json"))
    monkeypatch.setenv("LOMA_LLM_MODEL", "from-the-env")
    assert Settings.from_env().llm_model == "from-the-env"


def test_data_paths_live_under_the_data_dir(tmp_path):
    settings = Settings(data_dir=tmp_path)
    assert settings.memory_file == tmp_path / "memory" / "MEMORY.md"
    assert settings.conversations_dir == tmp_path / "conversations"
