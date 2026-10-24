from loma.memory import MemorySnapshot, format_snapshot, process_rss_mb


def test_total_adds_python_and_ollama():
    snap = MemorySnapshot(
        python_rss_mb=150.0,
        ollama_models=[
            {"name": "bonsai-2:2b", "size": 876_000_000, "runner": "llamacpp"},
            {"name": "phonon:latest", "size": 121_000_000, "runner": "mlx"},
        ],
    )
    assert round(snap.ollama_mb, 1) == 997.0
    assert round(snap.total_mb, 1) == 1147.0


def test_snapshot_with_no_models():
    snap = MemorySnapshot(python_rss_mb=80.0)
    assert snap.ollama_mb == 0
    assert snap.total_mb == 80.0


def test_format_marks_a_budget_breach():
    snap = MemorySnapshot(
        python_rss_mb=200.0,
        ollama_models=[{"name": "big", "size": 3_000_000_000, "runner": "llamacpp"}],
    )
    text = format_snapshot(snap, budget_mb=3072)
    assert "OVER BUDGET" in text
    assert "big" in text


def test_format_marks_ok_within_budget():
    snap = MemorySnapshot(python_rss_mb=100.0)
    assert "-> OK" in format_snapshot(snap, budget_mb=3072)


def test_as_dict_is_json_shaped():
    snap = MemorySnapshot(python_rss_mb=1.0, ollama_models=[{"name": "m", "size": 2_000_000}])
    data = snap.as_dict()
    assert data["python_rss_mb"] == 1.0
    assert data["models"][0]["name"] == "m"
    assert data["models"][0]["size_mb"] == 2.0


def test_process_rss_is_positive():
    assert process_rss_mb() > 0
