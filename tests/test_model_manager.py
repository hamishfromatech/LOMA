"""The text/vision swap policy.

No Ollama: the manager only needs ``unload``, ``capabilities`` and
``context_length``, so a stub client is enough to prove the policy.
"""

from __future__ import annotations

from loma.config import Settings
from loma.model_manager import ModelManager


class FakeClient:
    def __init__(self, capabilities=None, context=None) -> None:
        self.unloaded: list[str] = []
        self.capabilities_map = capabilities or {}
        self.context_map = context or {}

    def unload(self, model: str) -> None:
        self.unloaded.append(model)

    def capabilities(self, model: str) -> list[str]:
        return self.capabilities_map.get(model, [])

    def context_length(self, model: str) -> int:
        return self.context_map.get(model, 0)


def manager(**overrides) -> tuple[ModelManager, FakeClient]:
    values = dict(
        llm_model="lfm3:latest",
        vlm_model="lfm3-vl:latest",
        dynamic_models=True,
        num_ctx=262_144,
    )
    values.update(overrides)
    settings = Settings(**values)
    client = FakeClient(
        capabilities={"lfm3-vl:latest": ["completion", "vision"]},
        context={"lfm3:latest": 262_144, "lfm3-vl:latest": 128_000},
    )
    return ModelManager(client, settings), client


class TestModelChoice:
    def test_text_turn_uses_the_language_model(self):
        models, _ = manager()
        assert models.model_for(has_images=False) == "lfm3:latest"

    def test_image_turn_uses_the_vision_model(self):
        models, _ = manager()
        assert models.model_for(has_images=True) == "lfm3-vl:latest"

    def test_no_auto_vision_when_dynamic_loading_is_off(self):
        models, _ = manager(dynamic_models=False)
        assert models.model_for(has_images=True) == "lfm3:latest"


class TestSwapping:
    def test_first_turn_loads_without_unloading(self):
        models, client = manager()
        assert models.prepare("lfm3:latest") == "lfm3:latest"
        assert client.unloaded == []
        assert models.active == "lfm3:latest"

    def test_switching_unloads_the_other_model(self):
        models, client = manager()
        models.prepare("lfm3:latest")
        models.prepare("lfm3-vl:latest")
        assert client.unloaded == ["lfm3:latest"]
        assert models.active == "lfm3-vl:latest"
        assert models.swaps == 1
        assert models.last_swap is not None
        assert models.last_swap.previous == "lfm3:latest"
        assert models.last_swap.current == "lfm3-vl:latest"

    def test_switching_back_unloads_the_vision_model(self):
        models, client = manager()
        models.prepare("lfm3:latest")
        models.prepare("lfm3-vl:latest")
        models.prepare("lfm3:latest")
        assert client.unloaded == ["lfm3:latest", "lfm3-vl:latest"]
        assert models.swaps == 2

    def test_the_same_model_is_not_unloaded(self):
        models, client = manager()
        models.prepare("lfm3:latest")
        models.prepare("lfm3:latest")
        assert client.unloaded == []
        assert models.swaps == 0

    def test_dynamic_off_never_unloads(self):
        models, client = manager(dynamic_models=False)
        models.prepare("lfm3:latest")
        models.prepare("lfm3-vl:latest")
        assert client.unloaded == []
        assert models.swaps == 0

    def test_forget_drops_the_belief_so_the_next_turn_reloads(self):
        models, client = manager()
        models.prepare("lfm3:latest")
        models.forget()
        assert models.active is None
        models.prepare("lfm3-vl:latest")
        assert client.unloaded == []


class TestContextClamp:
    def test_a_smaller_model_gets_its_own_window(self):
        models, _ = manager()
        assert models.num_ctx("lfm3:latest") == 262_144
        assert models.num_ctx("lfm3-vl:latest") == 128_000

    def test_an_unknown_window_is_not_clamped(self):
        models, _ = manager()
        assert models.num_ctx("something-else") == 262_144

    def test_a_larger_model_keeps_the_configured_window(self):
        client = FakeClient(context={"lfm3:latest": 1_000_000})
        settings = Settings(num_ctx=131_072, dynamic_models=True)
        models = ModelManager(client, settings)
        assert models.num_ctx("lfm3:latest") == 131_072


class TestVisionSupport:
    def test_reports_the_vision_capability(self):
        models, _ = manager()
        assert models.supports_vision("lfm3-vl:latest") is True
        assert models.supports_vision("lfm3:latest") is False

    def test_unknown_is_no(self):
        client = FakeClient()
        models = ModelManager(client, Settings())
        assert models.supports_vision("nope") is False
