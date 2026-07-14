"""LLM factory model defaulting without a request override."""

import pytest

pytestmark = pytest.mark.unit


def test_llm_factory_uses_backend_default_without_override(monkeypatch):
    from src.config.llm_factory import build_llm
    from src.config.settings import DocMindSettings

    class _OLike:
        def __init__(self, *, model: str, **_):
            self.model = model

    # Patch OpenAILike symbol
    import types as _t

    mod = _t.ModuleType("llama_index.llms.openai_like")
    mod.OpenAILike = _OLike  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "llama_index.llms.openai_like", mod)

    cfg = DocMindSettings()
    cfg.llm_backend = "vllm"
    out = build_llm(cfg)
    assert getattr(out, "model", None) == "Qwen/Qwen3-4B-Instruct-2507-FP8"
