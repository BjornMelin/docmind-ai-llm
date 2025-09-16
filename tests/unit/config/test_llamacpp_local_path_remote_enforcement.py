"""Ensure llama.cpp local GGUF path does not trigger remote endpoint checks."""

from __future__ import annotations

from src.config.settings import DocMindSettings


def test_llamacpp_local_path_skips_remote_enforcement(monkeypatch):
    # Import target module
    import src.config.integrations as integ

    # Prepare a fake Settings object to avoid real LlamaIndex imports
    class _FakeSettings:
        llm = None
        embed_model = None
        context_window = None
        num_output = None

    # Monkeypatch global Settings in module to a simple container
    monkeypatch.setitem(integ.__dict__, "Settings", _FakeSettings)

    # Patch build_llm to return a dummy llm object
    monkeypatch.setitem(integ.__dict__, "build_llm", lambda _s: object())

    # Patch HuggingFaceEmbedding to avoid heavy imports during embed model init
    class _DummyEmb:
        def __init__(self, *a, **k): ...

    monkeypatch.setitem(integ.__dict__, "HuggingFaceEmbedding", _DummyEmb)

    # Compose a minimal settings namespace using the real model
    cfg = DocMindSettings(llm_backend="llamacpp")
    cfg.model = "local-gguf-path"  # type: ignore[assignment]
    cfg.context_window = 4096
    cfg.vllm.llamacpp_model_path = "models/foo.gguf"  # type: ignore[assignment]
    cfg.security.allow_remote_endpoints = False
    cfg.llamacpp_base_url = None  # type: ignore[assignment]

    # Patch module-level settings object
    monkeypatch.setitem(integ.__dict__, "settings", cfg)

    # Call setup; should not raise since a file path is not treated as URL
    integ.setup_llamaindex(force_llm=True, force_embed=False)

    assert integ.Settings.llm is not None
