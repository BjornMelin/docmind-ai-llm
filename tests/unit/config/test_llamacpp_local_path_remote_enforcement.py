"""Ensure llama.cpp local GGUF path does not trigger remote endpoint checks."""

from __future__ import annotations

from types import SimpleNamespace


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

    # Compose a minimal settings namespace
    fake_settings = SimpleNamespace(
        llm_backend="llamacpp",
        model=None,
        ollama_base_url="http://localhost:11434",
        lmstudio_base_url="http://localhost:1234/v1",
        vllm_base_url=None,
        llamacpp_base_url=None,  # important: no URL provided
        vllm=SimpleNamespace(
            model="qwen2:7b-instruct",
            context_window=8192,
            max_tokens=256,
            llamacpp_model_path="models/foo.gguf",
        ),
        embedding=SimpleNamespace(model_name="BAAI/bge-m3", device="cpu"),
        context_window=8192,
        llm_context_window_max=131072,
    )

    # Patch module-level settings object
    monkeypatch.setitem(integ.__dict__, "settings", fake_settings)

    # Ensure allow-remote not set
    monkeypatch.delenv("DOCMIND_ALLOW_REMOTE_ENDPOINTS", raising=False)

    # Call setup; should not raise since a file path is not treated as URL
    integ.setup_llamaindex(force_llm=True, force_embed=False)

    assert integ.Settings.llm is not None
