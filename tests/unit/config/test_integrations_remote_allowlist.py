"""Remote allowlist tests for integrations base URL enforcement."""

import pytest

pytestmark = pytest.mark.unit


def test_remote_forbidden_without_allow_env(monkeypatch):
    import src.config.integrations as integ

    # Patch Settings and build_llm to be no-op
    class _S:
        llm = None
        embed_model = None
        context_window = None
        num_output = None

    monkeypatch.setitem(integ.__dict__, "Settings", _S)
    monkeypatch.setitem(integ.__dict__, "build_llm", lambda _s: object())
    monkeypatch.setitem(integ.__dict__, "ClipEmbedding", lambda *a, **k: object())

    # Remote URL for vLLM
    s = integ.settings
    monkeypatch.setenv("DOCMIND_ALLOW_REMOTE_ENDPOINTS", "0")
    s.llm_backend = "vllm"  # type: ignore[assignment]
    s.vllm_base_url = "http://remote.host:8000"  # type: ignore[assignment]

    integ.setup_llamaindex(force_llm=True, force_embed=False)
    # Should not bind a remote LLM when allowlist is not set
    assert integ.Settings.llm is None


def test_remote_allowed_with_env(monkeypatch):
    import src.config.integrations as integ

    class _S:
        llm = None
        embed_model = None
        context_window = None
        num_output = None

    monkeypatch.setitem(integ.__dict__, "Settings", _S)
    monkeypatch.setitem(integ.__dict__, "build_llm", lambda _s: object())
    monkeypatch.setitem(integ.__dict__, "ClipEmbedding", lambda *a, **k: object())

    s = integ.settings
    # Allow remote explicitly
    monkeypatch.setenv("DOCMIND_ALLOW_REMOTE_ENDPOINTS", "true")
    s.llm_backend = "lmstudio"  # type: ignore[assignment]
    s.lmstudio_base_url = "http://remote.host:1234/v1"  # type: ignore[assignment]

    # Should not raise
    integ.setup_llamaindex(force_llm=True, force_embed=False)
    assert integ.Settings.llm is not None
