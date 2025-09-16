"""Remote allowlist tests for integrations base URL enforcement."""

import pytest

from src.config.settings import DocMindSettings

pytestmark = pytest.mark.unit


def _stubbed_integrations(monkeypatch):
    import src.config.integrations as integ

    class _SettingsProxy:
        llm = None
        embed_model = None
        context_window = None
        num_output = None

    monkeypatch.setitem(integ.__dict__, "Settings", _SettingsProxy)
    monkeypatch.setitem(integ.__dict__, "build_llm", lambda _s: object())
    monkeypatch.setitem(
        integ.__dict__, "HuggingFaceEmbedding", lambda *a, **k: object()
    )
    return integ


def test_remote_forbidden_without_allowance(monkeypatch):
    integ = _stubbed_integrations(monkeypatch)

    cfg = DocMindSettings()
    cfg.llm_backend = "vllm"  # type: ignore[assignment]
    cfg.vllm_base_url = "http://remote.host:8000"  # type: ignore[assignment]
    cfg.security.allow_remote_endpoints = False
    cfg.security.endpoint_allowlist = [
        "http://localhost",
        "https://localhost",
        "http://127.0.0.1",
        "https://127.0.0.1",
    ]

    monkeypatch.setitem(integ.__dict__, "settings", cfg)

    integ.setup_llamaindex(force_llm=True, force_embed=False)

    assert integ.Settings.llm is None


def test_remote_allowed_with_policy(monkeypatch):
    integ = _stubbed_integrations(monkeypatch)

    cfg = DocMindSettings()
    cfg.llm_backend = "lmstudio"  # type: ignore[assignment]
    cfg.lmstudio_base_url = "http://remote.host:1234/v1"  # type: ignore[assignment]
    cfg.security.allow_remote_endpoints = True

    monkeypatch.setitem(integ.__dict__, "settings", cfg)

    integ.setup_llamaindex(force_llm=True, force_embed=False)

    assert integ.Settings.llm is not None
