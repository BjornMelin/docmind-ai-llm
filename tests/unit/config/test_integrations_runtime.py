"""Smoke test for initialize_integrations rebind behavior (SPEC-001)."""

from __future__ import annotations

import pytest
from llama_index.core import Settings
from llama_index.core.llms.mock import MockLLM

from src.config.integrations import initialize_integrations
from src.config.settings import settings as global_settings


@pytest.mark.unit
def test_initialize_integrations_rebinds_settings_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply runtime should bind ``Settings.llm`` without real provider calls."""
    original_llm = getattr(Settings, "_llm", None)
    sentinel_llm = MockLLM()

    monkeypatch.setattr(
        "src.config.integrations.build_llm", lambda _settings=None: sentinel_llm
    )
    monkeypatch.setattr("src.config.integrations.settings", global_settings)
    monkeypatch.setattr("src.config.integrations.setup_vllm_env", lambda: None)
    monkeypatch.setattr("src.config.integrations._configure_embeddings", lambda: None)
    monkeypatch.setattr(
        "src.config.integrations._configure_structured_outputs", lambda: None
    )
    monkeypatch.setattr(
        "src.config.integrations._configure_context_settings", lambda: None
    )
    monkeypatch.setattr(
        "src.config.integrations.get_settings_embed_model", lambda: sentinel_llm
    )
    monkeypatch.setattr(
        "src.config.integrations.settings._validate_endpoints_security", lambda: None
    )

    original_backend = global_settings.llm_backend
    original_base_url = getattr(global_settings, "lmstudio_base_url", None)
    original_model = global_settings.model

    global_settings.llm_backend = "lmstudio"  # type: ignore[assignment]
    global_settings.lmstudio_base_url = "http://localhost:1234/v1"  # type: ignore[assignment]
    global_settings.model = "Hermes-2-Pro-Llama-3-8B"  # type: ignore[assignment]

    try:
        Settings._llm = None
        initialize_integrations(force_llm=True, force_embed=False)
        assert getattr(Settings, "_llm", None) is sentinel_llm
    finally:
        Settings._llm = original_llm
        global_settings.llm_backend = original_backend  # type: ignore[assignment]
        if original_base_url is not None:
            global_settings.lmstudio_base_url = original_base_url  # type: ignore[assignment]
        global_settings.model = original_model  # type: ignore[assignment]
