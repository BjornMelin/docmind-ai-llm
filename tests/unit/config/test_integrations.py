"""Unit tests for src.config.integrations helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config.integrations import (
    _configure_context_settings,
    _configure_embeddings,
    _configure_llm,
    _configure_structured_outputs,
    _should_configure_embeddings,
    _should_configure_llm,
    setup_llamaindex,
)


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    """Provide a lightweight stand-in for llama_index.core.Settings."""
    fake = SimpleNamespace(
        llm=None,
        embed_model=None,
        context_window=None,
        num_output=None,
        guided_json_enabled=False,
    )
    monkeypatch.setattr("src.config.integrations.Settings", fake)
    return fake


@pytest.mark.unit
def test_should_configure_llm_checks_existing(fake_settings):
    fake_settings.llm = object()
    assert not _should_configure_llm(force_llm=False)
    assert _should_configure_llm(force_llm=True)
    fake_settings.llm = None
    assert _should_configure_llm(force_llm=False)


@pytest.mark.unit
def test_configure_llm_happy_path(monkeypatch, fake_settings):
    sentinel = object()
    monkeypatch.setattr("src.config.integrations.settings", MagicMock())
    with (
        patch("src.config.integrations.settings._validate_endpoints_security"),
        patch("src.config.integrations.build_llm", return_value=sentinel),
    ):
        _configure_llm()
        assert fake_settings.llm is sentinel


@pytest.mark.unit
def test_configure_llm_security_failure(monkeypatch, fake_settings, caplog):
    fake_settings.llm = object()
    monkeypatch.setattr("src.config.integrations.settings", MagicMock())
    with (
        patch(
            "src.config.integrations.settings._validate_endpoints_security",
            side_effect=ValueError("blocked"),
        ),
        patch("src.config.integrations.build_llm"),
    ):
        _configure_llm()
        assert fake_settings.llm is None
        assert any("blocked" in record.message for record in caplog.records)


@pytest.mark.unit
def test_should_configure_embeddings_checks_existing(fake_settings):
    fake_settings.embed_model = object()
    assert not _should_configure_embeddings(force_embed=False)
    assert _should_configure_embeddings(force_embed=True)
    fake_settings.embed_model = None
    assert _should_configure_embeddings(force_embed=False)


@pytest.mark.unit
def test_configure_embeddings_handles_failure(monkeypatch, fake_settings, caplog):
    fake_settings.embed_model = None
    monkeypatch.setattr("src.config.integrations.settings", MagicMock())
    with patch(
        "src.config.integrations.HuggingFaceEmbedding",
        side_effect=RuntimeError("embed error"),
    ):
        _configure_embeddings()
        assert fake_settings.embed_model is None
        assert any("embed error" in record.message for record in caplog.records)


@pytest.mark.unit
def test_configure_context_settings_sets_values(monkeypatch, fake_settings):
    mock_settings = MagicMock(
        context_window=4096,
        vllm=MagicMock(context_window=8192, max_tokens=256),
        llm_context_window_max=5000,
    )
    monkeypatch.setattr("src.config.integrations.settings", mock_settings)
    _configure_context_settings()
    assert fake_settings.context_window == 4096
    assert fake_settings.num_output == 256


@pytest.mark.unit
def test_configure_structured_outputs_sets_flag(monkeypatch):
    mock_settings = MagicMock(llm_backend="vllm")
    monkeypatch.setattr("src.config.integrations.settings", mock_settings)
    _configure_structured_outputs()
    assert mock_settings.guided_json_enabled is True


@pytest.mark.unit
def test_setup_llamaindex_invokes_helpers(monkeypatch):
    with (
        patch("src.config.integrations._should_configure_llm", return_value=True),
        patch("src.config.integrations._configure_llm") as mock_llm,
        patch(
            "src.config.integrations._should_configure_embeddings", return_value=True
        ),
        patch("src.config.integrations._configure_embeddings") as mock_emb,
        patch("src.config.integrations._configure_context_settings") as mock_ctx,
        patch("src.config.integrations._configure_structured_outputs") as mock_struct,
    ):
        setup_llamaindex(force_llm=False, force_embed=False)
        mock_llm.assert_called_once_with()
        mock_emb.assert_called_once_with()
        mock_ctx.assert_called_once_with()
        mock_struct.assert_called_once_with()
