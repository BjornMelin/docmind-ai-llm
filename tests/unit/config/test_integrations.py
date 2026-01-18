"""Unit tests for src.config.integrations helpers."""

from __future__ import annotations

import os
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
def test_configure_llm_security_failure(monkeypatch, fake_settings):
    fake_settings.llm = object()
    monkeypatch.setattr("src.config.integrations.settings", MagicMock())

    from loguru import logger as loguru_logger

    messages: list[str] = []

    def _sink(msg):  # type: ignore[no-untyped-def]
        messages.append(str(msg.record.get("message", "")))

    sink_id = loguru_logger.add(_sink, level="WARNING")
    with (
        patch(
            "src.config.integrations.settings._validate_endpoints_security",
            side_effect=ValueError("blocked"),
        ),
        patch("src.config.integrations.build_llm"),
    ):
        try:
            _configure_llm()
        finally:
            loguru_logger.remove(sink_id)

        assert fake_settings.llm is None
        assert any("blocked" in m for m in messages)


@pytest.mark.unit
def test_should_configure_embeddings_checks_existing(fake_settings):
    fake_settings.embed_model = object()
    assert not _should_configure_embeddings(force_embed=False)
    assert _should_configure_embeddings(force_embed=True)
    fake_settings.embed_model = None
    assert _should_configure_embeddings(force_embed=False)


@pytest.mark.unit
def test_configure_embeddings_handles_failure(monkeypatch, fake_settings):
    fake_settings.embed_model = None
    monkeypatch.setattr("src.config.integrations.settings", MagicMock())

    from loguru import logger as loguru_logger

    messages: list[str] = []

    def _sink(msg):  # type: ignore[no-untyped-def]
        messages.append(str(msg.record.get("message", "")))

    sink_id = loguru_logger.add(_sink, level="WARNING")
    with patch(
        "src.config.integrations.HuggingFaceEmbedding",
        side_effect=RuntimeError("embed error"),
    ):
        try:
            _configure_embeddings()
        finally:
            loguru_logger.remove(sink_id)

        assert fake_settings.embed_model is None
        assert any("Could not configure embeddings" in m for m in messages)
        assert any("[redacted:" in m for m in messages)
        assert not any("embed error" in m for m in messages)


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


@pytest.mark.unit
def test_setup_vllm_env_sets_missing_only(monkeypatch):
    from src.config import integrations as integ

    mock_settings = MagicMock()
    mock_settings.get_vllm_env_vars.return_value = {
        "VLLM_FP8_OPT": "1",
        "VLLM_EXISTING": "value",
    }
    monkeypatch.setattr("src.config.integrations.settings", mock_settings)
    monkeypatch.setenv("VLLM_EXISTING", "keep")
    monkeypatch.delenv("VLLM_FP8_OPT", raising=False)

    integ.setup_vllm_env()

    assert os.environ["VLLM_FP8_OPT"] == "1"
    assert os.environ["VLLM_EXISTING"] == "keep"


@pytest.mark.unit
def test_get_vllm_server_command_includes_chunked_prefill(monkeypatch):
    from src.config import integrations as integ

    mock_settings = SimpleNamespace(
        vllm=SimpleNamespace(
            model="qwen",
            context_window=8192,
            kv_cache_dtype="fp8",
            gpu_memory_utilization=0.8,
            max_num_seqs=8,
            max_num_batched_tokens=512,
            enable_chunked_prefill=True,
        )
    )
    monkeypatch.setattr("src.config.integrations.settings", mock_settings)

    cmd = integ.get_vllm_server_command()

    assert cmd[:3] == ["vllm", "serve", "qwen"]
    assert "--enable-chunked-prefill" in cmd


@pytest.mark.unit
def test_startup_init_creates_directories(monkeypatch, tmp_path):
    from src.config import integrations as integ

    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        log_file=tmp_path / "logs" / "app.log",
        database=SimpleNamespace(sqlite_db_path=tmp_path / "db" / "doc.db"),
        observability=SimpleNamespace(enabled=False, endpoint=None, sampling_ratio=1.0),
        llm_backend="vllm",
        backend_base_url_normalized="http://localhost",
        llm_request_timeout_seconds=30,
        retrieval=SimpleNamespace(enable_server_hybrid=False, fusion_mode="rrf"),
    )
    tracer = MagicMock()
    metrics = MagicMock()
    monkeypatch.setattr("src.config.integrations.setup_tracing", tracer)
    monkeypatch.setattr("src.config.integrations.setup_metrics", metrics)
    monkeypatch.setattr("loguru.logger", MagicMock())

    integ.startup_init(cfg)

    assert (tmp_path / "data").exists()
    assert (tmp_path / "cache").exists()
    tracer.assert_called_once_with(cfg)
    metrics.assert_called_once_with(cfg)


@pytest.mark.unit
def test_get_settings_embed_model_handles_attribute_error(monkeypatch):
    from src.config import integrations as integ

    class _WeirdSettings:
        def __getattr__(self, name: str):  # pragma: no cover - simple stub
            raise AttributeError(name)

    monkeypatch.setattr("src.config.integrations.Settings", _WeirdSettings())
    assert integ.get_settings_embed_model() is None
