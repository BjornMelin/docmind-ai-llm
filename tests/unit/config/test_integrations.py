"""Unit tests for src.config.integrations helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config.integrations import (
    _configure_context_settings,
    _configure_embeddings,
    _configure_llm,
    _should_configure_embeddings,
    _should_configure_llm,
    is_embedding_ready,
    setup_llamaindex,
)
from src.config.settings import DocMindSettings


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    """Provide a lightweight stand-in for llama_index.core.Settings."""

    class _FakeSettings:
        def __init__(self) -> None:
            self._llm = None
            self._embed_model = None
            self.context_window = None
            self.num_output = None

        @property
        def llm(self):  # type: ignore[no-untyped-def]
            return self._llm

        @llm.setter
        def llm(self, value):  # type: ignore[no-untyped-def]
            self._llm = value

        @property
        def embed_model(self):  # type: ignore[no-untyped-def]
            return self._embed_model

        @embed_model.setter
        def embed_model(self, value):  # type: ignore[no-untyped-def]
            self._embed_model = value

    fake = _FakeSettings()
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
def test_should_configure_llm_reads_real_backing_slot_without_lazy_resolution(
    monkeypatch,
):
    """Inspect real LlamaIndex state without invoking its default resolver."""
    from llama_index.core import Settings as LlamaIndexSettings

    from src.config import integrations as integ

    resolver = MagicMock(side_effect=AssertionError("lazy resolver called"))
    monkeypatch.setattr(integ, "Settings", LlamaIndexSettings)
    monkeypatch.setattr("llama_index.core.settings.resolve_llm", resolver)
    monkeypatch.setattr(LlamaIndexSettings, "_llm", None)

    assert integ._should_configure_llm(force_llm=False)
    resolver.assert_not_called()

    configured = MagicMock()
    monkeypatch.setattr(LlamaIndexSettings, "_llm", configured)
    assert not integ._should_configure_llm(force_llm=False)
    resolver.assert_not_called()


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
    previous = object()
    fake_settings.llm = previous
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
            with pytest.raises(ValueError, match="blocked"):
                _configure_llm()
        finally:
            loguru_logger.remove(sink_id)

        assert fake_settings.llm is previous
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
    previous = object()
    fake_settings.embed_model = previous
    config_owner = DocMindSettings(
        enable_gpu_acceleration=False,
        _env_file=None,  # type: ignore[arg-type]
    )
    monkeypatch.setattr("src.config.integrations.settings", config_owner)

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
            with pytest.raises(RuntimeError, match="embed error"):
                _configure_embeddings()
        finally:
            loguru_logger.remove(sink_id)

        assert fake_settings.embed_model is previous
        assert any("Could not configure embeddings" in m for m in messages)
        assert any("[redacted:" in m for m in messages)
        assert not any("embed error" in m for m in messages)


@pytest.mark.unit
def test_configure_embeddings_passes_native_huggingface_options(
    monkeypatch, fake_settings
):
    """Forward the canonical text settings through native adapter keywords."""
    from llama_index.core.base.embeddings.base import BaseEmbedding

    from src.config.embedding_defaults import (
        DEFAULT_BGE_M3_MODEL_ID,
        DEFAULT_BGE_M3_MODEL_REVISION,
    )

    embed_model = MagicMock(spec=BaseEmbedding)
    constructor = MagicMock(return_value=embed_model)
    config_owner = DocMindSettings(
        enable_gpu_acceleration=False,
        embedding={
            "batch_size_text_cpu": 7,
            "cache_folder": "/models/cache",
        },
        _env_file=None,  # type: ignore[arg-type]
    )
    monkeypatch.setattr("src.config.integrations.settings", config_owner)
    monkeypatch.setattr("src.config.integrations.HuggingFaceEmbedding", constructor)

    _configure_embeddings()

    constructor.assert_called_once_with(
        model_name=DEFAULT_BGE_M3_MODEL_ID,
        device="cpu",
        max_length=8192,
        normalize=True,
        embed_batch_size=7,
        trust_remote_code=False,
        local_files_only=True,
        cache_folder="/models/cache",
        revision=DEFAULT_BGE_M3_MODEL_REVISION,
    )
    assert fake_settings._embed_model is embed_model


@pytest.mark.unit
def test_embedding_failure_preserves_real_backing_slot(monkeypatch):
    """A load failure preserves the previously configured embedding."""
    from llama_index.core import Settings as LlamaIndexSettings
    from llama_index.core.embeddings import MockEmbedding

    from src.config import integrations as integ

    config_owner = DocMindSettings(
        enable_gpu_acceleration=False,
        _env_file=None,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(integ, "Settings", LlamaIndexSettings)
    monkeypatch.setattr(integ, "settings", config_owner)
    monkeypatch.setattr(
        integ,
        "HuggingFaceEmbedding",
        MagicMock(side_effect=OSError("model missing")),
    )
    monkeypatch.setattr(
        LlamaIndexSettings,
        "_embed_model",
        MockEmbedding(embed_dim=1),
    )

    with pytest.raises(OSError, match="model missing"):
        integ._configure_embeddings()

    assert isinstance(LlamaIndexSettings._embed_model, MockEmbedding)
    assert not integ.is_embedding_ready()


@pytest.mark.unit
def test_embedding_readiness_rejects_llamaindex_mock(monkeypatch, fake_settings):
    """Only a real BaseEmbedding implementation is production-ready."""
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.embeddings import MockEmbedding

    fake_settings._embed_model = MockEmbedding(embed_dim=1024)
    assert not is_embedding_ready()

    fake_settings._embed_model = MagicMock(spec=BaseEmbedding)
    assert is_embedding_ready()


@pytest.mark.unit
def test_configure_context_settings_sets_values(monkeypatch, fake_settings):
    mock_settings = MagicMock(
        llm_request=MagicMock(context_window=8192, max_output_tokens=256),
    )
    monkeypatch.setattr("src.config.integrations.settings", mock_settings)
    _configure_context_settings()
    assert fake_settings.context_window == 8192
    assert fake_settings.num_output == 256


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
    ):
        setup_llamaindex(force_llm=False, force_embed=False)
        mock_llm.assert_called_once_with()
        mock_emb.assert_called_once_with()
        mock_ctx.assert_called_once_with()


@pytest.mark.unit
def test_startup_init_creates_directories(monkeypatch, tmp_path):
    from src.config import integrations as integ

    cfg = DocMindSettings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        chat={"sqlite_path": tmp_path / "chat" / "chat.db"},
        log_file=tmp_path / "logs" / "app.log",
        llm_backend="vllm",
        llm_request_timeout_seconds=30,
        retrieval={"enable_server_hybrid": False, "fusion_mode": "rrf"},
        _env_file=None,  # type: ignore[arg-type]
    )
    tracer = MagicMock()
    metrics = MagicMock()
    monkeypatch.setattr("src.config.integrations.setup_tracing", tracer)
    monkeypatch.setattr("src.config.integrations.setup_metrics", metrics)
    monkeypatch.setattr("loguru.logger", MagicMock())

    integ.startup_init(cfg)

    assert (tmp_path / "data").exists()
    assert (tmp_path / "cache").exists()
    assert (tmp_path / "chat").exists()
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


@pytest.mark.unit
def test_get_settings_embed_model_handles_lazy_default_import_error(monkeypatch):
    from src.config import integrations as integ

    class _MissingDefaultAdapter:
        _embed_model = None

        @property
        def embed_model(self):  # type: ignore[no-untyped-def]
            raise AssertionError("lazy default must not be inspected")

    monkeypatch.setattr("src.config.integrations.Settings", _MissingDefaultAdapter())

    assert integ.get_settings_embed_model() is None
