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
    is_embedding_ready,
    setup_llamaindex,
)


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    """Provide a lightweight stand-in for llama_index.core.Settings."""

    class _FakeSettings:
        def __init__(self) -> None:
            self._llm = None
            self._embed_model = None
            self.context_window = None
            self.num_output = None
            self.guided_json_enabled = False

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
def test_configure_embeddings_passes_native_huggingface_options(
    monkeypatch, fake_settings
):
    """Forward the canonical text settings through native adapter keywords."""
    from llama_index.core.base.embeddings.base import BaseEmbedding

    from src.config.embedding_defaults import (
        BGE_M3_EMBEDDING_DIMENSION,
        DEFAULT_BGE_M3_MODEL_ID,
        DEFAULT_BGE_M3_MODEL_REVISION,
    )

    embed_model = MagicMock(spec=BaseEmbedding)
    constructor = MagicMock(return_value=embed_model)
    config = {
        "batch_size_text": 7,
        "cache_folder": "/models/cache",
        "device": "cpu",
        "dimension": BGE_M3_EMBEDDING_DIMENSION,
        "local_files_only": True,
        "local_model_path": None,
        "max_length": 8192,
        "model_id": DEFAULT_BGE_M3_MODEL_ID,
        "model_name": DEFAULT_BGE_M3_MODEL_ID,
        "model_revision": DEFAULT_BGE_M3_MODEL_REVISION,
        "normalize_text": True,
        "trust_remote_code": False,
    }
    config_owner = SimpleNamespace(get_embedding_config=lambda: config)
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
def test_embedding_failure_clears_real_slot_without_mock_fallback(monkeypatch):
    """A load failure leaves the real backing slot empty and not ready."""
    from llama_index.core import Settings as LlamaIndexSettings
    from llama_index.core.embeddings import MockEmbedding

    from src.config import integrations as integ
    from src.config.embedding_defaults import DEFAULT_BGE_M3_MODEL_ID

    config_owner = SimpleNamespace(
        get_embedding_config=lambda: {
            "dimension": 1024,
            "model_id": DEFAULT_BGE_M3_MODEL_ID,
            "model_name": DEFAULT_BGE_M3_MODEL_ID,
        }
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

    integ._configure_embeddings()

    assert LlamaIndexSettings._embed_model is None
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
        cache=SimpleNamespace(dir=tmp_path / "cache"),
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
