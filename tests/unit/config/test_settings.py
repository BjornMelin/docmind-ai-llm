"""Settings security regression tests."""

from __future__ import annotations

import pytest
from _pytest.monkeypatch import MonkeyPatch

from src.config.embedding_defaults import (
    BGE_M3_EMBEDDING_DIMENSION,
    DEFAULT_BGE_M3_MODEL_ID,
    DEFAULT_BGE_M3_MODEL_REVISION,
)
from src.config.settings import DocMindSettings, EmbeddingConfig


@pytest.mark.unit
def test_rejects_link_local_metadata_ip_for_ollama_base_url(
    monkeypatch: MonkeyPatch,
) -> None:
    """Reject link-local Ollama base URLs when remote endpoints disallowed."""
    monkeypatch.setenv("DOCMIND_OLLAMA_BASE_URL", "http://169.254.169.254")
    with pytest.raises(ValueError, match="Remote endpoints are disabled"):
        DocMindSettings(_env_file=None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_custom_siglip_model_does_not_inherit_default_revision(
    monkeypatch: MonkeyPatch,
) -> None:
    """Custom SigLIP model IDs load unpinned unless a revision is configured."""
    monkeypatch.setenv("DOCMIND_EMBEDDING__SIGLIP_MODEL_ID", "example/custom-siglip")
    monkeypatch.delenv("DOCMIND_EMBEDDING__SIGLIP_MODEL_REVISION", raising=False)

    settings = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    assert settings.embedding.siglip_model_id == "example/custom-siglip"
    assert settings.embedding.siglip_model_revision is None


@pytest.mark.unit
def test_embedding_config_uses_hardware_aware_device(
    monkeypatch: MonkeyPatch,
) -> None:
    """Resolve enabled acceleration through the canonical hardware selector."""
    monkeypatch.setattr("src.utils.core.select_device", lambda _prefer: "cpu")
    settings = DocMindSettings(
        enable_gpu_acceleration=True,
        _env_file=None,  # type: ignore[arg-type]
    )

    config = settings.get_embedding_config()

    assert config["device"] == "cpu"
    assert config["siglip_model_id"] == "google/siglip-base-patch16-224"
    assert "embed_device" not in config


@pytest.mark.unit
def test_embedding_config_resolves_pinned_bge_m3_contract() -> None:
    """Expose one pinned 1024D BGE-M3 runtime contract."""
    settings = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    config = settings.get_embedding_config()

    assert config["model_id"] == DEFAULT_BGE_M3_MODEL_ID
    assert config["model_name"] == DEFAULT_BGE_M3_MODEL_ID
    assert config["model_revision"] == DEFAULT_BGE_M3_MODEL_REVISION
    assert config["dimension"] == BGE_M3_EMBEDDING_DIMENSION
    assert config["cache_folder"] == "models_cache"
    assert config["local_files_only"] is True


@pytest.mark.unit
def test_embedding_config_prefers_explicit_local_model_path(tmp_path) -> None:
    """A local snapshot replaces the Hub identifier and revision at runtime."""
    settings = DocMindSettings(
        embedding={"local_model_path": tmp_path},
        _env_file=None,
    )  # type: ignore[arg-type]

    config = settings.get_embedding_config()

    assert config["model_name"] == str(tmp_path)
    assert config["local_model_path"] == str(tmp_path)
    assert config["model_revision"] is None


@pytest.mark.unit
def test_custom_text_model_does_not_inherit_bge_m3_revision() -> None:
    """Custom text models remain unpinned unless configured explicitly."""
    embedding = EmbeddingConfig(model_name="example/custom-text-model")

    assert embedding.model_revision is None


@pytest.mark.unit
def test_bge_m3_rejects_noncanonical_dimension() -> None:
    """Prevent Qdrant schema drift for the canonical BGE-M3 model."""
    with pytest.raises(ValueError, match="requires embedding dimension 1024"):
        EmbeddingConfig(
            model_name=DEFAULT_BGE_M3_MODEL_ID,
            dimension=768,
        )
