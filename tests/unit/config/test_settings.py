"""Settings security regression tests."""

from __future__ import annotations

import pytest
from _pytest.monkeypatch import MonkeyPatch

from src.config.settings import DocMindSettings


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

    settings = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    assert settings.embedding.siglip_model_id == "example/custom-siglip"
    assert settings.embedding.siglip_model_revision is None
