"""Settings security regression tests."""

from __future__ import annotations

import pytest
from _pytest.monkeypatch import MonkeyPatch

from src.config.settings import DocMindSettings


def test_rejects_link_local_metadata_ip_for_ollama_base_url(
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setenv("DOCMIND_OLLAMA_BASE_URL", "http://169.254.169.254")
    with pytest.raises(Exception, match="Remote endpoints are disabled"):
        DocMindSettings(_env_file=None)  # type: ignore[arg-type]
