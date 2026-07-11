"""Parsing/OCR settings regression tests."""

from __future__ import annotations

import pytest

from src.config.settings import DocMindSettings

pytestmark = pytest.mark.unit


def test_parsing_defaults_are_local_first() -> None:
    cfg = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    assert cfg.parsing.framework == "docling"
    assert cfg.parsing.profile == "cpu_safe"
    assert cfg.ocr.engine == "rapidocr"
    assert cfg.ocr.model_cache_dir == cfg.cache.dir / "models"
    assert cfg.pdf_backend.render_dpi == 200
    assert cfg.pdf_backend.min_text_chars_per_page == 24
    assert cfg.parsing.max_pages == 500
    assert cfg.parsing.max_render_pixels == 40_000_000
    assert cfg.parsing.max_total_text_chars == 10_000_000
    assert cfg.parsing.parse_timeout_seconds == 300.0
    assert cfg.parsing.ocrmypdf_timeout_seconds == 300.0
    assert cfg.security.allow_remote_endpoints is False
    assert cfg.security.trust_remote_code is False


def test_legacy_profile_is_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCMIND_PARSING__PROFILE", "legacy_transition")

    with pytest.raises(ValueError, match=r"parsing\.profile"):
        DocMindSettings(_env_file=None)  # type: ignore[arg-type]


def test_non_rapidocr_engine_is_not_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCMIND_OCR__ENGINE", "paddleocr")

    with pytest.raises(ValueError, match=r"ocr\.engine"):
        DocMindSettings(_env_file=None)  # type: ignore[arg-type]
