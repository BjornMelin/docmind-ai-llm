"""Parser production-readiness tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import DocMindSettings
from src.processing.parsing import health

pytestmark = pytest.mark.unit


def _settings(model_cache_dir: Path) -> DocMindSettings:
    return DocMindSettings.model_validate(
        {"parsing": {"model_cache_dir": model_cache_dir}}
    )


def test_pdf_dependencies_ready_requires_every_core_package_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _package_health(
        package: str,
        *,
        module: str | None = None,
    ) -> dict[str, object]:
        del module
        return {
            "available": True,
            "version": "1.0",
            "importable": package != "onnxruntime",
        }

    monkeypatch.setattr(health, "_package_health", _package_health)
    monkeypatch.setattr(
        health, "cached_model_directory_issues", lambda _root, _bundle: {}
    )
    monkeypatch.setattr(
        health,
        "ocrmypdf_health",
        lambda: {"ocrmypdf": False, "tesseract": False},
    )

    result = health.parser_health(_settings(tmp_path))

    assert result["pdf_dependencies_ready"] is False
    assert result["onnxruntime"]["importable"] is False


def test_pdf_dependencies_ready_requires_models_but_not_optional_ocrmypdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        health,
        "_package_health",
        lambda _package, **_kwargs: {
            "available": True,
            "version": "1.0",
            "importable": True,
        },
    )
    monkeypatch.setattr(
        health, "cached_model_directory_issues", lambda _root, _bundle: {}
    )
    monkeypatch.setattr(
        health,
        "ocrmypdf_health",
        lambda: {"ocrmypdf": False, "tesseract": False},
    )

    ready = health.parser_health(_settings(tmp_path))
    monkeypatch.setattr(
        health,
        "cached_model_directory_issues",
        lambda _root, _bundle: {"model.safetensors": "SHA-256 mismatch"},
    )
    missing_model = health.parser_health(_settings(tmp_path))

    assert ready["pdf_dependencies_ready"] is True
    assert ready["rapidocr"]["dependencies_ready"] is True
    assert missing_model["pdf_dependencies_ready"] is False
    assert missing_model["docling"]["models_ready"] is False
    assert missing_model["docling"]["model_issues"] == {
        "model.safetensors": "SHA-256 mismatch"
    }


def test_prefetch_command_uses_configured_cache_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_cache = tmp_path / "models with spaces"
    monkeypatch.setattr(
        health,
        "_package_health",
        lambda _package, **_kwargs: {
            "available": True,
            "version": "1.0",
            "importable": True,
        },
    )
    monkeypatch.setattr(
        health, "cached_model_directory_issues", lambda _root, _bundle: {}
    )
    monkeypatch.setattr(
        health,
        "ocrmypdf_health",
        lambda: {"ocrmypdf": False, "tesseract": False},
    )

    result = health.parser_health(_settings(model_cache))

    assert result["prefetch_command"].endswith(f"--parser-cache-dir '{model_cache}'")
